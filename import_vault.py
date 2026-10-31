"""Import an Obsidian vault into Neo4j as a knowledge graph.

Read-only on the vault. Builds:
  (:Note)-[:LINKS_TO]->(:Note)      from [[wikilinks]] and ![[embeds]] of notes
  (:Note)-[:TAGGED]->(:Tag)         from #tags and front-matter tags
  (:Note)-[:IN_FOLDER]->(:Folder)   and (:Folder)-[:SUBFOLDER_OF]->(:Folder)
Links to notes that don't exist become (:Note:Missing) placeholder nodes.

Files synced as online-only placeholders (Google Drive / iCloud on macOS, the "dataless" flag) are skipped
without triggering a download unless --fetch is given; slow reads time out and are skipped too.

Usage:
  python import_vault.py --dry-run
  python import_vault.py                    # the vault in VAULT_PATH (default: sample_vault/)
  python import_vault.py --folder Clients   # one subfolder, for test runs
  python import_vault.py --prune
"""

import argparse
import concurrent.futures as cf
import json
import os
import posixpath
import re
import stat
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

import config

SKIP_DIRS = {".obsidian", ".trash", ".git", ".stfolder", "node_modules",
             "copilot"}  # Obsidian Copilot keeps its custom prompts as notes: plugin config, not knowledge
FM_MIN_NOTES = 50  # front-matter keys used by fewer notes (or 10% of a small vault) go into one fm_other JSON property
FM_DROP = {"fm_generation_complete", "fm_default_model", "fm_models", "fm_contentHash", "fm_chat_id"}  # chat-export plumbing
SF_DATALESS = getattr(stat, "SF_DATALESS", 0x40000000)
READ_TIMEOUT_S = 15
FETCH_TIMEOUT_S = 120
BATCH = 500

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(\n|\Z)", re.DOTALL)
WIKILINK_RE = re.compile(r"!?\[\[([^\[\]]+?)\]\]")
TAG_RE = re.compile(r"(?:(?<=\s)|^)#([A-Za-z0-9_\-/]*[A-Za-z][A-Za-z0-9_\-/]*)")
CODE_RE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)


def birthtime(st):
    """File creation time where the OS records it (macOS), else last modification."""
    return getattr(st, "st_birthtime", st.st_mtime)


def is_online_only(p):
    try:
        return bool(os.stat(p).st_flags & SF_DATALESS)
    except (OSError, AttributeError):
        return False


def iter_markdown(root, subfolder=None):
    start = root / subfolder if subfolder else root
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for f in filenames:
            if f.lower().endswith(".md"):
                yield Path(dirpath) / f


def parse_frontmatter(text):
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        data = yaml.safe_load(m.group(1)) or {}
        if not isinstance(data, dict):
            data = {}
    except yaml.YAMLError:
        data = {}
    return data, text[m.end():]


def as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[,\s]+", value) if v.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return [str(value)]


def scalar_props(fm):
    """Front-matter scalars (and lists of scalars) as fm_* properties."""
    props = {}
    for k, v in fm.items():
        key = "fm_" + re.sub(r"[^A-Za-z0-9_]", "_", str(k))
        if isinstance(v, (str, int, float, bool)):
            props[key] = v
        elif isinstance(v, (datetime,)) or hasattr(v, "isoformat"):
            props[key] = v.isoformat()
        elif isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v):
            props[key] = [str(x) for x in v]
    return props


def parse_note(root, path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    fm, body = parse_frontmatter(raw)
    clean = CODE_RE.sub("", body)

    tags = {t.lstrip("#").lower() for t in as_list(fm.get("tags") or fm.get("tag"))}
    tags |= {t.lower().rstrip("/") for t in TAG_RE.findall(clean)}

    links = []
    for target in WIKILINK_RE.findall(clean):
        target = target.split("|", 1)[0].split("#", 1)[0].split("^", 1)[0].strip()
        if not target:
            continue
        suffix = Path(target).suffix.lower()
        if suffix and suffix != ".md":
            continue  # image/pdf/etc. attachment
        links.append(target[:-3] if suffix == ".md" else target)

    st = path.stat()
    rel = path.relative_to(root).as_posix()
    return {
        "path": rel,
        "title": path.stem,
        "folder": Path(rel).parent.as_posix() if "/" in rel else "",
        "aliases": as_list(fm.get("aliases") or fm.get("alias")),
        "words": len(clean.split()),
        "created": datetime.fromtimestamp(birthtime(st), timezone.utc).isoformat(),
        "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "props": scalar_props(fm),
        "tags": sorted(t for t in tags if t),
        "links": links,
    }


def read_all(root, files, fetch=False, workers=8):
    """With fetch=True, online-only files are read anyway, which makes Drive download just that file."""
    notes, skipped = [], []
    timeout = FETCH_TIMEOUT_S if fetch else READ_TIMEOUT_S
    pool = cf.ThreadPoolExecutor(max_workers=workers)
    futures = {}
    for p in files:
        if not fetch and is_online_only(p):
            skipped.append({"path": p.relative_to(root).as_posix(), "reason": "online-only"})
            continue
        futures[pool.submit(parse_note, root, p)] = p
    for i, (fut, p) in enumerate(futures.items(), 1):
        rel = p.relative_to(root).as_posix()
        if i % 500 == 0:
            print(f"  read {i}/{len(futures)}", flush=True)
        try:
            notes.append(fut.result(timeout=timeout))
        except cf.TimeoutError:
            skipped.append({"path": rel, "reason": f"read timed out >{timeout}s"})
        except Exception as e:  # noqa: BLE001 — log and move on, one bad file shouldn't stop the run
            skipped.append({"path": rel, "reason": f"{type(e).__name__}: {e}"})
    pool.shutdown(wait=False, cancel_futures=True)
    return notes, skipped


def build_resolver(all_paths):
    """Obsidian-style resolution: by basename, or by path suffix when the link has a '/'."""
    by_name = defaultdict(list)
    for p in all_paths:
        by_name[Path(p).stem.lower()].append(p)

    def resolve(target, aliases):
        t = target.lower()
        if "/" in t:
            want = t + ".md"
            hits = [p for p in all_paths if p.lower().endswith(want)]
        else:
            hits = by_name.get(t) or aliases.get(t, [])
        return min(hits, key=len) if hits else None

    return resolve


def tidy_frontmatter(notes):
    """Keep common front-matter keys as properties, fold rare ones into fm_other (JSON), drop plumbing."""
    counts = defaultdict(int)
    for n in notes:
        for k in n["props"]:
            counts[k] += 1
    threshold = min(FM_MIN_NOTES, max(2, len(notes) // 10))
    for n in notes:
        keep, other = {}, {}
        for k, v in n["props"].items():
            if k in FM_DROP:
                continue
            if counts[k] >= threshold:
                keep[k] = v
            else:
                other[k[3:]] = v
        if other:
            keep["fm_other"] = json.dumps(other, ensure_ascii=False, default=str)
        n["props"] = keep


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def write_graph(driver, database, notes, resolve, aliases, prune_scope, all_folders):
    edges, missing, folder_refs = [], set(), []
    for n in notes:
        for target in set(n["links"]):
            if target.startswith(("./", "../")):
                target = posixpath.normpath(posixpath.join(n["folder"], target)).lstrip("./")
            # Obsidian also resolves "sub/note" and folder names relative to the linking note's folder.
            local = posixpath.join(n["folder"], target) if n["folder"] else target
            folder_hit = next((f for f in (local, target) if f in all_folders), None)
            if folder_hit:
                folder_refs.append({"src": n["path"], "dst": folder_hit})
                continue
            dst = (resolve(local, aliases) if "/" in target else None) or resolve(target, aliases)
            if dst is None:
                dst = "missing:" + target.lower()
                missing.add((dst, target))
            if dst != n["path"]:
                edges.append({"src": n["path"], "dst": dst})

    folders = set()
    for n in notes:
        parts = n["folder"].split("/") if n["folder"] else []
        for i in range(1, len(parts) + 1):
            folders.add("/".join(parts[:i]))

    with driver.session(database=database) as s:
        for q in (
            "CREATE CONSTRAINT note_path IF NOT EXISTS FOR (n:Note) REQUIRE n.path IS UNIQUE",
            "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT folder_path IF NOT EXISTS FOR (f:Folder) REQUIRE f.path IS UNIQUE",
            "CREATE INDEX note_title IF NOT EXISTS FOR (n:Note) ON (n.title)",
        ):
            s.run(q).consume()

        folder_rows = [{"path": f, "name": f.rsplit("/", 1)[-1],
                        "parent": f.rsplit("/", 1)[0] if "/" in f else None} for f in sorted(folders)]
        for batch in chunks(folder_rows, BATCH):
            s.run("""
                UNWIND $rows AS r
                MERGE (f:Folder {path: r.path}) SET f.name = r.name
                WITH f, r WHERE r.parent IS NOT NULL
                MERGE (p:Folder {path: r.parent})
                MERGE (f)-[:SUBFOLDER_OF]->(p)
            """, rows=batch).consume()

        for batch in chunks(notes, BATCH):
            s.run("""
                UNWIND $rows AS r
                MERGE (n:Note {path: r.path})
                REMOVE n:Missing
                FOREACH (k IN [k IN keys(n) WHERE k STARTS WITH 'fm_'] | REMOVE n[k])
                SET n += r.props, n.title = r.title, n.folder = r.folder, n.aliases = r.aliases,
                    n.words = r.words, n.created = datetime(r.created), n.modified = datetime(r.modified),
                    n.imported_at = datetime()
                WITH n, r
                OPTIONAL MATCH (n)-[old:LINKS_TO|TAGGED|IN_FOLDER]->()
                DELETE old
                WITH DISTINCT n, r
                FOREACH (_ IN CASE WHEN r.folder <> '' THEN [1] ELSE [] END |
                    MERGE (f:Folder {path: r.folder}) MERGE (n)-[:IN_FOLDER]->(f))
                FOREACH (t IN r.tags | MERGE (tg:Tag {name: t}) MERGE (n)-[:TAGGED]->(tg))
            """, rows=batch).consume()

        rows = [{"path": p, "title": t} for p, t in sorted(missing)]
        for batch in chunks(rows, BATCH):
            s.run("""
                UNWIND $rows AS r
                MERGE (m:Note {path: r.path})
                ON CREATE SET m:Missing, m.title = r.title
            """, rows=batch).consume()

        for batch in chunks(edges, BATCH * 4):
            s.run("""
                UNWIND $rows AS r
                MATCH (a:Note {path: r.src}), (b:Note {path: r.dst})
                MERGE (a)-[:LINKS_TO]->(b)
            """, rows=batch).consume()

        for batch in chunks(folder_refs, BATCH * 4):
            s.run("""
                UNWIND $rows AS r
                MATCH (a:Note {path: r.src})
                MERGE (f:Folder {path: r.dst}) ON CREATE SET f.name = split(r.dst, '/')[-1]
                MERGE (a)-[:REFERENCES_FOLDER]->(f)
            """, rows=batch).consume()

        # Navigation/housekeeping notes link to everything; flag them so traversal can skip them.
        s.run("MATCH (n:Note) REMOVE n:Hub").consume()
        s.run("""
            MATCH (n:Note) WHERE NOT n:Missing
              AND (toLower(n.title) IN ['index', 'log', 'claude', 'agents', 'readme', 'home', '000 home']
                   OR toLower(n.title) ENDS WITH ' index')
            SET n:Hub
        """).consume()

        pruned = 0
        if prune_scope is not None:
            keep = [n["path"] for n in notes]
            pruned = s.run("""
                MATCH (n:Note) WHERE NOT n:Missing AND NOT n.path IN $keep
                  AND ($scope = '' OR n.path STARTS WITH $scope)
                DETACH DELETE n RETURN count(*) AS c
            """, keep=keep, scope=prune_scope).single()["c"]
            s.run("MATCH (m:Missing) WHERE NOT ()-[:LINKS_TO]->(m) DELETE m").consume()
            s.run("MATCH (t:Tag) WHERE NOT ()-[:TAGGED]->(t) DELETE t").consume()

    return {"links": len(edges), "folder_refs": len(folder_refs), "missing_targets": len(missing),
            "folders": len(folders), "pruned": pruned}


def main():
    here = config.ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", help="only import this subfolder (relative to the vault), for test runs")
    ap.add_argument("--dry-run", action="store_true", help="parse and report, don't touch Neo4j")
    ap.add_argument("--prune", action="store_true", help="delete notes from the graph that no longer exist in the vault")
    ap.add_argument("--fetch", action="store_true", help="download online-only Google Drive notes (text only) instead of skipping them")
    args = ap.parse_args()

    root = config.vault_path()
    if not root.is_dir():
        sys.exit(f"Vault not found: {root}")

    t0 = time.time()
    files = list(iter_markdown(root, args.folder))
    print(f"Found {len(files)} markdown files{' in ' + args.folder if args.folder else ''}; reading…")
    notes, skipped = read_all(root, files, fetch=args.fetch)
    print(f"Parsed {len(notes)} notes, skipped {len(skipped)} in {time.time() - t0:.0f}s")
    if not args.folder:  # key frequencies only make sense across the whole vault
        tidy_frontmatter(notes)

    # Resolve links against the whole vault's file names (cheap: names only, no reads),
    # so a folder test run still links to notes outside that folder.
    all_paths = [p.relative_to(root).as_posix() for p in (files if not args.folder else iter_markdown(root))]
    aliases = defaultdict(list)
    for n in notes:
        for a in n["aliases"]:
            aliases[a.lower()].append(n["path"])
    resolve = build_resolver(all_paths)

    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "folder": args.folder or "(whole vault)",
        "files_found": len(files),
        "notes_parsed": len(notes),
        "skipped": len(skipped),
        "skipped_by_reason": {},
        "tags": len({t for n in notes for t in n["tags"]}),
        "raw_links": sum(len(n["links"]) for n in notes),
    }
    for s in skipped:
        key = s["reason"].split(":")[0]
        report["skipped_by_reason"][key] = report["skipped_by_reason"].get(key, 0) + 1

    if not args.dry_run:
        driver = config.driver()
        driver.verify_connectivity()
        try:
            prune_scope = None
            if args.prune:
                prune_scope = (args.folder.rstrip("/") + "/") if args.folder else ""
            all_folders = {Path(p).parent.as_posix() for p in all_paths}
            all_folders |= {"/".join(f.split("/")[:i]) for f in list(all_folders) for i in range(1, f.count("/") + 1)}
            report.update(write_graph(driver, config.database(),
                                      notes, resolve, aliases, prune_scope, all_folders))
        finally:
            driver.close()

    report["seconds"] = round(time.time() - t0, 1)
    out = here / "reports"
    out.mkdir(exist_ok=True)
    (out / "last-run.json").write_text(json.dumps(report, indent=2))
    (out / "skipped-files.txt").write_text("\n".join(f"{s['reason']}\t{s['path']}" for s in skipped))
    print(json.dumps(report, indent=2))
    # Threads stuck on a hung Drive read would otherwise keep the process alive.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
