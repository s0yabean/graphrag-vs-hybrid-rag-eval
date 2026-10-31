"""Build the (:Chunk) layer for the vault (or one folder of it), embedded locally.

Chunking rule (see CHUNK_* settings):
  - a note of up to CHUNK_WHOLE_NOTE_MAX tokens is one chunk
  - longer notes split at headings into chunks of up to CHUNK_MAX tokens; sections under
    CHUNK_MIN merge with a neighbour, oversized sections split at paragraphs with a small overlap
  - every chunk carries a "Note title > Heading > Subheading" breadcrumb and the note's date
Stores:
  (:Chunk {id, text, breadcrumb, seq, tokens, note_date, embedding})-[:PART_OF]->(:Note)
  (:Chunk)-[:NEXT_CHUNK]->(:Chunk)   in reading order
  a cosine vector index on Chunk.embedding and full-text indexes on chunk text and note titles.

Embeddings come from config.embed_many: bge-m3 in Ollama by default (what the published eval used), or a small
fastembed model for the quickstart (EMBED_BACKEND=fastembed in .env).

Usage:
  python build_chunks.py                       # the whole vault
  python build_chunks.py --folder Clients      # one folder
  python build_chunks.py --dry-run             # chunk and report sizes only
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import config
from config import EMB_DIMS, EMB_MODEL, embed_many as embed
from import_vault import birthtime, iter_markdown, parse_frontmatter

CHUNK_WHOLE_NOTE_MAX = 1500   # tokens: notes up to this size stay whole
CHUNK_MAX = 1000              # tokens: upper bound for a split chunk
CHUNK_MIN = 100               # tokens: smaller pieces merge with a neighbour
CHUNK_OVERLAP_CHARS = 200     # carried over when a section has to be split mid-way

EMB_BATCH = 16

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
CJK_RE = re.compile(r"[㐀-鿿豈-﫿぀-ヿ가-힯]")
TITLE_DATE_RE = re.compile(r"^(\d{4}-\d{2}(?:-\d{2})?)")


def est_tokens(text):
    """Rough count: CJK characters are ~1 token each, other text ~4 characters per token."""
    cjk = len(CJK_RE.findall(text))
    return cjk + (len(text) - cjk) // 4


def sections(body):
    """Split markdown into (heading_path, text) sections, ignoring '#' lines inside code fences."""
    out, stack, buf, in_code = [], [], [], False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING_RE.match(line)
        if m:
            if "".join(buf).strip():
                out.append((list(stack), "\n".join(buf).strip()))
            level = len(m.group(1))
            stack = [h for h in stack if h[0] < level] + [(level, m.group(2))]
            buf = [line]
        else:
            buf.append(line)
    if "".join(buf).strip():
        out.append((list(stack), "\n".join(buf).strip()))
    return [([h for _, h in path], text) for path, text in out]


def split_long(text):
    """Split an oversized section at paragraph breaks, then lines, then characters."""
    limit_chars = CHUNK_MAX * 4
    parts = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces = []
    for p in parts:
        if est_tokens(p) <= CHUNK_MAX:
            pieces.append(p)
            continue
        lines, cur = p.splitlines(), ""
        for ln in lines:
            while est_tokens(ln) > CHUNK_MAX:  # one enormous line (e.g. a pasted transcript)
                pieces.append(ln[:limit_chars])
                ln = ln[limit_chars:]
            if est_tokens(cur + "\n" + ln) > CHUNK_MAX and cur:
                pieces.append(cur)
                cur = ln
            else:
                cur = (cur + "\n" + ln) if cur else ln
        if cur:
            pieces.append(cur)

    chunks, cur = [], ""
    for p in pieces:
        if cur and est_tokens(cur + "\n\n" + p) > CHUNK_MAX:
            chunks.append(cur)
            cur = cur[-CHUNK_OVERLAP_CHARS:].split(" ", 1)[-1] + "\n\n" + p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def chunk_note(title, body):
    """Return [(heading_path, text)] following the chunking rule above."""
    if est_tokens(body) <= CHUNK_WHOLE_NOTE_MAX:
        return [([], body.strip())] if body.strip() else []

    pieces = []
    for path, text in sections(body):
        if est_tokens(text) > CHUNK_MAX:
            pieces.extend((path, t) for t in split_long(text))
        else:
            pieces.append((path, text))

    # Pack neighbouring sections up to CHUNK_MAX while they share a top-level heading;
    # anything still under CHUNK_MIN joins its neighbour regardless.
    merged = []
    for path, text in pieces:
        if merged:
            prev_path, prev_text = merged[-1]
            fits = est_tokens(prev_text + "\n\n" + text) <= CHUNK_MAX
            same_top = prev_path[:1] == path[:1]
            tiny = est_tokens(text) < CHUNK_MIN or est_tokens(prev_text) < CHUNK_MIN
            if fits and (same_top or tiny):
                common = []
                for a, b in zip(prev_path, path):
                    if a != b:
                        break
                    common.append(a)
                merged[-1] = (common, prev_text + "\n\n" + text)
                continue
        merged.append((path, text))
    return merged


def note_date(path, fm, stat_birth):
    for key in ("date", "created", "updated"):
        v = fm.get(key)
        if v:
            s = v.isoformat() if hasattr(v, "isoformat") else str(v)
            if re.match(r"^\d{4}-\d{2}", s):
                return s[:10]
    m = TITLE_DATE_RE.match(Path(path).stem)
    if m:
        return m.group(1)
    return time.strftime("%Y-%m-%d", time.gmtime(stat_birth))


def text_hash(breadcrumb, text):
    """Identity of a chunk's embedded content; unchanged chunks keep their embedding across rebuilds."""
    return hashlib.sha1(f"{EMB_MODEL}\n{breadcrumb}\n\n{text}".encode()).hexdigest()[:16]


def build_rows(vault, folder):
    rows = []
    for md in iter_markdown(vault, folder or None):
        rel = md.relative_to(vault).as_posix()
        raw = md.read_text(encoding="utf-8", errors="replace")
        fm, body = parse_frontmatter(raw)
        date = note_date(rel, fm, birthtime(md.stat()))
        title = md.stem
        pieces = chunk_note(title, body)
        mtime = md.stat().st_mtime
        for seq, (path, text) in enumerate(pieces):
            crumb = " > ".join([title] + path)
            rows.append({"id": f"{rel}::{seq}", "note": rel, "seq": seq, "breadcrumb": crumb,
                         "text": text, "tokens": est_tokens(text), "note_date": date,
                         "note_mtime": mtime, "text_hash": text_hash(crumb, text),
                         "whole_note": len(pieces) == 1})
    return rows


def main():
    here = config.ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="", help="vault subfolder, e.g. 'Clients' (default: the whole vault)")
    ap.add_argument("--dry-run", action="store_true", help="chunk and report sizes, no embeddings or Neo4j")
    args = ap.parse_args()

    vault = config.vault_path()
    folder = args.folder.strip("/")
    prefix = config.scope_prefix(folder)
    rows = build_rows(vault, folder)
    notes = len({r["note"] for r in rows})
    sizes = sorted(r["tokens"] for r in rows)
    report = {
        "folder": folder or "(whole vault)", "notes": notes, "chunks": len(rows),
        "chunks_per_note": round(len(rows) / max(notes, 1), 1),
        "tokens_p10_median_p90_max": [sizes[int(len(sizes) * .1)], sizes[len(sizes) // 2],
                                      sizes[int(len(sizes) * .9)], sizes[-1]] if sizes else [],
        "whole_note_chunks": sum(1 for r in rows if r["whole_note"]),
        "total_tokens": sum(sizes),
        "settings": {"whole_note_max": CHUNK_WHOLE_NOTE_MAX, "max": CHUNK_MAX, "min": CHUNK_MIN,
                     "overlap_chars": CHUNK_OVERLAP_CHARS, "embedding": EMB_MODEL},
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return

    driver = config.driver()
    db = config.database()

    # Reuse embeddings of chunks whose breadcrumb + text haven't changed since the last build.
    t0 = time.time()
    with driver.session(database=db) as s:
        existing = {}
        for rec in s.run("""
            MATCH (c:Chunk)-[:PART_OF]->(n:Note)
            WHERE n.path STARTS WITH $prefix AND c.embedding IS NOT NULL AND c.emb_model = $model
            RETURN c.breadcrumb AS b, c.text AS t, c.embedding AS e
        """, prefix=prefix, model=EMB_MODEL):
            existing[text_hash(rec["b"], rec["t"])] = list(rec["e"])
    todo = []
    for r in rows:
        if r["text_hash"] in existing:
            r["embedding"] = existing[r["text_hash"]]
        else:
            todo.append(r)
    print(f"Reusing {len(rows) - len(todo)} embeddings; embedding {len(todo)} new or changed chunks", flush=True)
    report["reused_embeddings"], report["new_embeddings"] = len(rows) - len(todo), len(todo)

    for i in range(0, len(todo), EMB_BATCH):
        batch = todo[i:i + EMB_BATCH]
        vecs = embed([r["breadcrumb"] + "\n\n" + r["text"] for r in batch])
        for r, v in zip(batch, vecs):
            r["embedding"] = v
        if (i // EMB_BATCH) % 20 == 0:
            print(f"  embedded {i + len(batch)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    with driver.session(database=db) as s:
        s.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE").consume()
        existing = s.run("SHOW VECTOR INDEXES YIELD name, options WHERE name = 'chunk_embedding' "
                         "RETURN options.indexConfig['vector.dimensions'] AS dims").single()
        if existing and existing["dims"] != EMB_DIMS:
            s.run("DROP INDEX chunk_embedding").consume()
        s.run(f"""
            CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON c.embedding
            OPTIONS {{indexConfig: {{`vector.dimensions`: {EMB_DIMS}, `vector.similarity_function`: 'cosine'}}}}
        """).consume()
        s.run("CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text, c.breadcrumb]").consume()
        s.run("CREATE FULLTEXT INDEX note_title_ft IF NOT EXISTS FOR (n:Note) ON EACH [n.title]").consume()

        # Rebuild this folder's chunks from scratch so old chunk boundaries don't linger.
        s.run("""
            MATCH (c:Chunk)-[:PART_OF]->(n:Note) WHERE n.path STARTS WITH $prefix
            CALL (c) { DETACH DELETE c } IN TRANSACTIONS OF 1000 ROWS
        """, prefix=prefix).consume()

        for i in range(0, len(rows), 200):
            s.run("""
                UNWIND $rows AS r
                MATCH (n:Note {path: r.note})
                CREATE (c:Chunk {id: r.id, text: r.text, breadcrumb: r.breadcrumb, seq: r.seq,
                                 tokens: r.tokens, note_date: date(r.note_date), emb_model: $model,
                                 text_hash: r.text_hash, note_mtime: r.note_mtime, built_at: datetime()})
                CREATE (c)-[:PART_OF]->(n)
                WITH c, r
                CALL db.create.setNodeVectorProperty(c, 'embedding', r.embedding)
            """, rows=rows[i:i + 200], model=EMB_MODEL).consume()

        s.run("""
            MATCH (n:Note)<-[:PART_OF]-(c:Chunk) WHERE n.path STARTS WITH $prefix
            WITH n, c ORDER BY c.seq
            WITH n, collect(c) AS cs WHERE size(cs) > 1
            UNWIND range(0, size(cs) - 2) AS i
            WITH cs[i] AS a, cs[i + 1] AS b
            MERGE (a)-[:NEXT_CHUNK]->(b)
        """, prefix=prefix).consume()
        s.run("CALL db.awaitIndexes(300)").consume()  # the vector index fills asynchronously
    driver.close()

    report["embed_seconds"] = round(time.time() - t0)
    (here / "reports").mkdir(exist_ok=True)
    (here / "reports" / "chunks-last-run.json").write_text(json.dumps(report, indent=2))
    print(f"Done: {len(rows)} chunks stored ({len(todo)} newly embedded with {EMB_MODEL}) "
          f"in {report['embed_seconds']}s")


if __name__ == "__main__":
    sys.exit(main())
