"""Resolve extracted entities and load the entity + claim layer into Neo4j.

Reads one extraction per chunk and rebuilds, from scratch:
  (:Entity:<Type> {id, name, aliases, description, mentions, first_seen, last_seen})
  (:Chunk)-[:MENTIONS]->(:Entity)
  (:Entity)-[:<REL_TYPE> {count, details, modalities, sources, first_seen, last_seen, valid_from, valid_to}]->(:Entity)
  (:Claim {attribute, value, quote, modality, confidence, valid_from, valid_to, note_date, is_current})
      -[:ABOUT]->(:Entity), (:Chunk)-[:SUPPORTS]->(:Claim), (:Claim)-[:EVIDENCED_BY]->(:Entity:Document)
  (:Claim)-[:SUPERSEDED_BY]->(:Claim)  a later note gives a different value, same subject + attribute (facts only)
  (:Claim)-[:CONFLICTS_WITH]-(:Claim)  two notes of the same date disagree
  (:Entity)-[:DESCRIBED_BY]->(:Note)   when a wiki/entities or wiki/concepts note matches the entity

Extractions come from one of two sources:
  --source api      extractions/<hash>.json, written by extract_entities.py (Claude Haiku)
  --source sample   data/sample_extractions.json, a hand-written extraction of sample_vault/ (no API key needed)

Resolution: names are normalised (case, punctuation, spacing, ".ai", company suffixes); Organization,
Product, Project and Concept share one namespace (type by majority vote); other types resolve within
their own type. Specific aliases (full names, not single first names) merge entities; ambiguous
aliases are ignored. Nothing here calls an API. (The original build also applied two LLM passes that merged
near-duplicate names and dropped unsupported links; they are not part of this repo.)

Usage:
  python load_graph.py --source sample --dry-run
  python load_graph.py --source sample
  python load_graph.py --source api --folder Clients
"""

import argparse
import collections
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import config
from extract_entities import cache_path, chunk_rows

HERE = config.ROOT
SAMPLE = HERE / "data" / "sample_extractions.json"
FAMILY = {"Organization": "thing", "Product": "thing", "Project": "thing", "Concept": "thing",
          "Person": "person", "Location": "location", "Event": "event", "Document": "document",
          "Decision": "decision", "Lesson": "lesson"}
TYPE_PRIORITY = ["Organization", "Product", "Project", "Concept"]  # tie-break within "thing"
SUFFIX_RE = re.compile(r"\b(pte|ltd|limited|inc|llc|co|corp|corporation|company|sdn bhd)\b\.?")
AUTHOR = config.AUTHOR_NAME.lower()
AUTHOR_FIRST = AUTHOR.split()[0]
AUTHOR_FORMS = config.AUTHOR_ALIASES | {AUTHOR.replace(" ", "")}
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
# Attributes that hold one of many values (a paper's findings, a product's bugs), so a different value is
# another item, not a newer one.
LIST_ATTRS = {"finding", "findings", "feature", "features", "bug", "bugs", "feedback", "issue", "issues",
              "challenge", "challenges", "risk", "risks", "insight", "insights", "example", "examples",
              "use case", "use cases", "benefit", "benefits", "pain point", "pain points", "quote", "idea",
              "ideas", "task", "tasks", "question", "questions", "recommendation", "recommendations",
              "observation", "observations", "limitation", "limitations", "step", "steps", "note", "goal", "goals"}


def same_value(a, b):
    """Treat '3,000+' and 'over 3,000' (or '15-50%' and '15–50% uplift') as the same value: when both
    contain numbers, compare the numbers; otherwise compare the normalised text."""
    na = [x.replace(",", "") for x in NUM_RE.findall(a)]
    nb = [x.replace(",", "") for x in NUM_RE.findall(b)]
    if na and nb:
        return na == nb
    return norm(a).replace(" ", "") == norm(b).replace(" ", "")


# Entities mentioned in more chunks than this get :HubEntity so traversal can skip them. The published build used
# 300 on 31,047 chunks; .env.example sets a value that suits the 50-note sample vault.
HUB_MENTIONS = int(config.get("HUB_MENTIONS", 300))


def norm(name):
    s = unicodedata.normalize("NFKC", name).lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"\.(ai|com|io|co|app)\b", "", s)
    s = re.sub(r"^the\s+", "", s)
    s = SUFFIX_RE.sub(" ", s)
    s = re.sub(r"[^\w\s㐀-鿿]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# A person the notes call by the author's bare first name is usually the author, but reviews and transcripts
# mention namesakes ("the branch manager, Ruth"). Such a mention stays the author unless its description gives
# someone else's role and nothing about the author.
SELF_RE = re.compile(r"\b(author|note|journal|user|founder|himself|herself|writer|owner|narrator|"
                     r"speaker|i|me|my|interviewer|host|" + re.escape(AUTHOR) + r")\b")


def separate_namesakes(doc):
    """Rename a person called by the author's bare first name who is someone else (described in another role, with
    another full name as alias) to that alias, and repoint this chunk's relationships and claims that name them,
    so they are not merged into the author."""
    for e in doc.get("entities", []):
        if FAMILY.get(e["type"]) != "person" or norm(e["name"]) != AUTHOR_FIRST:
            continue
        desc = norm(e.get("description") or "")
        others = [a for a in e.get("aliases", []) if norm(a) not in AUTHOR_FORMS and len(norm(a).split()) >= 2]
        if not desc or SELF_RE.search(desc) or not others:
            continue
        bare, new = e["name"], max(others, key=len)
        e["name"], e["aliases"] = new, []
        for r in doc.get("relationships", []):
            for side in ("source", "target"):
                if r.get(side) == bare:
                    r[side] = new
        for c in doc.get("claims", []):
            if c.get("subject") == bare:
                c["subject"] = new
    return doc


def key_of(name, etype):
    n = norm(name)
    if FAMILY.get(etype) == "person" and n in AUTHOR_FORMS:
        n = AUTHOR
    compact = n.replace(" ", "")
    return f"{FAMILY.get(etype, 'thing')}:{compact}" if compact else None


def specific_alias(alias, etype):
    n = norm(alias)
    if FAMILY.get(etype) == "person":
        return len(n.split()) >= 2  # full names only; "Sam" alone could be anyone
    return len(n.replace(" ", "")) >= 4


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def date_key(d):
    return (d or "")[:10]


def resolve(docs, wiki, merges=()):
    uf = UnionFind()
    surface = collections.defaultdict(collections.Counter)   # key -> surface names
    types = collections.defaultdict(collections.Counter)     # key -> entity types
    alias_targets = collections.defaultdict(set)              # alias key -> entity keys claiming it

    for d in docs:
        for e in d["entities"]:
            k = key_of(e["name"], e["type"])
            if not k:
                continue
            uf.find(k)
            surface[k][e["name"]] += 1
            types[k][e["type"]] += 1
            for a in e["aliases"]:
                if specific_alias(a, e["type"]):
                    ak = key_of(a, e["type"])
                    if ak and ak != k:
                        alias_targets[ak].add(k)

    # An alias merges entities only when it is a spelling nobody uses as an entity name in its own right.
    # Otherwise one chunk calling "Harbour Blend" an alias of "Fernwick" would fold the two together, and the
    # merges would chain.
    ambiguous = 0
    for ak, targets in alias_targets.items():
        if ak in surface or len(targets) > 1:
            ambiguous += 1
            continue
        uf.union(ak, next(iter(targets)))
    for a, b in merges:  # confirmed near-duplicates ("Fernwick Roasting" / "Fernwick Roasting Co")
        if a in surface and b in surface:
            uf.union(a, b)

    groups = collections.defaultdict(list)
    for k in list(surface):
        groups[uf.find(k)].append(k)
    canon = {}
    for keys in groups.values():
        root = max(keys, key=lambda k: (sum(surface[k].values()), -len(k)))  # most-mentioned real name
        tcount = collections.Counter()
        names = collections.Counter()
        for k in keys:
            tcount.update(types[k])
            names.update(surface[k])
        best = max(tcount.items(), key=lambda kv: (kv[1], -TYPE_PRIORITY.index(kv[0]) if kv[0] in TYPE_PRIORITY else -99))
        name = max(names.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        for k in keys:
            canon[k] = {"id": root, "name": name, "type": best[0]}
    for k in list(canon):  # point every member key at the group's shared record
        canon[k] = canon[canon[k]["id"]] if canon[k]["id"] in canon else canon[k]

    wiki_links = {}
    for path, title, aliases in wiki:
        for nm in [title.replace("-", " ")] + list(aliases or []):
            for fam in ("thing", "person", "location", "event", "document"):
                k = f"{fam}:{norm(nm).replace(' ', '')}"
                if k in canon:
                    wiki_links.setdefault(canon[k]["id"], path)
    return canon, wiki_links, ambiguous


def lookup(canon, name, local):
    if name in local:
        return local[name]
    for fam in ("thing", "person", "document", "decision", "lesson", "event", "location"):
        k = f"{fam}:{norm(name).replace(' ', '')}"
        if fam == "person" and norm(name) in AUTHOR_FORMS:
            k = "person:" + AUTHOR.replace(" ", "")
        if k in canon:
            return canon[k]["id"]
    return None


def build(docs, canon, dropped_links=()):
    ents, mentions, rels, claims = {}, set(), {}, []
    skipped_rel_endpoints = 0
    drop_links = {(canon[s]["id"], t, canon[o]["id"]) for s, t, o in dropped_links if s in canon and o in canon}
    for d in docs:
        cid, nd = d["chunk_id"], d["note_date"]
        local = {}
        for e in d["entities"]:
            k = key_of(e["name"], e["type"])
            if not k or k not in canon:
                continue
            c = canon[k]
            local[e["name"]] = c["id"]
            rec = ents.setdefault(c["id"], {"id": c["id"], "name": c["name"], "type": c["type"], "aliases": set(),
                                            "descriptions": collections.Counter(), "chunks": set(),
                                            "first_seen": nd, "last_seen": nd})
            rec["aliases"].add(e["name"])
            for a in e["aliases"]:  # only aliases that resolve to this same entity, not other entities' names
                ak = key_of(a, e["type"])
                if ak and (ak not in canon or canon[ak]["id"] == c["id"]):
                    rec["aliases"].add(a)
            if e["description"]:
                rec["descriptions"][e["description"]] += 1
            rec["chunks"].add(cid)
            rec["first_seen"] = min(rec["first_seen"] or nd, nd or rec["first_seen"])
            rec["last_seen"] = max(rec["last_seen"] or nd, nd or rec["last_seen"])
            mentions.add((cid, c["id"]))

        for r in d["relationships"]:
            s, t = lookup(canon, r["source"], local), lookup(canon, r["target"], local)
            if not s or not t or s == t:
                skipped_rel_endpoints += 1
                continue
            if (s, r["type"], t) in drop_links:
                continue
            agg = rels.setdefault((s, r["type"], t), {"count": 0, "details": [], "modalities": set(), "sources": set(),
                                                      "confidence": set(), "first_seen": nd, "last_seen": nd,
                                                      "valid_from": None, "valid_to": None})
            agg["count"] += 1
            if len(agg["details"]) < 5 and r["detail"] not in agg["details"]:
                agg["details"].append(r["detail"])
            agg["modalities"].add(r["modality"])
            agg["confidence"].add(r["confidence"])
            agg["sources"].add(cid)
            agg["first_seen"] = min(agg["first_seen"] or nd, nd or agg["first_seen"])
            agg["last_seen"] = max(agg["last_seen"] or nd, nd or agg["last_seen"])
            if r["valid_from"]:
                agg["valid_from"] = min(agg["valid_from"] or r["valid_from"], r["valid_from"])
            if r["valid_to"]:
                agg["valid_to"] = max(agg["valid_to"] or r["valid_to"], r["valid_to"])

        for c in d["claims"]:
            subj = lookup(canon, c["subject"], local)
            if not subj:
                continue
            ev = lookup(canon, c["evidence"], local) if c.get("evidence") else None
            cid_hash = hashlib.sha1(f"{cid}|{c['subject']}|{c['attribute']}|{c['value']}".encode()).hexdigest()[:16]
            claims.append({"id": cid_hash, "chunk": cid, "path": d.get("path"), "about": subj, "evidence": ev,
                           "attribute": c["attribute"], "attr_key": norm(c["attribute"]),
                           "value": c["value"], "value_key": norm(c["value"]).replace(" ", ""),
                           "quote": c["quote"], "modality": c["modality"], "confidence": c["confidence"],
                           "valid_from": c["valid_from"], "valid_to": c["valid_to"], "note_date": nd})

    claims = list({c["id"]: c for c in claims}.values())  # the same claim can be listed twice in one chunk

    # Drop one-off concepts with nothing attached: the most common extraction noise.
    linked = {s for s, _, t in rels} | {t for _, _, t in rels} | {c["about"] for c in claims}
    drop = {i for i, e in ents.items() if e["type"] == "Concept" and len(e["chunks"]) == 1 and i not in linked}
    for i in drop:
        del ents[i]
    mentions = {(c, e) for c, e in mentions if e not in drop}
    return ents, mentions, rels, claims, skipped_rel_endpoints, len(drop)


def timelines(claims):
    """Supersession and conflicts among factual claims sharing subject + attribute.

    Ordered by note date, the day the author wrote it down, not valid_from: a 2025 note saying "launched in 2022"
    is a newer view than a 2023 note saying "prototype not built", even though its valid_from is earlier.
    Claims from the same note are items of one list (two models' accuracy, two findings), so they are never
    paired, and list-like attributes get no timeline at all."""
    groups = collections.defaultdict(list)
    for c in claims:
        c["is_current"] = None
        if c["modality"] in ("fact", "reported") and c["attr_key"] not in LIST_ATTRS:
            groups[(c["about"], c["attr_key"])].append(c)
    superseded, conflicts = [], []
    for items in groups.values():
        if len({c["path"] for c in items}) < 2:
            continue
        items.sort(key=lambda c: (date_key(c["note_date"]), date_key(c["valid_from"])))
        latest = date_key(items[-1]["note_date"])
        for c in items:
            c["is_current"] = date_key(c["note_date"]) == latest
        for a, b in zip(items, items[1:]):
            if a["path"] == b["path"] or same_value(a["value"], b["value"]):
                continue
            da, db = date_key(a["note_date"]), date_key(b["note_date"])
            (conflicts if da == db else superseded).append({"a": a["id"], "b": b["id"]})
    return superseded, conflicts


def load_docs(folders, source):
    sample = json.loads(SAMPLE.read_text()) if source == "sample" else None
    docs, missing = [], 0
    for folder in folders:
        for row in chunk_rows(folder):
            if sample is not None:
                d = sample.get(row["id"])
                d = {**d, "note_date": row["note_date"]} if d else None
            else:
                p = cache_path(HERE, row)
                d = json.loads(p.read_text()) if p.exists() else None
            if d is None:
                missing += 1
                continue
            d["chunk_id"] = row["id"]  # identical-text chunks share a cache file; keep each chunk's own id
            d["path"] = row["path"]
            docs.append(separate_namesakes(d))
    return docs, missing


def wiki_notes(s):
    return [(r["p"], r["t"], r["a"]) for r in s.run(
        "MATCH (n:Note) WHERE n.path STARTS WITH 'wiki/entities/' OR n.path STARTS WITH 'wiki/concepts/' "
        "RETURN n.path AS p, n.title AS t, n.aliases AS a")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["sample", "api"], default="sample",
                    help="sample: data/sample_extractions.json (no API key); api: extract_entities.py's cache")
    ap.add_argument("--folder", action="append", help="vault subfolder (repeatable; default: the whole vault)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    docs, missing = load_docs(args.folder or [""], args.source)

    driver = config.driver()
    db = config.database()
    with driver.session(database=db) as s:
        wiki = wiki_notes(s)

    canon, wiki_links, ambiguous = resolve(docs, wiki)
    ents, mentions, rels, claims, skipped, dropped = build(docs, canon)
    superseded, conflicts = timelines(claims)
    raw_entities = sum(len(d["entities"]) for d in docs)
    top = sorted(ents.values(), key=lambda e: -len(e["chunks"]))[:12]
    report = {
        "chunks_with_results": len(docs), "chunks_missing_results": missing,
        "raw_entity_mentions": raw_entities, "canonical_entities": len(ents),
        "dropped_one_off_concepts": dropped, "ambiguous_aliases_ignored": ambiguous,
        "entity_types": dict(collections.Counter(e["type"] for e in ents.values()).most_common()),
        "mentions": len(mentions), "relationships": len(rels),
        "relationship_observations": sum(r["count"] for r in rels.values()),
        "relationships_skipped_unresolved_endpoint": skipped,
        "claims": len(claims), "superseded_links": len(superseded), "conflicts": len(conflicts),
        "wiki_linked_entities": sum(1 for i in wiki_links if i in ents),
        "top_entities": [(e["name"], e["type"], len(e["chunks"]), len(e["aliases"])) for e in top],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    ent_rows = [{"id": e["id"], "name": e["name"], "type": e["type"],
                 "aliases": sorted(a for a in e["aliases"] if a != e["name"])[:30],
                 "description": e["descriptions"].most_common(1)[0][0] if e["descriptions"] else "",
                 "mentions": len(e["chunks"]), "first_seen": e["first_seen"], "last_seen": e["last_seen"],
                 "wiki": wiki_links.get(e["id"])} for e in ents.values()]
    mention_rows = [{"c": c, "e": e} for c, e in mentions]
    rel_rows = [{"s": s, "t": t, "type": typ, "count": v["count"], "details": v["details"],
                 "modalities": sorted(v["modalities"]), "confidence": sorted(v["confidence"]),
                 "sources": sorted(v["sources"])[:50], "first_seen": v["first_seen"], "last_seen": v["last_seen"],
                 "valid_from": v["valid_from"], "valid_to": v["valid_to"]}
                for (s, typ, t), v in rels.items() if s in ents and t in ents]
    claim_rows = [{k: v for k, v in c.items() if k != "path"} for c in claims if c["about"] in ents]
    for c in claim_rows:
        if c["evidence"] not in ents:
            c["evidence"] = None

    def run_batches(s, query, rows, size=1000, **params):
        for i in range(0, len(rows), size):
            s.run(query, rows=rows[i:i + size], **params).consume()

    with driver.session(database=db) as s:
        s.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE").consume()
        s.run("CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE").consume()
        s.run("CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]").consume()
        s.run("CREATE INDEX claim_attr IF NOT EXISTS FOR (c:Claim) ON (c.attr_key)").consume()
        s.run("CREATE FULLTEXT INDEX claim_ft IF NOT EXISTS FOR (c:Claim) ON EACH [c.attribute, c.value, c.quote]").consume()
        for label in ("Claim", "Entity"):
            s.run(f"MATCH (n:{label}) CALL (n) {{ DETACH DELETE n }} IN TRANSACTIONS OF 5000 ROWS").consume()

        run_batches(s, """
            UNWIND $rows AS r
            CREATE (e:Entity {id: r.id, name: r.name, type: r.type, aliases: r.aliases, description: r.description,
                              mentions: r.mentions, first_seen: date(r.first_seen), last_seen: date(r.last_seen)})
            SET e:$(r.type)
            WITH e, r WHERE r.mentions > $hub
            SET e:HubEntity
        """, ent_rows, hub=HUB_MENTIONS)
        run_batches(s, """
            UNWIND $rows AS r
            MATCH (e:Entity {id: r.id}), (n:Note {path: r.wiki})
            CREATE (e)-[:DESCRIBED_BY]->(n)
        """, [r for r in ent_rows if r["wiki"]])
        run_batches(s, """
            UNWIND $rows AS r
            MATCH (c:Chunk {id: r.c}), (e:Entity {id: r.e})
            CREATE (c)-[:MENTIONS]->(e)
        """, mention_rows, size=5000)
        run_batches(s, """
            UNWIND $rows AS r
            MATCH (a:Entity {id: r.s}), (b:Entity {id: r.t})
            CREATE (a)-[x:$(r.type) {count: r.count, details: r.details, modalities: r.modalities,
                                    confidence: r.confidence, sources: r.sources,
                                    first_seen: date(r.first_seen), last_seen: date(r.last_seen),
                                    valid_from: r.valid_from, valid_to: r.valid_to}]->(b)
        """, rel_rows)
        run_batches(s, """
            UNWIND $rows AS r
            MATCH (e:Entity {id: r.about}), (ch:Chunk {id: r.chunk})
            CREATE (c:Claim {id: r.id, attribute: r.attribute, attr_key: r.attr_key, value: r.value,
                             quote: r.quote, modality: r.modality, confidence: r.confidence,
                             valid_from: r.valid_from, valid_to: r.valid_to, note_date: date(r.note_date),
                             is_current: r.is_current})
            CREATE (c)-[:ABOUT]->(e)
            CREATE (ch)-[:SUPPORTS]->(c)
            WITH c, r WHERE r.evidence IS NOT NULL
            MATCH (d:Entity {id: r.evidence})
            CREATE (c)-[:EVIDENCED_BY]->(d)
        """, claim_rows)
        run_batches(s, "UNWIND $rows AS r MATCH (a:Claim {id: r.a}), (b:Claim {id: r.b}) CREATE (a)-[:SUPERSEDED_BY]->(b)",
                    superseded)
        run_batches(s, "UNWIND $rows AS r MATCH (a:Claim {id: r.a}), (b:Claim {id: r.b}) CREATE (a)-[:CONFLICTS_WITH]->(b)",
                    conflicts)
        s.run("CALL db.awaitIndexes(300)").consume()
    driver.close()

    report["seconds"] = round(time.time() - t0, 1)
    (HERE / "reports").mkdir(exist_ok=True)
    (HERE / "reports" / "load-last-run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Loaded in {report['seconds']}s")


if __name__ == "__main__":
    main()
