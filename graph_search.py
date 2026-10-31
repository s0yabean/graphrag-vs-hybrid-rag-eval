"""GraphRAG search over the Obsidian graph: hybrid retrieval, then entities, typed links and dated claims on top.

`ask` starts from the same base as the hybrid method (evals/methods/hybrid.py): one BM25 list (chunk_text) and one
embedding list (chunk_embedding), each collapsed to its best chunk per note and cut to 20 notes, fused by RRF
(k=60). Those chunks keep their order at the top of `ranked`. The graph adds the claim and link lines in `context`
and appends chunks only the graph reached (via claims and link sources) after the hybrid ones.

Read-only. Rules enforced here rather than left to the caller:
  - hops: 1 by default, 2 at most; `path` searches up to 4 hops.
  - hubs: :HubEntity nodes (more than HUB_MENTIONS mentions, e.g. the company and the note author) are never
    traversed through. When a hub is named in the question it is still an entry point, but only its links to
    entities the question's top chunks also mention are followed, plus its 10 strongest links.
  - seeds: besides the entities named in the question, the 8 non-hub entities the hybrid top-20 chunks mention
    most, weighted by fused rank, are entry points too, with a smaller fan-out (8 links each).
  - fan-out: at most 25 links per entity at hop 1 and 10 at hop 2, strongest (most observed) first; a named hub
    follows up to 60 links to entities in the top chunks (trimming then removes the irrelevant ones).
  - trimming: claim and neighbourhood-link lines are scored by embedding similarity to the question and dropped when
    more than 0.15 below the best line (relative, because absolute scores vary by question). Links of the
    relationship types the question asked for are always kept. --no-trim turns it off.
  - budget: output is cut to --budget tokens (default 50,000, ~4 characters each), context lines first, then chunks.
  - outdated facts are returned, marked superseded, never hidden; conflicts are shown both sides.

Commands (all print JSON):
  ask "<question>"            entry entities + claims + links + ranked evidence chunks
  entity "<name>"             one entity: type, aliases, description, strongest links, latest claims
  timeline "<name>" [--attribute A]   claims about an entity in note-date order with supersessions
  path "<name A>" "<name B>"  shortest chain of typed links between two entities, hubs excluded

Usage:
  python graph_search.py ask "How did the wholesale price of Harbour Blend change?"
  python graph_search.py timeline "Harbour Blend" --attribute "wholesale price"
  python graph_search.py path "Late Alta Verde shipments 2025" "The Copper Kettle"
"""

import argparse
import json
import re
import sys
import unicodedata

import config
from config import embed, embed_many

MAX_HOPS = 2
FANOUT = {1: 25, 2: 10}
HUB_TOP_LINKS = 10
HUB_FOCUS_FANOUT = 60  # a named hub's links to entities in the question's top chunks; trimming filters them after
TRIM_MARGIN = 0.15  # keep claim/link lines whose embedding similarity to the question is within this of the best line
TRIM_MAX_LINES = 400
HYBRID_NOTES = 20     # notes per list and after fusion, as evals/methods/hybrid.py uses
HYBRID_CHUNKS = 60    # chunks fetched per list before collapsing to notes
LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')
SEEDS = 8          # entry entities added from the question's top chunks
SEED_FANOUT = 8    # links followed per seed (named entities get FANOUT)
OPINION = {"opinion", "plan", "hypothetical", "ai_suggestion"}
# Question words that name a relationship type. Prefixes of 4+ letters match word starts; shorter ones whole words.
REL_HINTS = {
    "introduc": ["INTRODUCED"], "referr": ["INTRODUCED"], "client": ["CLIENT_OF"], "customer": ["CLIENT_OF"],
    "invest": ["INVESTED_IN"], "funded": ["INVESTED_IN"], "partner": ["PARTNER_OF"], "advis": ["ADVISES"],
    "mentor": ["ADVISES"], "found": ["FOUNDED"], "decid": ["DECIDED", "LED_TO"], "decision": ["DECIDED", "LED_TO"],
    "learn": ["LEARNED_FROM"], "lesson": ["LEARNED_FROM"], "compet": ["COMPETES_WITH", "POSITIONED_AGAINST"],
    "met": ["MET_WITH"], "meet": ["MET_WITH"], "built": ["BUILT"], "build": ["BUILT"], "role": ["ROLE_AT"],
    "worked": ["ROLE_AT"], "hired": ["ROLE_AT"], "supplier": ["SUPPLIER_TO"], "vendor": ["SUPPLIER_TO"],
    "replac": ["REPLACED"], "award": ["RECEIVED"], "grant": ["RECEIVED"], "led": ["LED_TO"], "result": ["LED_TO"],
    "outcome": ["LED_TO"], "promot": ["PROMOTES"], "agreement": ["PARTY_TO"], "contract": ["PARTY_TO"],
    "proof": ["EVIDENCED_BY"], "evidence": ["EVIDENCED_BY"], "receipt": ["EVIDENCED_BY"],
}
STOP = set("""a about after again all also am an and any are as at be been before being between both but by can
change changed changes could did do does doing during each for from had has have how i if in into is it its just
me more most my no not now of on once only or other our out over own same she should so some such than that the
their them then there these they this those through time times to too under until up very was we were what when
where which while who whom why will with would you your he him his her hers herself himself us ours
yours itself""".split())  # pronouns: the extractor stores some as aliases ("his" on a dozen people)
STOP |= {w for w in re.findall(r"\w+", config.AUTHOR_NAME.lower()) if w != "the"}  # the author is in every note


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def lucene(text, drop=()):
    words = [w for w in re.findall(r"[\w-]+", text) if len(w) > 2 and w.lower() not in STOP and w.lower() not in drop]
    return " OR ".join(re.sub(r'([+\-!(){}\[\]^"~*?:\\/])', r"\\\1", w) for w in words) or "*"


def rel_hints(text):
    types = []
    for w in norm(text).split():
        for k, ts in REL_HINTS.items():
            if (w.startswith(k) if len(k) >= 4 else w == k):
                types += [t for t in ts if t not in types]
    return types


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / ((sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5 or 1)


def trim(question_vec, lines):
    """Drop lines far less relevant to the question than the best one. Returns (kept lines best first, dropped count)."""
    if not lines:
        return [], 0
    lines = lines[:TRIM_MAX_LINES]
    vecs = embed_many([ln[:300] for ln in lines])
    scored = sorted(((cosine(question_vec, v), ln) for v, ln in zip(vecs, lines)), key=lambda x: -x[0])
    floor = scored[0][0] - TRIM_MARGIN
    kept = [ln for sc, ln in scored if sc >= floor]
    return kept, len(lines) - len(kept)


class Graph:
    def __init__(self):
        self.driver = config.driver()
        self.db = config.database()

    def q(self, cypher, **params):
        with self.driver.session(database=self.db, default_access_mode="READ") as s:
            return s.run(cypher, **params).data()

    # ---------- building blocks ----------

    def find_entities(self, text, limit=40):
        """Entities whose name or alias appears in the text (named), else the best full-text matches."""
        rows = self.q("""
            CALL db.index.fulltext.queryNodes('entity_name_ft', $q, {limit: $limit}) YIELD node, score
            RETURN node.id AS id, node.name AS name, node.type AS type, node.aliases AS aliases,
                   node.mentions AS mentions, node:HubEntity AS hub, score
        """, q=lucene(text), limit=limit)
        # Full-text ranking buries short exact names ("Fernwick" under "Fernwick at Home"), so also look up
        # every 1-5 word run of the question as an exact name or alias.
        words = norm(text).split()
        grams = list({" ".join(words[i:j]) for i in range(len(words)) for j in range(i + 1, min(i + 6, len(words) + 1))})
        known = {r["id"] for r in rows}
        rows += [r for r in self.q("""
            MATCH (e:Entity)
            WHERE toLower(e.name) IN $grams OR any(a IN e.aliases WHERE toLower(a) IN $grams)
            RETURN e.id AS id, e.name AS name, e.type AS type, e.aliases AS aliases,
                   e.mentions AS mentions, e:HubEntity AS hub, 10.0 AS score
        """, grams=grams) if r["id"] not in known]
        t = f" {norm(text)} "
        named = []
        for r in rows:
            # Names match case-insensitively. A one-word alias is often a common word the extractor attached
            # ("framework", "map", "his"), so it must appear in the question with the same casing ("MAP", "HIS").
            raw = " " + re.sub(r"[^\w\s]", " ", text) + " "
            forms = [(r["name"], False)] + [(a, len(norm(a).split()) == 1) for a in (r["aliases"] or [])]
            hit = max((len(norm(f)) for f, strict in forms
                       if len(norm(f)) >= 3 and f" {norm(f)} " in t
                       and (not strict or (f != f.lower() and f" {f} " in raw))
                       and (norm(f) not in STOP or (f.isupper() and f" {f} " in raw))), default=0)
            if hit:
                named.append({**r, "match": hit})
        # keep the longest matches: "Fernwick at Home" beats "Fernwick" when both appear
        named.sort(key=lambda r: (-r["match"], -r["mentions"]))
        kept = []
        for r in named:
            if not any(f" {norm(r['name'])} " in f" {norm(k['name'])} " and k["name"] != r["name"] and not r["hub"]
                       for k in kept):
                kept.append(r)
        if kept:
            return kept[:6], True
        return [r for r in rows[:3] if r["score"] > 1.0], False

    def hybrid_chunks(self, question, scope, vec):
        """The hybrid base list: BM25 + embeddings, best chunk per note, RRF k=60. BM25 takes the question words
        as typed, OR'd."""
        tail = """
            MATCH (node)-[:PART_OF]->(n:Note) WHERE n.path STARTS WITH $prefix
            RETURN node.id AS chunk_id, n.path AS path, score ORDER BY score DESC LIMIT $chunks"""
        lists = []
        for head, params in (
                ("CALL db.index.fulltext.queryNodes('chunk_text', $q, {limit: $fetch}) YIELD node, score",
                 {"q": " OR ".join(LUCENE_SPECIAL.sub(r"\\\1", w) for w in question.split()) or "*"}),
                ("CALL db.index.vector.queryNodes('chunk_embedding', $fetch, $vec) YIELD node, score",
                 {"vec": vec})):
            best = {}
            for r in self.q(head + tail, fetch=HYBRID_CHUNKS * 8, prefix=scope, chunks=HYBRID_CHUNKS, **params):
                best.setdefault(r["path"], r["chunk_id"])
            lists.append(list(best.items())[:HYBRID_NOTES])
        score, record = {}, {}
        for lst in lists:
            for rank, (path, cid) in enumerate(lst, 1):
                score[path] = score.get(path, 0) + 1 / (60 + rank)
                record.setdefault(path, cid)
        order = sorted(score, key=lambda p: (-score[p], p))[:HYBRID_NOTES]
        ents = {r["id"]: r["e"] for r in self.q("""
            UNWIND $ids AS id MATCH (c:Chunk {id: id})
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity) WHERE NOT e:HubEntity
            RETURN c.id AS id, collect(e.id) AS e""", ids=[record[p] for p in order])}
        return [{"chunk_id": record[p], "path": p, "entities": ents.get(record[p], [])} for p in order]

    def claims_for(self, question, entity_ids, scope, limit=40, drop=(), seed_ids=()):
        """Claims matching the question's words, about the entry entities first, with their timeline neighbours."""
        return self.q("""
            CALL db.index.fulltext.queryNodes('claim_ft', $q, {limit: 400}) YIELD node AS c, score
            MATCH (c)-[:ABOUT]->(e:Entity), (ch:Chunk)-[:SUPPORTS]->(c), (ch)-[:PART_OF]->(n:Note)
            WHERE n.path STARTS WITH $prefix
            WITH c, e, ch, n, score * CASE WHEN e.id IN $ids THEN 3.0 WHEN e.id IN $seeds THEN 1.5 ELSE 1.0 END
                 * CASE WHEN any(w IN $words WHERE c.attr_key CONTAINS w) THEN 2.0 ELSE 1.0 END AS s
            ORDER BY s DESC LIMIT $limit
            OPTIONAL MATCH (c)-[:SUPERSEDED_BY]->(newer:Claim)
            OPTIONAL MATCH (older:Claim)-[:SUPERSEDED_BY]->(c)
            OPTIONAL MATCH (c)-[:CONFLICTS_WITH]-(other:Claim)
            OPTIONAL MATCH (c)-[:EVIDENCED_BY]->(doc:Entity)
            RETURN c.id AS id, e.name AS subject, c.attribute AS attribute, c.value AS value, c.quote AS quote,
                   c.modality AS modality, toString(c.note_date) AS date, c.is_current AS current,
                   n.path AS path, ch.id AS chunk_id, s AS score, newer.value AS newer_value,
                   toString(newer.note_date) AS newer_date, older.value AS older_value,
                   toString(older.note_date) AS older_date, other.value AS conflicting_value, doc.name AS evidence
        """, q=lucene(question, drop), ids=entity_ids, seeds=list(seed_ids), prefix=scope, limit=limit,
            words=[w for w in norm(question).split() if len(w) > 3 and w not in STOP and w not in drop])

    def links(self, entity_ids, hops, focus_ids, fan=None):
        """Typed entity-entity links out to `hops`, never passing through a hub."""
        out, frontier, seen = [], list(entity_ids), set(entity_ids)
        for hop in range(1, min(hops, MAX_HOPS) + 1):
            rows = self.q("""
                UNWIND $ids AS id
                MATCH (a:Entity {id: id})-[r]-(b:Entity)
                WITH a, r, b, a:HubEntity AS a_hub
                WHERE NOT a_hub OR b.id IN $focus
                WITH a, r, b, a_hub ORDER BY r.count DESC
                WITH a, a_hub, collect({r: r, b: b}) AS all
                WITH a, a_hub, all[..CASE WHEN a_hub THEN $hubfan ELSE $fan END] AS picked
                UNWIND picked AS p
                WITH a, p.r AS r, p.b AS b
                RETURN a.id AS from_id, startNode(r) = a AS outgoing, a.name AS a, type(r) AS type, b.name AS b,
                       b.id AS b_id, b.type AS b_type, b:HubEntity AS b_hub, r.count AS count,
                       r.details[0] AS detail, r.modalities AS modalities, toString(r.first_seen) AS first_seen,
                       toString(r.last_seen) AS last_seen, r.sources[..3] AS sources
            """, ids=frontier, focus=list(focus_ids), fan=min(fan or FANOUT[hop], FANOUT[hop]), hubfan=HUB_FOCUS_FANOUT)
            hub_rows = self.q("""
                UNWIND $ids AS id
                MATCH (a:HubEntity {id: id})-[r]-(b:Entity)
                WITH a, r, b ORDER BY r.count DESC
                WITH a, collect({r: r, b: b})[..$n] AS picked
                UNWIND picked AS p
                WITH a, p.r AS r, p.b AS b
                RETURN a.id AS from_id, startNode(r) = a AS outgoing, a.name AS a, type(r) AS type, b.name AS b,
                       b.id AS b_id, b.type AS b_type, b:HubEntity AS b_hub, r.count AS count,
                       r.details[0] AS detail, r.modalities AS modalities, toString(r.first_seen) AS first_seen,
                       toString(r.last_seen) AS last_seen, r.sources[..3] AS sources
            """, ids=frontier, n=HUB_TOP_LINKS) if hop == 1 else []
            nxt = []
            for r in rows + hub_rows:
                r["hop"] = hop
                out.append(r)
                if not r["b_hub"] and r["b_id"] not in seen:
                    seen.add(r["b_id"])
                    nxt.append(r["b_id"])
            frontier = nxt
            if not frontier:
                break
        uniq = {}
        for r in out:
            uniq.setdefault((min(r["a"], r["b"]), r["type"], max(r["a"], r["b"])), r)
        return list(uniq.values())

    def typed_links(self, types, entry_ids, scope, limit=40):
        """Links of the types the question names, anchored on the entry entities when there are any."""
        return self.q("""
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE type(r) IN $types AND (size($ids) = 0 OR a.id IN $ids OR b.id IN $ids)
              AND any(src IN r.sources WHERE src STARTS WITH $prefix)
            WITH a, r, b ORDER BY r.count DESC, r.last_seen DESC
            WITH type(r) AS t, collect({a: a, r: r, b: b})[..$per] AS picked  // a fair share per requested type
            UNWIND picked AS p
            WITH p.a AS a, p.r AS r, p.b AS b
            RETURN a.id AS from_id, true AS outgoing, a.name AS a, type(r) AS type, b.name AS b, b.id AS b_id,
                   b.type AS b_type, b:HubEntity AS b_hub, r.count AS count, r.details[0] AS detail,
                   r.modalities AS modalities, toString(r.first_seen) AS first_seen,
                   toString(r.last_seen) AS last_seen, r.sources[..3] AS sources, 0 AS hop
        """, types=types, ids=entry_ids, prefix=scope, per=max(5, limit // len(types)))

    def entity_names(self, ids):
        return self.q("""UNWIND $ids AS id MATCH (e:Entity {id: id})
                         RETURN e.id AS id, e.name AS name, e.type AS type, e.mentions AS mentions,
                                e:HubEntity AS hub, e.aliases AS aliases""", ids=ids)

    def chunk_texts(self, ids):
        return {r["id"]: r for r in self.q("""
            UNWIND $ids AS id MATCH (c:Chunk {id: id})-[:PART_OF]->(n:Note)
            RETURN c.id AS id, n.path AS path, c.breadcrumb AS breadcrumb, c.text AS text
        """, ids=list(ids))}


# ---------- formatting ----------

def link_line(r):
    a, b = (r["a"], r["b"]) if r["outgoing"] else (r["b"], r["a"])
    mods = ",".join(m for m in (r["modalities"] or []) if m != "fact")
    when = r["first_seen"] if r["first_seen"] == r["last_seen"] else f"{r['first_seen']}..{r['last_seen']}"
    return f"{a} -{r['type']}-> {b} ({r['detail']}; seen {r['count']}x, {when}{'; ' + mods if mods else ''})"


def claim_line(c):
    state = "" if c["current"] is not False else " [superseded"
    if c["newer_value"]:
        state = f" [superseded {c['newer_date']} by: {c['newer_value']}"
    if state:
        state += "]"
    if c["older_value"]:
        state += f" [earlier {c['older_date']}: {c['older_value']}]"
    if c["conflicting_value"]:
        state += f" [CONFLICTS with same-date note: {c['conflicting_value']}]"
    mod = "" if c["modality"] == "fact" else f" ({c['modality']})"
    ev = f" evidence: {c['evidence']}" if c["evidence"] else ""
    return f"[{c['date']}] {c['subject']} · {c['attribute']}: {c['value']}{mod}{state}{ev} — \"{c['quote']}\" ({c['path']})"


def link_key(r):
    """Same link whichever end it was reached from."""
    return (r["a"], r["type"], r["b"]) if r["outgoing"] else (r["b"], r["type"], r["a"])


def fit(lines, budget_chars):
    out, used = [], 0
    for ln in lines:
        if used + len(ln) > budget_chars:
            break
        out.append(ln)
        used += len(ln) + 1
    return out, used


# ---------- commands ----------

def ask(g, question, scope, k, hops, budget, facts_only, no_trim=False):
    entries, named = g.find_entities(question)
    qvec = embed(question)
    chunks = g.hybrid_chunks(question, scope, qvec)
    focus = {e for c in chunks for e in c["entities"]}
    if not named:
        entries = []
    # Seeds: the entities the best-matching chunks talk about, scored by fused rank and by how many top chunks
    # mention them (hybrid_chunks already leaves hubs out). They widen the entry beyond the names in the question.
    seed_score = {}
    for i, c in enumerate(chunks):
        for e in set(c["entities"]):
            seed_score[e] = seed_score.get(e, 0) + 1 / (5 + i)
    named_ids = [e["id"] for e in entries]
    seed_ids = [e for e in sorted(seed_score, key=lambda e: (-seed_score[e], e)) if e not in named_ids][:SEEDS]
    seeds = {r["id"]: r for r in g.entity_names(seed_ids)}
    seeds = [seeds[i] for i in seed_ids if i in seeds]
    entry_ids = named_ids + seed_ids
    types = rel_hints(question)
    # Named entities are handled by the ABOUT boost; leaving them in the claim query just matches every claim about them.
    name_words = {w for e in entries for w in norm(e["name"]).split()}
    claims = g.claims_for(question, named_ids, scope, drop=name_words, seed_ids=seed_ids)
    if facts_only:
        claims = [c for c in claims if c["modality"] not in OPINION]
    links = g.links(named_ids, hops, focus) if named_ids else []
    seen = {link_key(r) for r in links}
    links += [r for r in (g.links(seed_ids, hops, focus, fan=SEED_FANOUT) if seed_ids else [])
              if link_key(r) not in seen and not seen.add(link_key(r))]
    if types:  # typed links first: anchored on the entry entities, or across the whole scope when nothing is named
        typed = g.typed_links(types, entry_ids if named else [], scope)
        seen = {link_key(r) for r in typed}
        links = typed + [r for r in links if link_key(r) not in seen]
    # Links matter more when they reach entities the question's best chunks also talk about.
    links.sort(key=lambda r: (r["hop"], -(r["b_id"] in focus), -r["count"]))

    # Evidence: the hybrid chunks in their own order, then chunks only the graph reached (claims, link sources).
    base = [c["chunk_id"] for c in chunks]
    score = {}
    for i, c in enumerate(claims):
        score[c["chunk_id"]] = score.get(c["chunk_id"], 0) + 1 / (60 + i)
    for i, r in enumerate(links[:30]):
        for src in r["sources"] or []:
            score[src] = score.get(src, 0) + 0.5 / (60 + i)
    extra = [c for c in sorted(score, key=score.get, reverse=True) if c not in set(base)]
    top = (base + extra)[:k]
    texts = g.chunk_texts(top)

    chars = budget * 4
    # Typed links the question asked for, then claims, then the rest of the neighbourhood.
    by_type = {}
    for r in links:
        if r["hop"] == 0:
            by_type.setdefault(r["type"], []).append(link_line(r))
    typed_lines = [ln for group in zip(*[g + [None] * 40 for g in by_type.values()]) for ln in group if ln]  # round-robin
    other_lines = [link_line(r) for r in links if r["hop"] != 0]
    claim_lines = [claim_line(c) for c in claims]
    dropped = 0
    if not no_trim:  # typed links were asked for by name, so only claims and the neighbourhood are trimmed
        kept, dropped = trim(qvec, claim_lines + other_lines)
        is_claim = set(claim_lines)
        claim_lines = [ln for ln in kept if ln in is_claim]
        other_lines = [ln for ln in kept if ln not in is_claim]
    context, used = fit(typed_lines + claim_lines + other_lines, int(chars * 0.4))
    ranked = []
    for cid in top:
        t = texts.get(cid)
        if not t:
            continue
        body = t["text"][: max(0, min(4000, chars - used))]
        if not body:
            break
        ranked.append({"path": t["path"], "chunk_id": cid, "breadcrumb": t["breadcrumb"], "text": body})
        used += len(body)
    return {
        "question": question,
        "entry_entities": [{"name": e["name"], "type": e["type"], "mentions": e["mentions"], "hub": e["hub"],
                            "via": "named"} for e in entries] +
                          [{"name": e["name"], "type": e["type"], "mentions": e["mentions"], "hub": False,
                            "via": "top chunks"} for e in seeds],
        "entry_by": ("named in question + top-chunk seeds" if named else "top-chunk seeds") if entry_ids
                    else "none: chunks only",
        "relationship_types_from_question": types,
        "rules": {"hops": min(hops, MAX_HOPS), "fanout": FANOUT, "hubs": "entry only, never traversed",
                  "budget_tokens": budget, "facts_only": facts_only,
                  "trim": None if no_trim else f"dropped lines more than {TRIM_MARGIN} below the best match"},
        "context": context,
        "ranked": ranked,
        "counts": {"claims": len(claims), "links": len(links), "hybrid_chunks": len(base),
                   "graph_only_chunks": len([c for c in top if c not in set(base)]),
                   "lines_trimmed_as_irrelevant": dropped, "context_lines_returned": len(context)},
    }


def resolve_one(g, name):
    rows, _ = g.find_entities(name)
    exact = [r for r in rows if norm(r["name"]) == norm(name)]
    if not (exact or rows):
        sys.exit(json.dumps({"error": f"no entity matches '{name}'"}))
    return (exact or rows)[0]


def entity(g, name, budget):
    e = resolve_one(g, name)
    info = g.q("""
        MATCH (e:Entity {id: $id})
        OPTIONAL MATCH (e)-[:DESCRIBED_BY]->(w:Note)
        RETURN e.name AS name, e.type AS type, e.aliases AS aliases, e.description AS description,
               e.mentions AS mentions, toString(e.first_seen) AS first_seen, toString(e.last_seen) AS last_seen,
               e:HubEntity AS hub, w.path AS wiki_note
    """, id=e["id"])[0]
    links = g.links([e["id"]], 1, set())
    links.sort(key=lambda r: -r["count"])
    claims = g.q("""
        MATCH (c:Claim)-[:ABOUT]->(:Entity {id: $id}), (ch:Chunk)-[:SUPPORTS]->(c), (ch)-[:PART_OF]->(n:Note)
        WHERE c.is_current IS NULL OR c.is_current
        RETURN toString(c.note_date) AS date, c.attribute AS attribute, c.value AS value, c.modality AS modality,
               n.path AS path ORDER BY c.note_date DESC LIMIT 40
    """, id=e["id"])
    lines, _ = fit([link_line(r) for r in links] +
                   [f"[{c['date']}] {c['attribute']}: {c['value']} ({c['modality']}; {c['path']})" for c in claims],
                   budget * 4)
    return {**info, "context": lines}


def timeline(g, name, attribute, budget, facts_only):
    e = resolve_one(g, name)
    rows = g.q("""
        MATCH (c:Claim)-[:ABOUT]->(:Entity {id: $id}), (ch:Chunk)-[:SUPPORTS]->(c), (ch)-[:PART_OF]->(n:Note)
        WHERE ($attr IS NULL OR c.attr_key CONTAINS $attr) AND (NOT $facts OR c.modality IN ['fact', 'reported'])
        OPTIONAL MATCH (c)-[:SUPERSEDED_BY]->(newer:Claim)
        OPTIONAL MATCH (c)-[:CONFLICTS_WITH]-(other:Claim)
        RETURN toString(c.note_date) AS date, c.attribute AS attribute, c.value AS value, c.modality AS modality,
               c.valid_from AS valid_from, c.is_current AS current, newer IS NOT NULL AS superseded,
               other.value AS conflicting_value, c.quote AS quote, n.path AS path
        ORDER BY c.note_date, c.attr_key
    """, id=e["id"], attr=norm(attribute) if attribute else None, facts=facts_only)
    lines = []
    for r in rows:
        tag = " (superseded)" if r["superseded"] else (" (current)" if r["current"] else "")
        tag += f" [conflicts: {r['conflicting_value']}]" if r["conflicting_value"] else ""
        vf = f", valid from {r['valid_from']}" if r["valid_from"] else ""
        lines.append(f"[{r['date']}{vf}] {r['attribute']}: {r['value']} ({r['modality']}){tag} — {r['path']}")
    lines, _ = fit(lines, budget * 4)
    return {"entity": e["name"], "attribute": attribute, "claims": len(rows), "context": lines}


def path(g, a, b, max_hops=4):
    ea, eb = resolve_one(g, a), resolve_one(g, b)
    rows = g.q(f"""
        MATCH (a:Entity {{id: $a}}), (b:Entity {{id: $b}})
        MATCH p = shortestPath((a)-[*..{max_hops}]-(b))
        WHERE all(n IN nodes(p) WHERE n:Entity) AND none(n IN nodes(p)[1..-1] WHERE n:HubEntity)
        RETURN [n IN nodes(p) | n.name] AS names,
               [r IN relationships(p) | {{type: type(r), from: startNode(r).name, detail: r.details[0]}}] AS rels
    """, a=ea["id"], b=eb["id"])
    if not rows:
        return {"from": ea["name"], "to": eb["name"], "path": None,
                "note": f"no chain of typed links within {max_hops} hops that avoids hub entities"}
    return {"from": ea["name"], "to": eb["name"], "hops": len(rows[0]["rels"]),
            "path": [f"{r['from']} -{r['type']}-> ({r['detail']})" for r in rows[0]["rels"]], "nodes": rows[0]["names"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--scope", default="", help="folder prefix, e.g. Clients (default: the whole vault)")
    p.add_argument("--k", type=int, default=30, help="evidence chunks to return: the 20 hybrid ones, then graph-only")
    p.add_argument("--hops", type=int, default=1, choices=[1, 2])
    p.add_argument("--budget", type=int, default=50000, help="output token budget")
    p.add_argument("--no-trim", action="store_true", help="keep claim/link lines however unrelated to the question")
    p.add_argument("--facts-only", action="store_true", help="drop opinion/plan/hypothetical/ai_suggestion claims")
    p = sub.add_parser("entity")
    p.add_argument("name")
    p.add_argument("--budget", type=int, default=3000)
    p = sub.add_parser("timeline")
    p.add_argument("name")
    p.add_argument("--attribute")
    p.add_argument("--facts-only", action="store_true")
    p.add_argument("--budget", type=int, default=3000)
    p = sub.add_parser("path")
    p.add_argument("a")
    p.add_argument("b")
    args = ap.parse_args()

    g = Graph()
    try:
        if args.cmd == "ask":
            out = ask(g, args.question, config.scope_prefix(args.scope), args.k, args.hops, args.budget,
                      args.facts_only, args.no_trim)
        elif args.cmd == "entity":
            out = entity(g, args.name, args.budget)
        elif args.cmd == "timeline":
            out = timeline(g, args.name, args.attribute, args.budget, args.facts_only)
        else:
            out = path(g, args.a, args.b)
    except Exception as e:  # surface setup problems in one parseable line
        msg = str(e).splitlines()[0][:300]
        print(f"GRAPH_ERROR {type(e).__name__}: {msg}", file=sys.stderr)
        sys.exit(2)
    finally:
        g.driver.close()
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
