"""Knowledge-layer extraction: entities, relationships and dated claims from (:Chunk) text with Claude Haiku 5.5.

OPTIONAL. The quickstart skips this step: data/sample_extractions.json holds a hand-written extraction of the
synthetic sample vault in the same format, and `load_graph.py --source sample` loads it. Run this script only on
your own vault, or to compare the API's extraction with the hand-written one.

Each chunk is sent once; results are cached in extractions/<content-hash>.json so reruns never re-bill finished
chunks. Nothing is written to Neo4j here; load_graph.py does that after entity resolution. Before sending, the
script estimates the cost and stops unless it is under --max-usd (default $0.50).

Usage:
  python extract_entities.py --sample 5            # 5 random chunks, about a cent
  python extract_entities.py                       # every chunk in the vault
  python extract_entities.py --folder Clients
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import sys
import threading
import time
from pathlib import Path

import config

MODEL = config.get("EXTRACT_MODEL", "claude-haiku-5-5")
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6   # USD per token, prompts under 100K tokens
WORKERS = 8  # parallel requests; the original run used 64, well under a 10M input-tokens/min limit

ENTITY_TYPES = ["Person", "Organization", "Project", "Product", "Concept", "Event", "Location", "Document",
                "Decision", "Lesson"]
REL_TYPES = [
    "ROLE_AT", "FOUNDED", "ADVISES", "INVESTED_IN", "CLIENT_OF", "PARTNER_OF", "SUPPLIER_TO",
    "BUILT", "USES", "PART_OF", "COMPETES_WITH", "INTRODUCED", "MET_WITH", "PARTICIPATED_IN",
    "LOCATED_IN", "DECIDED", "LED_TO", "LEARNED_FROM", "EVIDENCED_BY", "AUTHORED", "REPLACED",
    "RECEIVED", "MENTIONED_IN", "POSITIONED_AGAINST", "EXAMPLE_OF", "PARTY_TO", "COVERS", "AFFECTS",
    "PROMOTES", "RELATED_TO",
]
# The last six types were added after grouping RELATED_TO links by their evidence phrases; in the original
# build, links extracted before then were re-typed by a cheap batch pass instead of being re-extracted.
REL_TYPE_GUIDE = {
    "POSITIONED_AGAINST": "a product or company framed as an alternative to, or better than, another approach or framework (an order portal framed as the alternative to spreadsheets)",
    "EXAMPLE_OF": "something cited as an example or instance of a concept, pattern or category",
    "PARTY_TO": "a person or organization that signed or is bound by a contract or agreement (source = party, target = the Document)",
    "COVERS": "a contract, policy or agreement governs or applies to a project, product or information (source = the Document)",
    "AFFECTS": "a decision changes, retires, launches or concerns something (source = the Decision)",
    "PROMOTES": "a piece of content (landing page, script, ad, post) promotes or markets a product or organization",
}
MODALITY = ["fact", "reported", "plan", "opinion", "hypothetical", "ai_suggestion"]

DATE = {"type": ["string", "null"], "description": "YYYY, YYYY-MM or YYYY-MM-DD; null if the text gives no date"}
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities", "relationships", "claims"],
    "properties": {
        "entities": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "type", "aliases", "description"],
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": ENTITY_TYPES},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"},
            }}},
        "relationships": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["source", "target", "type", "detail", "valid_from", "valid_to", "modality", "confidence"],
            "properties": {
                "source": {"type": "string"},
                "target": {"type": "string"},
                "type": {"type": "string", "enum": REL_TYPES},
                "detail": {"type": "string"},
                "valid_from": DATE,
                "valid_to": DATE,
                "modality": {"type": "string", "enum": MODALITY},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            }}},
        "claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["subject", "attribute", "value", "valid_from", "valid_to", "modality", "confidence",
                         "evidence", "quote"],
            "properties": {
                "subject": {"type": "string"},
                "attribute": {"type": "string"},
                "value": {"type": "string"},
                "evidence": {"type": ["string", "null"],
                             "description": "name of a Document entity the passage cites as proof of this claim, else null"},
                "valid_from": DATE,
                "valid_to": DATE,
                "modality": {"type": "string", "enum": MODALITY},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "quote": {"type": "string"},
            }}},
    },
}

SYSTEM = f"""You extract a knowledge graph from one passage of a private Obsidian vault. The note author is {config.AUTHOR_NAME}; refer to them as "{config.AUTHOR_NAME}" when the text says I/me/my.

Return three lists:

entities: the specific people, organizations, projects, products, concepts, events, locations, documents, decisions and lessons the passage is about or meaningfully mentions. Use the most complete proper name that appears (e.g. "Maria Santos", not "Maria" when the full name is present) and list other forms used in the passage as aliases. Skip generic nouns ("customers", "the app") unless they name a specific thing. description: one sentence on what this passage says it is.
- Decision: a specific choice that was made or proposed, named as a short action phrase (e.g. "Switch to a new main supplier", "Hire a part-time designer"). Link who made it with DECIDED, what prompted it with LED_TO (cause -> decision), and what it changed with LED_TO (decision -> outcome).
- Lesson: a takeaway or principle drawn from experience, named as a short statement (e.g. "Short notice of a price rise loses customers"). Link it with LEARNED_FROM to the project, event or decision it came from.
- Document: a specific artefact such as a deck, study, contract, report, screenshot, invoice or article.

Conversation transcripts: many passages are chats between the author and an AI assistant (ChatGPT, Claude). Never extract the conversation itself: no entities or relationships for "the AI assistant", "ChatGPT generated/answered/drafted X" or "the author asked Y". Extract only the substance discussed, and mark anything the assistant proposed or asserted as modality ai_suggestion. Name ChatGPT or Claude as an entity only when the passage is about the product itself (e.g. its pricing or its use inside the business).

relationships: links between two of your entities, using exactly one type from: {", ".join(REL_TYPES)}. Use ROLE_AT for job titles and roles, putting the role in detail. Use RECEIVED for awards, funding, grants and recognition (source = the recipient, target = the award, event or giver). Use MENTIONED_IN when an entity is cited by a publication or document, AUTHORED for who wrote or created a document, and EVIDENCED_BY when a decision, result or other entity is backed by a specific document (e.g. a study, deck or contract). {chr(10).join(f"{t}: {d}." for t, d in REL_TYPE_GUIDE.items())}
RELATED_TO is a last resort: if the link is really a fact or a label about one entity (a figure, a status, an award name, "listed as", a contract term), record it as a claim instead. The author's own wishes, intentions and views ("I want…", "I am considering…", "I think…") are claims with modality plan or opinion, not relationships. detail: a short phrase in the passage's own terms.

Companies vs products: a company or team is an Organization; the software or app it sells is a Product. When a name is used for both (e.g. a startup and its app), create the Organization, and a Product only if the passage discusses the product separately, linking them with BUILT. Do not record the same link twice under two types (e.g. FOUNDED and BUILT for the same founding).

claims: specific facts that could change over time or be contradicted elsewhere: figures, prices, metrics, counts, dates, statuses, roles, decisions. subject is an entity name, attribute is a short lowercase label (e.g. "delivery time", "price per seat", "status"), value is the value as written, quote is the shortest exact phrase from the passage that supports it.

Time: valid_from / valid_to are when the fact was true in the world, only if the passage says so ("since 2023", "until August"). Leave them null otherwise; never infer them from the note's date, which is supplied separately.

modality:
- fact: stated as having happened or being true, including measurements and metrics ("login success is near 100%"), people's backgrounds ("15 years in the industry") and agreed terms of contracts ("3-month probation").
- reported: what a named third party said, asked for, complained about or gave as feedback ("the client wants weekly deliveries"). The fact that they said it is certain; whether it is true is theirs.
- plan: intended, proposed or scheduled by the author or their team, not yet done.
- opinion: a judgement, assessment or belief, usually the author's ("pricing is too low").
- hypothetical: a what-if, an illustrative example or a template value.
- ai_suggestion: anything an AI assistant proposed, drafted or asserted in a transcript.
confidence: how clearly the passage states it, not whether it is true.
evidence (claims only): the name of a Document entity the passage cites as the source or proof, else null.

Only extract what the passage itself supports. An empty list is a valid answer. Keep names in their original language."""


# Changing the instructions or schema changes this, so cached results from an older prompt aren't reused.
PROMPT_VERSION = hashlib.sha1((SYSTEM + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:8]


def chunk_rows(folder):
    driver = config.driver()
    with driver.session(database=config.database()) as s:
        rows = s.run("""
            MATCH (c:Chunk)-[:PART_OF]->(n:Note) WHERE n.path STARTS WITH $prefix
            RETURN c.id AS id, c.text AS text, c.breadcrumb AS breadcrumb, toString(c.note_date) AS note_date,
                   n.path AS path ORDER BY c.id
        """, prefix=config.scope_prefix(folder)).data()
    driver.close()
    return rows


def cache_path(here, row):
    """Keyed by content, not position: an edited chunk gets a new key and is re-extracted,
    while an unchanged chunk that moved (renamed note, inserted section) reuses its result."""
    key = f"{MODEL}\n{PROMPT_VERSION}\n{row['breadcrumb']}\n{row['text']}"
    return here / "extractions" / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")


def extract(client, row):
    user = (f"Note: {row['path']}\nSection: {row['breadcrumb']}\nNote date: {row['note_date']}\n\n"
            f"<passage>\n{row['text']}\n</passage>")
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,  # under Haiku's minimum cacheable size, and every passage is unique: no caching
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {getattr(resp.stop_details, 'category', None)}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("hit max_tokens")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp.usage


def estimate_usd(rows, out_tokens=1500):
    """Rough pre-flight cost: system prompt + passage in, ~1,500 tokens of JSON out per chunk."""
    tokens_in = sum((len(SYSTEM) + len(json.dumps(SCHEMA)) + len(r["text"]) + 200) / 4 for r in rows)
    return tokens_in * PRICE_IN + len(rows) * out_tokens * PRICE_OUT


def main():
    here = config.ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="", help="vault subfolder (default: the whole vault)")
    ap.add_argument("--max-usd", type=float, default=0.50, help="stop before sending if the estimate is higher")
    ap.add_argument("--sample", type=int, help="extract only N randomly chosen chunks (seeded)")
    ap.add_argument("--workers", type=int, default=WORKERS, help=f"parallel API requests (default {WORKERS})")
    args = ap.parse_args()

    rows = chunk_rows(args.folder)
    if not rows:
        sys.exit("No chunks found; run build_chunks.py first.")
    if args.sample:
        rows = random.Random(42).sample(rows, min(args.sample, len(rows)))
    (here / "extractions").mkdir(exist_ok=True)
    todo = [r for r in rows if not cache_path(here, r).exists()]
    est = estimate_usd(todo)
    print(f"{len(rows)} chunks selected, {len(rows) - len(todo)} already extracted, {len(todo)} to send "
          f"(estimated ${est:.3f})", flush=True)
    if not todo:
        return
    if est > args.max_usd:
        sys.exit(f"Estimated ${est:.2f} is over --max-usd {args.max_usd:.2f}; raise it to continue.")
    if not config.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. To build the graph without it: python load_graph.py --source sample")
    import anthropic
    client = anthropic.Anthropic(api_key=config.get("ANTHROPIC_API_KEY"), max_retries=5)

    totals = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0, "done": 0, "failed": 0}
    lock = threading.Lock()
    failures = []

    def work(row):
        try:
            data, usage = extract(client, row)
        except (anthropic.BadRequestError, anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise SystemExit(f"Stopping: {type(e).__name__}: {e}")  # config problem; retrying won't help
        except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError, json.JSONDecodeError) as e:
            with lock:
                totals["failed"] += 1
                failures.append({"id": row["id"], "error": f"{type(e).__name__}: {e}"})
            return
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        out = {"chunk_id": row["id"], "path": row["path"], "note_date": row["note_date"], "model": MODEL,
               "usage": {"in": usage.input_tokens, "out": usage.output_tokens,
                         "cache_read": cache_read, "cache_write": cache_write}, **data}
        cache_path(here, row).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        with lock:
            totals["in"] += usage.input_tokens
            totals["out"] += usage.output_tokens
            totals["cache_read"] += cache_read
            totals["cache_write"] += cache_write
            totals["done"] += 1
            if totals["done"] % 50 == 0:
                print(f"  {totals['done']}/{len(todo)} done", flush=True)

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))

    cost = totals["in"] * PRICE_IN + totals["cache_read"] * PRICE_IN * 0.1 \
        + totals["cache_write"] * PRICE_IN * 1.25 + totals["out"] * PRICE_OUT
    report = {"chunks_sent": len(todo), **totals, "seconds": round(time.time() - t0),
              "cost_usd": round(cost, 4),
              "cost_per_chunk_usd": round(cost / max(totals["done"], 1), 6), "failures": failures}
    print(json.dumps(report, indent=2))
    (here / "reports").mkdir(exist_ok=True)
    (here / "reports" / "extract-last-run.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
