"""Check eval questions before review: schema, unique ids, gold paths exist, quotes appear in a gold note.

Usage (from the repo root, after build_chunks.py):
  python evals/validate.py                      # evals/questions.yaml
  python evals/validate.py path/to/draft.yaml
"""

import re
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402

CATEGORIES = {"single_fact", "multi_hop", "entity_rollup", "temporal", "modality_trap",
              "broad_theme", "paraphrase_alias", "disambiguation", "unanswerable"}
MODALITIES = {"fact", "reported", "plan", "opinion", "hypothetical", "ai_suggestion"}


def norm(text):
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[\s*_`>#|]+", " ", text).strip().lower()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "questions.yaml"
    questions = yaml.safe_load(path.read_text()) or []
    paths = {g["path"] for q in questions for g in (q.get("gold") or []) + (q.get("traps") or [])}

    driver = config.driver()
    with driver, driver.session(database=config.database()) as s:
        # Note titles count as content: chunk breadcrumbs carry them into both search indexes.
        texts = {r["p"]: norm(Path(r["p"]).stem + "\n" + "\n".join(r["t"])) for r in s.run(
            "MATCH (c:Chunk)-[:PART_OF]->(n:Note) WHERE n.path IN $paths "
            "WITH n, c ORDER BY c.seq RETURN n.path AS p, collect(c.text) AS t", paths=list(paths))}

    problems = []
    for qid, n in Counter(q.get("id") for q in questions).items():
        if n > 1:
            problems.append(f"{qid}: duplicate id")
    for q in questions:
        qid = q.get("id", "?")
        if q.get("category") not in CATEGORIES:
            problems.append(f"{qid}: unknown category {q.get('category')}")
        if q.get("split") not in ("dev", "test"):
            problems.append(f"{qid}: split must be dev or test")
        gold = q.get("gold") or []
        if q.get("category") != "unanswerable" and not gold:
            problems.append(f"{qid}: answerable question without gold notes")
        if q.get("category") == "multi_hop" and sum(g.get("grade") == 2 for g in gold) < 2:
            problems.append(f"{qid}: multi_hop needs at least two grade-2 notes")
        for g in gold:
            if g["path"] not in texts:
                problems.append(f"{qid}: gold path not in graph: {g['path']}")
            if g.get("grade") not in (1, 2):
                problems.append(f"{qid}: grade must be 1 or 2 for {g['path']}")
        for t in q.get("traps") or []:
            if t["path"] not in texts:
                problems.append(f"{qid}: trap path not in graph: {t['path']}")
            elif norm(str(t["value"]).lstrip("<>")) not in texts[t["path"]]:
                problems.append(f"{qid}: trap value '{t['value']}' not found in {t['path']}")
        haystack = [texts.get(g["path"], "") for g in gold]
        for f in q.get("key_facts") or []:
            if f.get("modality") not in MODALITIES:
                problems.append(f"{qid}: bad modality {f.get('modality')}")
            quote = f.get("quote")
            if not quote:
                problems.append(f"{qid}: key fact without quote: {f.get('fact', '')[:60]}")
            elif not any(norm(quote) in h for h in haystack):
                problems.append(f"{qid}: quote not found in any gold note: \"{quote[:80]}\"")

    by_cat = Counter(q.get("category") for q in questions)
    print(f"{len(questions)} questions | " + ", ".join(f"{c} {n}" for c, n in sorted(by_cat.items())))
    print(f"{len(paths)} distinct gold notes, {len(paths & texts.keys())} found in graph")
    print("\n".join(problems) if problems else "OK: no problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
