"""Run retrieval methods over the approved eval questions and score them against gold notes.

Writes results/<timestamp>/<method>.jsonl (one row per question, with the retrieved text) and summary.md
(a per-category verdict, metrics per category and split, and per-question wins/ties/losses with a sign test).

Metrics, per question:
  recall@5/10/20, all_gold@20, mrr, ndcg@10   note-level ranking against the gold notes
  fact_recall@20    share of key facts whose quote appears in the retrieved text of the top-20 notes
  note_recall@8k    gold notes present in the first 8,000 tokens the method hands the answerer
  fact_recall@8k    key facts whose quote appears in those 8,000 tokens
  precision@8k      share of those 8,000 tokens that come from gold notes
  current_first     temporal/modality questions with a distractor: 1 when an essential note ranks above
                    every distractor note
The 8K view is what an answering model would actually see: for the graph method that is its claim and link
lines first (up to 40% of the budget, as graph_search.py splits it; each line counts toward the note it cites),
then its chunks in rank order.

Usage (from the repo root):
  python evals/run_eval.py --methods hybrid graph
  python evals/run_eval.py --methods hybrid graph graph_notrim
  python evals/run_eval.py --methods hybrid graph --split dev
  python evals/run_eval.py --rescore evals/results/<run>          # recompute metrics from saved rows, no retrieval
  python evals/run_eval.py --methods hybrid graph --pool          # candidates for labelling, no scoring
  python evals/run_eval.py --methods hybrid --include-drafts      # dry-run on draft questions
"""

import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
KS = (5, 10, 20)
DEPTH = max(KS)
BUDGET_CHARS = 8000 * 4
CONTEXT_SHARE = 0.4
METRICS = ["recall@10", "recall@20", "all_gold@20", "mrr", "ndcg@10",
           "fact_recall@20", "note_recall@8k", "fact_recall@8k", "precision@8k", "current_first"]
# Metrics that decide the per-category verdict, in order of weight: did the facts reach the answerer, did the
# right notes, and how early.
VERDICT = ["fact_recall@8k", "note_recall@8k", "recall@10", "mrr", "precision@8k"]
NOISE = 0.10  # a mean gap smaller than this on ~5 questions is not a difference
PATH_RE = re.compile(r"\(([^()]*?\.md)\)")  # claim lines end with "(<note path>)"


def norm(text):
    """Same normalisation as validate.py, so a quote it accepted is matched the same way here."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[\s*_`>#|]+", " ", text).strip().lower()


def load_method(name):
    spec = importlib.util.spec_from_file_location(name, HERE / "methods" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.retrieve


def note_ranking(ranked):
    """Collapse chunk rows to notes, each at its best rank."""
    seen, order = set(), []
    for row in ranked:
        if row["path"] not in seen:
            seen.add(row["path"])
            order.append(row["path"])
    return order


def delivered(row):
    """The first BUDGET_CHARS of what the method hands the answerer: (text, cited note paths) pieces."""
    out, used = [], 0
    # Context lines get at most 40% of the budget, the split graph_search.py itself uses at --budget 8000.
    for ln in row.get("context") or []:
        if used + len(ln) > CONTEXT_SHARE * BUDGET_CHARS:
            break
        out.append((ln, set(PATH_RE.findall(ln))))
        used += len(ln)
    for r in row["ranked"]:
        if used >= BUDGET_CHARS:
            break
        text = f"{Path(r['path']).stem}\n{r['text']}"[: BUDGET_CHARS - used]
        out.append((text, {r["path"]}))
        used += len(text)
    return out


def facts_found(facts, text):
    quotes = [norm(f["quote"]) for f in facts if f.get("quote")]
    return sum(q in text for q in quotes) / len(quotes) if quotes else None


def score(q, row):
    gold = q["gold"]
    grades = {g["path"]: g["grade"] for g in gold}
    order = note_ranking(row["ranked"])
    essential = [p for p, g in grades.items() if g == 2]
    out = {f"recall@{k}": len(set(order[:k]) & grades.keys()) / len(grades) for k in KS}
    out["all_gold@20"] = float(all(p in order[:20] for p in essential))
    first = next((i for i, p in enumerate(order, 1) if p in grades), None)
    out["mrr"] = 1 / first if first else 0.0
    dcg = sum((2 ** grades.get(p, 0) - 1) / math.log2(i + 1) for i, p in enumerate(order[:10], 1))
    ideal = sorted(grades.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0

    facts = q.get("key_facts") or []
    top20 = set(order[:20])
    out["fact_recall@20"] = facts_found(facts, norm("\n".join(
        f"{Path(r['path']).stem}\n{r['text']}" for r in row["ranked"] if r["path"] in top20)))
    view = delivered(row)
    seen = set().union(*(p for _, p in view)) if view else set()
    out["note_recall@8k"] = len(seen & grades.keys()) / len(grades)
    out["fact_recall@8k"] = facts_found(facts, norm("\n".join(t for t, _ in view)))
    total = sum(len(t) for t, _ in view)
    out["precision@8k"] = sum(len(t) for t, p in view if p & grades.keys()) / total if total else 0.0

    distractors = [g["path"] for g in gold if g["grade"] == 1 and str(g.get("why", "")).startswith("distractor")]
    if q["category"] in ("temporal", "modality_trap") and distractors:
        rank = {p: i for i, p in enumerate(order)}
        best_e = min((rank.get(p, math.inf) for p in essential), default=math.inf)
        best_d = min((rank.get(p, math.inf) for p in distractors), default=math.inf)
        out["current_first"] = float(best_e < best_d)
    else:
        out["current_first"] = None
    return out


def mean(rows, key):
    vals = [r["metrics"][key] for r in rows if r["metrics"] and r["metrics"].get(key) is not None]
    return (statistics.mean(vals), len(vals)) if vals else (None, 0)


def sign_test(wins, losses):
    """Two-sided exact sign test over non-tied questions."""
    n = wins + losses
    if not n:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def fmt(v):
    return "–" if v is None else f"{v:.2f}"


def summarise(questions, results, split_arg, run_dir):
    lines = [f"# Retrieval eval {run_dir.name}", "",
             f"Questions: {len(questions)} (split: {split_arg}). Methods: {', '.join(results)}. "
             f"Context budget for the 8k metrics: {BUDGET_CHARS // 4:,} tokens.", ""]
    errors = [(m, r["id"], r["error"]) for m, rows in results.items() for r in rows if r["error"]]
    if errors:
        lines += ["## Failed runs (scored as zero)", "", *[f"- `{m}` {i}: {e}" for m, i, e in errors], ""]
    cats = sorted({q["category"] for q in questions if q["category"] != "unanswerable"})

    if len(results) >= 2:
        a, b = list(results)[:2]
        lines += [f"## Verdict by question type ({a} vs {b}, both splits)", "",
                  f"Each cell is {a} / {b}. A gap under {NOISE:.2f} is called a tie: with about 5 questions per type "
                  "smaller gaps are noise. The winner is decided on the metrics in this order: "
                  + ", ".join(f"`{m}`" for m in VERDICT) + " (first non-tied one wins).", "",
                  "| type | n | " + " | ".join(VERDICT) + " | current_first | winner |",
                  "|---" * (len(VERDICT) + 4) + "|"]
        for cat in [*cats, "ALL"]:
            sa = [r for r in results[a] if r["metrics"] and cat in ("ALL", r["category"])]
            sb = [r for r in results[b] if r["metrics"] and cat in ("ALL", r["category"])]
            if not sa:
                continue
            cells, winner = [], "tie"
            for m in VERDICT + ["current_first"]:
                (va, _), (vb, _) = mean(sa, m), mean(sb, m)
                cells.append(f"{fmt(va)} / {fmt(vb)}")
                if m in VERDICT and winner == "tie" and va is not None and vb is not None and abs(va - vb) >= NOISE:
                    winner = f"**{a if va > vb else b}** ({m})"
            lines.append(f"| {cat} | {len(sa)} | " + " | ".join(cells) + f" | {winner} |")
        lines.append("")

        for metric in ("fact_recall@8k", "recall@10"):
            lines += [f"## Per-question {metric}: {a} vs {b}", "",
                      "| type | wins | ties | losses | sign-test p |", "|---|---|---|---|---|"]
            tally = defaultdict(lambda: [0, 0, 0])
            for ra, rb in zip(results[a], results[b]):
                if not ra["metrics"] or ra["metrics"].get(metric) is None:
                    continue
                d = ra["metrics"][metric] - rb["metrics"][metric]
                for c in (ra["category"], "ALL"):
                    tally[c][0 if d > 0 else 1 if d == 0 else 2] += 1
            for c in [*cats, "ALL"]:
                if c in tally:
                    w, t, l = tally[c]
                    lines.append(f"| {c} | {w} | {t} | {l} | {sign_test(w, l):.2f} |")
            lines += ["", f"Wins are for `{a}`. p < 0.05 means the difference is unlikely to be chance.", ""]

    for split in (["dev", "test"] if split_arg == "all" else [split_arg]):
        if not any(q.get("split") == split for q in questions):
            continue
        lines += [f"## All metrics, {split} split", "", "| method | type | n | " + " | ".join(METRICS) + " | sec |",
                  "|---" * (len(METRICS) + 4) + "|"]
        for name, rows in results.items():
            for cat in [*cats, "ALL"]:
                sel = [r for r in rows if r["split"] == split and r["metrics"] and cat in ("ALL", r["category"])]
                if not sel:
                    continue
                cells = [fmt(mean(sel, m)[0]) for m in METRICS]
                secs = statistics.mean(r["seconds"] for r in sel)
                lines.append(f"| {name} | {cat} | {len(sel)} | " + " | ".join(cells) + f" | {secs:.1f} |")
        lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nSummary: {run_dir / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["hybrid"])
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--questions", type=Path, default=HERE / "questions.yaml")
    ap.add_argument("--include-drafts", action="store_true", help="also run status: draft questions")
    ap.add_argument("--pool", action="store_true", help="write pooled top-20 candidates for labelling")
    ap.add_argument("--rescore", type=Path, help="a results/<run> folder: recompute metrics from its saved rows")
    args = ap.parse_args()

    allowed = {"approved", "draft"} if args.include_drafts or args.pool else {"approved"}
    questions = [q for q in yaml.safe_load(args.questions.read_text()) or []
                 if q.get("status") in allowed and (args.split == "all" or q.get("split") == args.split)]
    if not questions:
        sys.exit(f"No questions with status {sorted(allowed)} in split '{args.split}'; see evals/README.md")
    by_id = {q["id"]: q for q in questions}

    results = {}
    if args.rescore:
        run_dir = args.rescore.resolve()
        for f in sorted(run_dir.glob("*.jsonl"), key=lambda f: args.methods.index(f.stem)
                        if f.stem in args.methods else 99):
            rows = [json.loads(ln) for ln in open(f)]
            rows = [r for r in rows if r["id"] in by_id]
            for r in rows:
                q = by_id[r["id"]]
                r["category"], r["split"] = q["category"], q.get("split")
                r["metrics"] = score(q, r) if q["category"] != "unanswerable" and q.get("gold") else None
            results[f.stem] = rows
    else:
        run_dir = HERE / "results" / datetime.now().strftime("%Y-%m-%d-%H%M")
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in args.methods:
            retrieve, rows = load_method(name), []
            for q in questions:
                t0 = time.time()
                out = retrieve(q["question"], q.get("scope", ""), DEPTH)
                row = {"id": q["id"], "category": q["category"], "split": q.get("split"),
                       "seconds": round(time.time() - t0, 2), "error": out.get("error"),
                       "ranked": out["ranked"], "context": out.get("context", [])}
                row["top"] = note_ranking(row["ranked"])[:DEPTH]
                row["metrics"] = score(q, row) if q["category"] != "unanswerable" and q.get("gold") else None
                rows.append(row)
                print(f"{name:8} {q['id']:20} {row['seconds']:6.2f}s  {out.get('error') or ''}", flush=True)
            results[name] = rows
            with open(run_dir / f"{name}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.pool:
        lines = ["# Pooled candidates for labelling", ""]
        for i, q in enumerate(questions):
            gold = {g["path"] for g in q.get("gold") or []}
            pool = sorted({p for rows in results.values() for p in rows[i]["top"]} - gold)
            lines += [f"## {q['id']} ({q['category']})", q["question"], "",
                      *[f"- [ ] {p}" for p in pool], ""]
        out = run_dir.parent / f"pool-{run_dir.name}.md"
        out.write_text("\n".join(lines))
        print(f"\nPool written to {out}")
        return
    summarise(questions, results, args.split, run_dir)


if __name__ == "__main__":
    main()
