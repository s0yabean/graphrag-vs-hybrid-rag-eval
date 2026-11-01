"""Scoring functions in evals/run_eval.py, checked against numbers worked out by hand."""

import math

import pytest
import run_eval

GOLD_A, GOLD_B, GOLD_C = "Decisions/A.md", "Journal/B.md", "Clients/C.md"


def question(category="temporal"):
    return {
        "id": "q1", "category": category,
        "gold": [{"path": GOLD_A, "grade": 2},
                 {"path": GOLD_B, "grade": 1, "why": "distractor: the old price"},
                 {"path": GOLD_C, "grade": 2}],
        "key_facts": [{"quote": "Harbour Blend goes to $28.00 per kg"},
                      {"quote": "“Copper Kettle” stopped ordering"}],
    }


def chunk(path, text="filler"):
    return {"path": path, "text": text}


def ranked_row():
    """Notes in rank order: X1, A, X2, X3, X4, X5, C, B (A also has a second, lower chunk)."""
    return {"context": [], "ranked": [
        chunk("X1.md"), chunk(GOLD_A, "**Harbour Blend** goes to $28.00 per kg"), chunk("X2.md"), chunk(GOLD_A),
        chunk("X3.md"), chunk("X4.md"), chunk("X5.md"),
        chunk(GOLD_C, "In October \"Copper Kettle\" stopped   ordering."), chunk(GOLD_B)]}


def test_note_ranking_keeps_best_rank_per_note():
    assert run_eval.note_ranking(ranked_row()["ranked"]) == [
        "X1.md", GOLD_A, "X2.md", "X3.md", "X4.md", "X5.md", GOLD_C, GOLD_B]


def test_ranking_metrics_match_hand_computation():
    m = run_eval.score(question(), ranked_row())
    assert m["recall@5"] == pytest.approx(1 / 3)       # only A in the top 5 notes
    assert m["recall@10"] == m["recall@20"] == 1.0
    assert m["all_gold@20"] == 1.0
    assert m["mrr"] == pytest.approx(1 / 2)             # first gold note at rank 2
    dcg = 3 / math.log2(3) + 3 / math.log2(8) + 1 / math.log2(9)   # A@2 (grade 2), C@7 (2), B@8 (1)
    idcg = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert m["ndcg@10"] == pytest.approx(dcg / idcg)
    assert m["ndcg@10"] == pytest.approx(0.5949, abs=1e-4)


def test_fact_recall_normalises_markdown_whitespace_and_smart_quotes():
    m = run_eval.score(question(), ranked_row())
    assert m["fact_recall@20"] == 1.0
    row = ranked_row()
    row["ranked"] = [r for r in row["ranked"] if r["path"] != GOLD_C]
    assert run_eval.score(question(), row)["fact_recall@20"] == 0.5


def test_current_first_needs_an_essential_note_above_every_distractor():
    assert run_eval.score(question(), ranked_row())["current_first"] == 1.0
    row = ranked_row()
    row["ranked"].insert(0, chunk(GOLD_B))
    assert run_eval.score(question(), row)["current_first"] == 0.0
    # Only temporal and modality questions are scored on it.
    assert run_eval.score(question("single_fact"), ranked_row())["current_first"] is None


def test_missing_gold_scores_zero_not_error():
    m = run_eval.score(question(), {"context": [], "ranked": [chunk("X1.md"), chunk("X2.md")]})
    assert m["recall@20"] == 0.0 and m["mrr"] == 0.0 and m["ndcg@10"] == 0.0 and m["all_gold@20"] == 0.0
    assert m["fact_recall@8k"] == 0.0 and m["precision@8k"] == 0.0


def test_8k_view_caps_context_lines_at_40_percent_then_fills_with_chunks():
    cap = 12_800                                                           # 40% of 8,000 tokens x 4 chars
    line = f"[2025-09-01] Harbour Blend · price: $28 — \"q\" ({GOLD_A})"
    n_fit = int(cap // len(line))
    row = {"context": [line] * (n_fit + 50),
           "ranked": [chunk("X1.md", "x" * 50_000), chunk(GOLD_C, "never reached")]}
    view = run_eval.delivered(row)
    ctx = [t for t, _ in view if t == line]
    assert len(ctx) == n_fit                                               # extra lines dropped at the cap
    assert sum(len(t) for t, _ in view) == 32_000                          # the chunk fills the rest exactly
    assert all(GOLD_C not in paths for _, paths in view)

    m = run_eval.score(question(), row)
    assert m["note_recall@8k"] == pytest.approx(1 / 3)                     # A, cited by the context lines
    assert m["precision@8k"] == pytest.approx(n_fit * len(line) / 32_000)


def test_sign_test_reproduces_published_p_values():
    # README: hybrid wins 15, loses 5 -> p = 0.04; trim off: 1 better, 12 worse -> p = 0.003.
    assert round(run_eval.sign_test(15, 5), 2) == 0.04
    assert round(run_eval.sign_test(1, 12), 3) == 0.003
    assert run_eval.sign_test(3, 3) == 1.0
    assert run_eval.sign_test(0, 0) == 1.0
    assert run_eval.sign_test(5, 15) == run_eval.sign_test(15, 5)


def test_mean_skips_unscored_rows():
    rows = [{"metrics": {"mrr": 1.0}}, {"metrics": {"mrr": 0.0}}, {"metrics": None}, {"metrics": {"mrr": None}}]
    assert run_eval.mean(rows, "mrr") == (0.5, 2)
    assert run_eval.mean([{"metrics": None}], "mrr") == (None, 0)
