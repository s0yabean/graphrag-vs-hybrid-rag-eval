"""evals/run_eval.py end to end on the synthetic sample: real questions, the real hybrid method file, Neo4j
replaced by FakeGraph. The graph methods call graph_search.py in a subprocess against Neo4j, so they only run
when a database is configured (GRAPHRAG_NEO4J_TESTS=1 after the quickstart)."""

import json
import os
import sys
from pathlib import Path

import pytest
import run_eval
import yaml

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals" / "questions.yaml"


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """Point run_eval at a temp folder (results land there) that still sees the real methods/."""
    (tmp_path / "methods").symlink_to(ROOT / "evals" / "methods", target_is_directory=True)
    monkeypatch.setattr(run_eval, "HERE", tmp_path)
    return tmp_path


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run_eval.py", *argv])
    run_eval.main()


def approved():
    return [q for q in yaml.safe_load(QUESTIONS.read_text()) if q.get("status") == "approved"]


def test_hybrid_run_writes_rows_and_summary(fake_graph, run_dir, monkeypatch):
    run(monkeypatch, "--methods", "hybrid", "--questions", str(QUESTIONS))
    (out,) = (run_dir / "results").iterdir()
    rows = [json.loads(ln) for ln in (out / "hybrid.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [q["id"] for q in approved()]
    assert all(r["error"] is None and r["top"] for r in rows)
    scored = [r for r in rows if r["metrics"]]
    assert len(scored) == len(rows) - 1                         # the unanswerable question is not scored
    assert next(r for r in rows if r["category"] == "unanswerable")["metrics"] is None

    mean = run_eval.mean(scored, "recall@10")[0]
    assert 0.5 < mean <= 1.0                                    # the sample is easy; a broken ranker scores ~0
    single = next(r for r in rows if r["id"] == "fw-single-01")
    assert single["metrics"]["mrr"] == 1.0 and single["metrics"]["fact_recall@8k"] == 1.0

    summary = (out / "summary.md").read_text()
    assert "## All metrics, dev split" in summary and "## All metrics, test split" in summary
    assert "| hybrid | ALL |" in summary and "Failed runs" not in summary


def test_rescore_two_methods_gives_verdict_and_sign_test(fake_graph, run_dir, monkeypatch):
    run(monkeypatch, "--methods", "hybrid", "--questions", str(QUESTIONS))
    (out,) = (run_dir / "results").iterdir()
    # A second "method": the same rows with every ranking reversed, which must lose.
    rows = [json.loads(ln) for ln in (out / "hybrid.jsonl").read_text().splitlines()]
    with open(out / "reversed.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({**r, "ranked": r["ranked"][::-1]}) + "\n")

    run(monkeypatch, "--rescore", str(out), "--methods", "hybrid", "reversed", "--questions", str(QUESTIONS))
    summary = (out / "summary.md").read_text()
    assert "## Verdict by question type (hybrid vs reversed, both splits)" in summary
    verdict = next(ln for ln in summary.splitlines() if ln.startswith("| ALL |"))
    assert "**hybrid**" in verdict
    tally = next(ln for ln in summary.split("## Per-question recall@10")[1].splitlines() if ln.startswith("| ALL |"))
    wins, ties, losses = (int(x) for x in tally.split("|")[2:5])
    assert wins + ties + losses == len(rows) - 1 and wins > losses


def test_failed_method_is_scored_zero_and_listed(run_dir, monkeypatch):
    (run_dir / "methods").unlink()
    (run_dir / "methods").mkdir()
    (run_dir / "methods" / "broken.py").write_text(
        "def retrieve(question, scope, k):\n    return {'ranked': [], 'context': [], 'error': 'down'}\n")
    run(monkeypatch, "--methods", "broken", "--questions", str(QUESTIONS), "--split", "dev")
    (out,) = (run_dir / "results").iterdir()
    summary = (out / "summary.md").read_text()
    assert "## Failed runs (scored as zero)" in summary and "`broken` fw-single-01: down" in summary
    rows = [json.loads(ln) for ln in (out / "broken.jsonl").read_text().splitlines()]
    assert all(r["split"] == "dev" for r in rows)
    assert all(r["metrics"]["recall@20"] == 0.0 for r in rows if r["metrics"])


def test_questions_file_is_well_formed():
    """The checks validate.py makes that need no database."""
    qs = yaml.safe_load(QUESTIONS.read_text())
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids))
    vault = ROOT / "sample_vault"
    for q in qs:
        assert q["split"] in ("dev", "test")
        for g in q.get("gold") or []:
            assert (vault / g["path"]).exists(), g["path"]
            assert g["grade"] in (1, 2)
        text = "\n".join(run_eval.norm(Path(g["path"]).stem + "\n" + (vault / g["path"]).read_text())
                         for g in q.get("gold") or [])
        for f in q.get("key_facts") or []:
            assert run_eval.norm(f["quote"]) in text, f"{q['id']}: {f['quote']}"
        if q["category"] == "multi_hop":
            assert sum(g["grade"] == 2 for g in q["gold"]) >= 2


@pytest.mark.skipif(os.environ.get("GRAPHRAG_NEO4J_TESTS") != "1",
                    reason="needs the quickstart's Neo4j (set GRAPHRAG_NEO4J_TESTS=1)")
def test_graph_method_against_live_neo4j():
    from methods import graph
    out = graph.retrieve("Did Fernwick open its second roastery in Millbrook?", "", 10)
    assert out["error"] is None and out["ranked"] and out["context"]
