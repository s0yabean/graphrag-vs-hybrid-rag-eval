"""Hybrid retrieval (BM25 + embeddings, best chunk per note, RRF k=60) through graph_search.Graph.hybrid_chunks,
with Neo4j replaced by the in-memory FakeGraph from conftest."""

import config
import graph_search
import pytest


def test_rrf_fusion_order_from_two_fixed_lists(monkeypatch):
    bm25 = [("a.md", "a.md::0"), ("b.md", "b.md::1"), ("b.md", "b.md::0"), ("c.md", "c.md::0")]
    vec = [("c.md", "c.md::0"), ("b.md", "b.md::0"), ("d.md", "d.md::0")]

    class Stub(graph_search.Graph):
        def __init__(self):
            pass

        def q(self, cypher, **p):
            if "'chunk_text'" in cypher:
                return [{"path": a, "chunk_id": b} for a, b in bm25]
            if "'chunk_embedding'" in cypher:
                return [{"path": a, "chunk_id": b} for a, b in vec]
            return [{"id": i, "e": []} for i in p["ids"]]

    out = Stub().hybrid_chunks("q", "", [0.0])
    # b: 1/62 + 1/62; c: 1/63 + 1/61; a: 1/61; d: 1/63.  1/62+1/62 = 0.032258 < 1/63+1/61 = 0.032266
    assert [r["path"] for r in out] == ["c.md", "b.md", "a.md", "d.md"]
    # A note keeps the chunk of its best (first) appearance in the first list that has it.
    assert {r["path"]: r["chunk_id"] for r in out}["b.md"] == "b.md::1"


@pytest.mark.parametrize("question, expected", [
    ("How much grant money did the roastery get from the local business fund?",
     "Funding/Port Alder Small Business Fund grant.md"),
    ("When did the home coffee subscription Fernwick at Home launch?", "Operations/Fernwick at Home.md"),
    ("Did Fernwick open its second roastery in Millbrook?", "Decisions/2026-01 Shelve Millbrook roastery.md"),
])
def test_sample_queries_rank_the_right_note_first(fake_graph, question, expected):
    out = fake_graph().hybrid_chunks(question, "", config.embed(question))
    assert out[0]["path"] == expected
    assert len(out) == graph_search.HYBRID_NOTES
    assert len({r["path"] for r in out}) == len(out)            # one chunk per note


def test_scope_prefix_filters_both_lists(fake_graph):
    out = fake_graph().hybrid_chunks("wholesale price per kg", config.scope_prefix("Clients"),
                                     config.embed("wholesale price per kg"))
    assert out and all(r["path"].startswith("Clients/") for r in out)
    # "Clients" must not also match a sibling folder such as "ClientsArchive/".
    assert config.scope_prefix("Clients") == "Clients/" and config.scope_prefix("") == ""


def test_hybrid_method_returns_full_chunk_text(fake_graph, monkeypatch):
    from methods import hybrid
    monkeypatch.setattr(hybrid, "Graph", fake_graph)
    monkeypatch.setattr(hybrid, "_graph", None)
    out = hybrid.retrieve("Who supplies Fernwick's green coffee today?", "", 10)
    assert out["error"] is None and len(out["ranked"]) == 10
    top = out["ranked"][0]
    assert top["chunk_id"].startswith(top["path"] + "::") and len(top["text"]) > 100


def test_hybrid_method_reports_errors_instead_of_raising(monkeypatch):
    from methods import hybrid

    class Broken:
        def __init__(self):
            raise RuntimeError("Neo4j is not running\nstack")

    monkeypatch.setattr(hybrid, "Graph", Broken)
    monkeypatch.setattr(hybrid, "_graph", None)
    out = hybrid.retrieve("anything", "", 5)
    assert out == {"ranked": [], "context": [], "error": "RuntimeError: Neo4j is not running"}
