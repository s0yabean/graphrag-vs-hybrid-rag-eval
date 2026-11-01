"""Shared fixtures. No Neo4j, no embedding model, no API key.

`FakeGraph` stands in for graph_search.Graph: it answers the hybrid queries (BM25 full-text, vector search, chunk
lookups) from an in-memory index of the sample vault's chunks, built with build_chunks.build_rows. The embedder is
a deterministic hashed bag of words, so "semantic" similarity is word overlap: enough to test ranking and fusion
without downloading a model.
"""

import hashlib
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

# Settings are read at import time; pin the ones the sample vault needs before any repo module loads.
os.environ.update({"VAULT_PATH": "sample_vault", "AUTHOR_NAME": "Ruth Calloway", "HUB_MENTIONS": "15",
                   "EMBED_BACKEND": "fastembed", "NEO4J_PASSWORD": ""})

import build_chunks  # noqa: E402
import config  # noqa: E402
import graph_search  # noqa: E402

DIMS = 256
WORD_RE = re.compile(r"[a-z0-9]+")


def words(text):
    return WORD_RE.findall(text.lower())


def fake_embed_many(texts):
    out = []
    for t in texts:
        v = [0.0] * DIMS
        for w in words(t):
            if w in graph_search.STOP:
                continue
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIMS] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def fake_embed(text):
    return fake_embed_many([text])[0]


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    for mod in (config, graph_search):
        monkeypatch.setattr(mod, "embed_many", fake_embed_many)
        monkeypatch.setattr(mod, "embed", fake_embed)


@pytest.fixture(scope="session")
def sample_rows():
    return build_chunks.build_rows(config.vault_path(), "")


class FakeGraph(graph_search.Graph):
    """In-memory replacement for the three Cypher queries hybrid retrieval runs."""

    rows = None  # set by the fixture

    def __init__(self):
        rows = type(self).rows
        self.chunks = {r["id"]: r for r in rows}
        self.tf = {r["id"]: Counter(words(r["breadcrumb"] + " " + r["text"])) for r in rows}
        self.vec = {r["id"]: fake_embed(r["breadcrumb"] + "\n\n" + r["text"]) for r in rows}
        n = len(rows)
        df = Counter(w for tf in self.tf.values() for w in tf)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.avg_len = sum(sum(tf.values()) for tf in self.tf.values()) / n
        self.queries = []

    def bm25(self, query):
        terms = [t.replace("\\", "") for t in query.split(" OR ")]
        terms = [w for t in terms for w in words(t)]
        scores = {}
        for cid, tf in self.tf.items():
            length = sum(tf.values())
            s = sum(self.idf.get(w, 0) * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * length / self.avg_len))
                    for w in terms if w in tf)
            if s > 0:
                scores[cid] = s
        return scores

    def q(self, cypher, **p):
        self.queries.append(cypher)
        if "'chunk_text'" in cypher:
            scored = self.bm25(p["q"])
        elif "'chunk_embedding'" in cypher:
            scored = {cid: graph_search.cosine(p["vec"], v) for cid, v in self.vec.items()}
        elif "MENTIONS" in cypher:
            return [{"id": i, "e": []} for i in p["ids"]]
        elif "c.breadcrumb AS breadcrumb" in cypher:
            return [{"id": i, "path": self.chunks[i]["note"], "breadcrumb": self.chunks[i]["breadcrumb"],
                     "text": self.chunks[i]["text"]} for i in p["ids"] if i in self.chunks]
        else:
            raise AssertionError(f"FakeGraph does not handle this query:\n{cypher}")
        hits = [(cid, s) for cid, s in scored.items() if self.chunks[cid]["note"].startswith(p["prefix"])]
        hits.sort(key=lambda x: (-x[1], x[0]))
        return [{"chunk_id": cid, "path": self.chunks[cid]["note"], "score": s} for cid, s in hits[:p["chunks"]]]


@pytest.fixture
def fake_graph(sample_rows, monkeypatch):
    FakeGraph.rows = sample_rows
    monkeypatch.setattr(graph_search, "Graph", FakeGraph)
    return FakeGraph
