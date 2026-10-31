"""Hybrid method: BM25 over chunk text + embedding search over chunks, fused by reciprocal rank (k = 60).

Each list is collapsed to its best chunk per note and cut to 20 notes before fusion. This is the retrieval stage
only (no reranker), so both methods are compared on raw candidates. It calls the same function graph_search.py
uses as its base, so `hybrid` and `graph` start from identical rankings and differ only by what the graph adds.

In the original setup this ran through a separate hybrid-search tool that printed a 420-character excerpt per
note; the eval looked up the full chunk behind each excerpt so text-based metrics were not penalised by the cut.
Here the full chunk text is returned directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from graph_search import Graph  # noqa: E402

_graph = None


def retrieve(question, scope, k):
    global _graph
    try:
        _graph = _graph or Graph()
        chunks = _graph.hybrid_chunks(question, config.scope_prefix(scope), config.embed(question))[:k]
        texts = _graph.chunk_texts([c["chunk_id"] for c in chunks])
    except Exception as e:  # noqa: BLE001 — a failed run is scored as zero and listed, never dropped
        return {"ranked": [], "context": [], "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
    ranked = [{"path": c["path"], "chunk_id": c["chunk_id"], "text": texts[c["chunk_id"]]["text"]}
              for c in chunks if c["chunk_id"] in texts]
    return {"ranked": ranked, "context": [], "error": None}
