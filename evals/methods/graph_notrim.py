"""Graph method with the embedding relevance trim switched off (`--no-trim`): the latency ablation.

Same as graph.py otherwise. The trim embeds every candidate claim and link line to drop the irrelevant ones,
which is most of the graph method's latency; this variant measures what that costs in retrieval quality.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("graph", Path(__file__).with_name("graph.py"))
graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(graph)


def retrieve(question, scope, k):
    return graph.retrieve(question, scope, k, extra=["--no-trim"])
