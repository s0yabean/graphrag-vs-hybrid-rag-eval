# Published results (aggregate only)

These numbers come from the private run described in the article
[GraphRAG vs hybrid RAG: testing Microsoft Research's claim on 13,000 private notes](https://tonytonggg.com/writing/graphrag-vs-hybrid-rag-on-private-notes).
They were measured on a private Obsidian vault with 47 hand-labelled questions (44 scored, 3 unanswerable) that
are not in this repo. Running the quickstart on `sample_vault/` will not reproduce them.

## Setup

| | |
|---|---|
| Corpus | 13,284 Markdown notes; the eval questions cover one 623-note folder |
| Graph (one folder) | 4,779 chunks, 5,673 entities, 5,304 typed relationships, 8,005 claims; about $7 of Claude Haiku 5.5 |
| Graph (13 folders) | 31,047 chunks, 40,903 entities, 34,459 typed relationships, 52,866 claims; about $28 in total |
| Embeddings | `bge-m3` in Ollama, local |
| Questions | 47 across 9 types; dev 17, test 30 |

## Round 1: graph search on its own (44 scored questions)

| Metric | Hybrid | Graph |
|---|---|---|
| Gold notes in the top 10 | 0.53 | 0.39 |
| Key facts within 8,000 tokens | 0.47 | 0.37 |

Gold notes in the top 10: hybrid won 15 questions, lost 5, tied 24 (sign test p = 0.04).

## Round 2: graph layered on hybrid search

| Question type | Key facts (8k), hybrid / hybrid + graph | Gold notes within budget, hybrid / hybrid + graph |
|---|---|---|
| Temporal | 0.28 / 0.36 | 0.40 / 0.53 |
| Modality trap | 0.40 / 0.40 | 0.60 / 0.47 |
| Broad theme | 0.27 / 0.30 | 0.38 / 0.43 |
| All 44 scored | 0.47 / 0.48 | 0.52 / 0.52 |

Overall a tie: 3 questions better, 3 worse, 38 the same.

## Round 3: the same questions on a graph seven times larger

| Over the 44 scored questions | Hybrid | Hybrid + graph | Hybrid + graph, no trim | Hybrid + graph, after the lookup fix |
|---|---|---|---|---|
| Key facts within 8,000 tokens | 0.51 | 0.50 | 0.40 | 0.49 |
| Gold notes within 8,000 tokens | 0.54 | 0.56 | 0.50 | 0.56 |
| Gold notes in the top 10 | 0.55 | 0.55 | 0.55 | 0.55 |
| Better / worse than hybrid on key facts | – | 1 / 5 (p = 0.22) | 1 / 12 (p = 0.003) | 1 / 6 (p = 0.13) |
| Median latency | 0.8 s | 9.2 s | 0.5 s | 4.2 s |
| 95th-percentile latency | 0.9 s | 13.2 s | 0.6 s | 6.0 s |

The "lookup fix" ignores pronouns and requires one-word aliases to match with their capitals; it is included in
`graph_search.py`.
