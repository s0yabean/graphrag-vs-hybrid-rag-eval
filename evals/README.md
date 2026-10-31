# Retrieval eval: hybrid RAG vs graph

Compares methods that find evidence in the vault for the same question. Each method is a file in `methods/` with
one function, `retrieve(question, scope, k) -> {"ranked": [...], "context": [...], "error": ...}`.

| Method | How it retrieves |
|---|---|
| `hybrid` | BM25 over chunk text + embedding search over chunks, best chunk per note, fused by RRF (k = 60) |
| `graph` | `graph_search.py ask`: the same hybrid list, plus typed links and dated claims as context lines, plus chunks only the graph reached |
| `graph_notrim` | `graph` with the embedding relevance trim switched off (the latency ablation) |

This measures retrieval only. Answer quality (given the same context budget, does the answering model state the
right, current facts?) is a separate eval and not built here.

## Question types

| Category | What it asks | Expected winner |
|---|---|---|
| `single_fact` | One fact stated in one note (the control group) | Hybrid or tie |
| `multi_hop` | Answer needs 2–3 notes joined through an entity | Graph |
| `entity_rollup` | Every X related to Y across many notes | Graph |
| `temporal` | Latest value when notes disagree over time | Graph (dated claims) |
| `modality_trap` | Something was planned or AI-suggested but did not happen, or the reverse | Graph (`modality`) |
| `broad_theme` | Synthesis across the vault | Neither clearly |
| `paraphrase_alias` | Worded with none of the note's terms, or an alias | Dense part of hybrid, graph aliases |
| `disambiguation` | One name, several entities | Graph, if resolution keeps them apart |
| `unanswerable` | Plausible but not in the notes; `traps` lists decoy values | Scored only at the answer level |

## Rules for writing questions

- Start from the evidence, then write the question. Do not copy the note's wording, except in `single_fact`.
- Multi-hop must really need the hop: record each hop's note as `grade: 2`.
- Temporal and modality questions need a distractor: the outdated or planned-only note as `grade: 1` with a
  `why` starting "distractor".
- Label `key_facts[].modality` the way the extractor does (`fact`, `reported`, `plan`, `opinion`,
  `hypothetical`, `ai_suggestion`), with an exact `quote` from a gold note.
- Split before you tune: about a third `dev`, the rest `test`.
- Label fairly with pooling: `run_eval.py --pool` writes the union of every method's top 20 per question, so
  you can add relevant notes you did not remember.

`validate.py` checks the schema, that gold paths exist in the graph, and that every quote appears in a gold note.

## Metrics

| Metric | Meaning |
|---|---|
| `recall@10/20`, `all_gold@20`, `mrr`, `ndcg@10` | Did the right notes rank, and how early? |
| `fact_recall@8k` | Share of key-fact quotes inside the first 8,000 tokens the method hands the answerer |
| `note_recall@8k`, `precision@8k` | Gold notes inside that budget, and the share of the budget they fill |
| `current_first` | Temporal and modality questions: does an essential note rank above every distractor? |
| sign test | Over per-question wins and losses, is the gap more than chance? |

The 8,000-token view gives the graph's claim and link lines at most 40% of the budget, the same split
`graph_search.py` uses, then fills the rest with chunks in rank order.

On the 54-note sample vault, 20 notes are a large share of the corpus, so the 8k metrics saturate quickly; the
ranking metrics (`recall@10`, `mrr`) are the more telling ones there.
