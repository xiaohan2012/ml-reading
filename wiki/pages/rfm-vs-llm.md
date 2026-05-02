---
title: "RFMs vs. LLMs: Inputs, Outputs, and Key Differences"
tags: [query, analysis, relational-deep-learning, foundation-model]
sources: [kumorfm-1, kumorfm-2, relational-transformer, griffin]
updated: 2026-04-29
---

# RFMs vs. LLMs: Inputs, Outputs, and Key Differences

## How RFMs Differ from LLMs

| Dimension | LLM (Transformer-based) | Relational Foundation Model |
|---|---|---|
| **Native input** | Text tokens (sequential) | Multi-table graph: rows as nodes, PK-FK links as edges |
| **Schema handling** | Serializes tables to text (JSON/natural language) — brittle | Native column-type encoders per semantic type (numerical, categorical, timestamp, text, embedding) |
| **Training objective** | Next-token prediction over text corpora | Masked token/cell prediction over relational databases ([relational-transformer](relational-transformer.md)) or structured prediction + ICL ([kumorfm-1](kumorfm-1.md)) |
| **Temporal awareness** | None — no concept of seed_time or leakage | Time-consistent subgraph snapshots `G^{≤t}[e]`; context examples always from `t̂ < t` |
| **Numerical handling** | Poor — text tokens don't preserve numerical semantics | Dedicated float encoder ([griffin](griffin.md)), numerical column encoders ([kumorfm-1](kumorfm-1.md)), or raw cell values ([relational-transformer](relational-transformer.md)) |
| **Relational structure** | Lost in serialization | Preserved: PK-FK topology is the graph structure; message passing or attention over it |
| **Output** | Text tokens | Structured: class probabilities, regression values, ranked item lists |
| **Zero-shot performance** | Gemma3-27B achieves 84% of supervised AUROC on RelBench binary classification | RT (22M params) achieves 93% — 10⁵× fewer FLOPs ([relational-transformer](relational-transformer.md)) |

The [kumorfm-1](kumorfm-1.md) whitepaper explicitly names the mismatch: LLMs' training objective (next-token prediction) is fundamentally misaligned with the goal (minimizing forecast error). They also hallucinate, struggle with numerical patterns, and suffer from potential data leakage on publicly available datasets.

## Inputs to an RFM

All three RFMs in the wiki share a common input structure:

**1. The relational database subgraph** `G^{≤t}[e]`
- k-hop neighborhood around the seed entity `e`, filtered to timestamp ≤ `t`
- Contains rows from multiple tables, linked by PK-FK edges
- Each row has heterogeneous multi-modal cells (numbers, strings, timestamps, embeddings)

**2. A task specification**
- **KumoRFM** ([kumorfm-1](kumorfm-1.md)): a PQL query — `PREDICT COUNT(orders.*, 0, 7) FOR users.user_id = 0`. Defines the target, entity, and time window declaratively.
- **RT** ([relational-transformer](relational-transformer.md)): a *task table* — an ordinary relational table appended to the database with `(EntityID, seed_time, label)` rows. The masked cell to predict is the label column.
- **Griffin** ([griffin](griffin.md)): a task embedding derived from the target column name (text encoding), used as the query in its cross-attention module.

**3. Context examples** (for ICL-based models)
- A set of `{(G^{≤t̂}[ê], y^{t̂}_ê)}` pairs — historical entity subgraphs from `t̂ < t` with known labels
- KumoRFM generates these on-the-fly via temporal neighbor sampling ([kumorfm-1](kumorfm-1.md))
- RT generates them via the training table abstraction at pretraining time

## Outputs

| Model | Output |
|---|---|
| KumoRFM | Class probabilities (binary/multi-class), regression value, ranked item list; plus optional embeddings + explanations |
| RT | Predicted cell values for masked positions (same format as pretraining MTP) |
| Griffin | Classification logits (inner product with label text embeddings) or regression value (float decoder) |

## Architectural Split Across RFMs

The three RFMs make different tradeoffs on *where* relational structure is encoded:

- **RT** ([relational-transformer](relational-transformer.md)): cell-level tokens + attention masks encoding column/feature/neighbor/global structure — no positional encodings, fully permutation-invariant, best for zero-shot transfer
- **KumoRFM** ([kumorfm-1](kumorfm-1.md), [kumorfm-2](kumorfm-2.md)): row-level tokens → RelGT (graph attention over FK edges) → ICL module — row-level but schema-agnostic via the type encoder; strongest overall on RelBench
- **Griffin** ([griffin](griffin.md)): cross-attention within rows (task-conditioned feature selection) + GNN message passing across tables — pretraining-focused, strongest in low-data regimes

## Related Pages

- [relational-foundation-model](relational-foundation-model.md) — overview of all RFM approaches
- [relational-deep-learning](relational-deep-learning.md) — the RDL paradigm RFMs build on
- [relational-entity-graph](relational-entity-graph.md) — the graph structure that is the input
- [training-table](training-table.md) — the (EntityID, seed_time, label) abstraction used as context/task spec
- [rdl-approaches](rdl-approaches.md) — broader comparison of all RDL methods including non-foundation supervised models
