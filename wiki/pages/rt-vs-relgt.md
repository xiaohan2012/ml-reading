---
title: "RT vs. RelGT: Relational Transformer vs. Relational Graph Transformer"
tags: [query, analysis, relational-deep-learning, graph-transformer, foundation-model]
sources: [ranjan2025relationaltr, dwivedi2025relgt]
updated: 2026-04-29
---

# RT vs. RelGT: Relational Transformer vs. Relational Graph Transformer

## Comparison Table

| Dimension | [RelGT](dwivedi2025relgt.md) | [RT](ranjan2025relationaltr.md) |
|---|---|---|
| **Granularity** | Row-level tokens (one token per table row/entity) | Cell-level tokens (one token per `(value, column, table)` triple) |
| **Positional encodings** | Rich 5-element tokenization: node type + hop distance + relative time + subgraph GNN PE | None — deliberately omitted for permutation invariance |
| **Schema dependence** | Schema-specific: encodings are tied to a fixed DB schema | Schema-agnostic: cell tokens are universal across any schema |
| **Attention scope** | Local all-pair over K=300 sampled subgraph tokens + global to B=4096 EMA centroid tokens | Four typed attention masks: column / feature / neighbor / global |
| **Pretraining** | No — supervised only | Yes — masked token prediction (MTP) across 6 RelBench databases |
| **Zero-shot transfer** | No — must be trained per schema | Yes — 93% of supervised AUROC zero-shot on unseen databases |
| **Best setting** | Supervised, schema-known tasks | Low-data or cross-database generalization |
| **Model size** | Not specified (schema-specific) | 22M parameters |
| **Venue** | Unknown 2025 | ICLR 2026 |
| **Authors** | Unknown | Ranjan, Hudovernik, Znidar, Kanatsoulis et al. (Leskovec group) |

## The Core Philosophical Split

**RelGT** bets that *more structural information = better performance*. It invests in the most information-rich tokenization possible: each node gets 5 separate encoding elements capturing its type, position in the graph, temporal distance, and structural role (via subgraph GNN PE). The subgraph GNN PE with random resampled initialization is its most critical component. This makes it the best supervised method on RelBench — but it can only work on databases whose schema it was trained on.

**RT** bets that *schema-agnosticism = scalability*. By tokenizing at the cell level, every database looks the same to the model — a bag of `(value, column, table)` triples linked by attention masks. Dropping all positional encodings is a deliberate choice: it preserves permutation invariance over tables, rows, and columns, enabling zero-shot transfer. The structural information that RelGT encodes as explicit positional signals is instead encoded implicitly in RT's four attention mask types (column / feature / neighbor / global).

## What They Agree On

Both treat FK links as the primary relational signal. RT's **neighbor attention** (P→F child rows) is the direct analog of RelGT's local graph attention — both are doing GNN-style message passing, just at different granularities (cell vs. row) and with different parameterization.

Both also beat LLMs: RelGT beats Griffin on 8/10 tasks; RT at 22M params beats Gemma3-27B at zero-shot relational classification.

## Open Question

Can RT's zero-shot approach match RelGT's supervised performance after fine-tuning at scale? Neither paper tests the other's regime, so the tradeoff is not yet fully resolved.

## Related Concepts

- [ranjan2025relationaltr](ranjan2025relationaltr.md) — RT source page
- [dwivedi2025relgt](dwivedi2025relgt.md) — RelGT source page
- [relational-deep-learning](relational-deep-learning.md) — the shared problem setting
- [relational-foundation-model](relational-foundation-model.md) — RT is the current best zero-shot approach
- [multi-element-tokenization](multi-element-tokenization.md) — RelGT's 5-element tokenization scheme
- [relational-attention](relational-attention.md) — RT's four-type attention mechanism
