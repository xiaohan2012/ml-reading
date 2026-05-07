---
title: Relational Attention
tags: [concept, attention, transformer, relational-deep-learning]
sources: [ranjan2025relationaltr]
updated: 2026-04-28
---

# Relational Attention

## Description

Relational Attention is the core mechanism of the [Relational Transformer (RT)](ranjan2025relationaltr.md). It replaces standard full self-attention with four specialized attention types, each encoding a distinct aspect of relational database structure via sparse attention masks. The fundamental unit is the **cell token** (value, column, table).

| Attention type | Mask | Captures |
|---|---|---|
| **Column attention** | Same column | Value distribution within a column |
| **Feature attention** | Same row ∪ F→P linked rows | Feature mixing within an entity and its parents (equivalent to joined-table row attention) |
| **Neighbor attention** | P→F linked rows | Aggregation from child rows — analogous to GNN message passing |
| **Global attention** | All pairs | Arbitrary pairwise interactions |

Each transformer block applies all four types sequentially. Column, feature, and neighbor attention capture relational structure inductive biases; global attention provides standard transformer expressiveness.

**Key design choice**: **no positional encodings** — RT deliberately omits them to preserve permutation invariance over tables, rows, and columns. This enables zero-shot generalization to unseen schemas. Contrast with [RelGT](dwivedi2025relgt.md)'s 5-element tokenization, which encodes hop distance and node type precisely but is tied to a specific schema.

**Implementation**: sparse attention masks compiled to FlashAttention-based kernels using FlexAttention.

## Appearances in Sources

- [ranjan2025relationaltr](ranjan2025relationaltr.md) — defined and used as the core architectural mechanism (Section 3.2)

## Related Concepts

- [ranjan2025relationaltr](ranjan2025relationaltr.md) — the model that uses this mechanism
- [multi-element-tokenization](multi-element-tokenization.md) — RelGT's contrasting approach: row-level tokens with rich structural encodings
- [graph-transformer](graph-transformer.md) — RT's neighbor attention is analogous to GNN message passing in graph transformers
