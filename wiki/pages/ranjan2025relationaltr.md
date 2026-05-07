---
title: "Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data"
tags: [source, relational-deep-learning, foundation-model, transformer, zero-shot]
sources: [ranjan2025relationaltr]
updated: 2026-05-06
---

# Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data

**Source:** https://arxiv.org/abs/2510.06377
**Title:** Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data
**Date ingested:** 2026-04-28
**Type:** paper
**Authors:** Rishabh Ranjan, Valter Hudovernik, Mark Znidar, Charilaos Kanatsoulis, Roshan Upendra, Mahmoud Mohammadi, Joe Meyer, Tom Palczewski, Carlos Guestrin, Jure Leskovec
**Venue:** ICLR 2026

## Summary

- **What:** Row-level tokenization (as used by GNN/GT approaches like RelGT) makes cross-schema transfer impossible, blocking zero-shot generalization to unseen relational databases.
- **How:** RT tokenizes at the *cell* level — each (value, column, table) triple is a token — enabling schema-agnostic self-supervised pretraining via masked token prediction, with relational structure encoded through four specialized attention masks.
- **So what:** A 22M-parameter RT achieves 93% of supervised AUROC zero-shot on RelBench, outperforming Gemma3-27B (84%) at 10⁵× lower inference FLOPs.

## Challenges & Novelty

Foundation models for relational data require schema-agnostic representations — databases differ in table structure, column types, and relational topology, so any row-level or schema-specific encoding cannot transfer across databases. RT makes the key observation that cells (individual values) are the universal atomic unit of relational data, enabling a single pretraining objective across arbitrary schemas.

- **Schema diversity blocks transfer:** row-level GNN/GT tokenization embeds schema-specific structure; a row from one database has no meaning in another database's representation space.
- **Relational structure without graphs:** encoding primary-foreign key relationships, parent-child row structure, and column semantics requires specialized attention patterns, not general-purpose full attention.
- **Unifying forecasting and autocomplete tasks:** prior RDL work handles only forecasting (predict future values); RT additionally handles autocomplete (fill in missing cell values) and unifies both as masked token prediction.

## Relation to Prior Work

| Model | Tokenization | PE | Zero-shot | Pretraining |
|---|---|---|---|---|
| HeteroGNN (RDL baseline) | Row (node) | No | No | No |
| [dwivedi2025relgt](dwivedi2025relgt.md) | Row (node) | 5-element | No | No |
| **RT** | Cell | None | Yes | Yes (MTP) |

- [dwivedi2025relgt](dwivedi2025relgt.md): the key contrast — RelGT uses rich row-level schema-specific encodings for best supervised performance; RT omits all PEs for zero-shot generalization. Same problem, opposite design philosophy.
- [relbench](relbench.md): RT pretrained leave-one-DB-out on RelBench and evaluated zero-shot; also evaluated on RelBench v2 autocomplete tasks.
- [relational-foundation-model](relational-foundation-model.md): RT is the first pretrained foundation model for relational databases; RelGT is the strongest supervised model.

## Technical Details

**Cell-level tokenization.** Each cell $(v, c, t)$ — value $v$ in column $c$ of table $t$ — is a universal token. Column and table embeddings are added to encode schema context. This representation is identical across any database schema.

**Task table prompting.** Task specifications (forecasting targets, autocomplete targets) are expressed as ordinary relational tables appended to the database. Both task types reduce to masked token prediction (MTP) — the model fills in masked cells.

**Relational Attention.** Four specialized attention types replace standard full attention, implemented with FlexAttention/FlashAttention sparse kernels:
1. *Column attention* — cells attend within their column (models value distributions per column)
2. *Feature attention* — cells attend within a row and its F→P parent rows (joined-table row attention)
3. *Neighbor attention* — propagates signals from P→F child rows (analogous to GNN message passing)
4. *Global full attention* — arbitrary pairwise cell interactions for long-range context

**No positional encodings** — deliberately omitted to preserve permutation invariance across tables, rows, and columns, which is critical for zero-shot transfer across schemas.

**Pretraining.** Pretrained on 6/7 RelBench databases (leave-one-DB-out) with masked token prediction. 22M parameters.

## Experiments

- Zero-shot RT achieves 93% of supervised AUROC on binary classification tasks — vs. 84% for Gemma3-27B at 10⁵× more inference FLOPs.
- Continued pretraining on the target database (not task) reaches 93.1% and fine-tunes 10–100× more sample-efficiently than baselines.
- RT underperforms RelGT in the fully-supervised setting — the PE-free design trades peak supervised accuracy for zero-shot generalizability.

## Entities & Concepts

- [relational-foundation-model](relational-foundation-model.md)
- [relational-attention](relational-attention.md)
- [relational-deep-learning](relational-deep-learning.md)
- [relational-entity-graph](relational-entity-graph.md)
- [relbench](relbench.md)
- [autocomplete-tasks](autocomplete-tasks.md)
- [graph-transformer](graph-transformer.md)
