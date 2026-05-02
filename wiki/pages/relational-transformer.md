---
title: "Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data"
tags: [source, relational-deep-learning, foundation-model, transformer, zero-shot]
sources: [relational-transformer]
updated: 2026-04-28
---

# Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data

**Source:** https://arxiv.org/abs/2510.06377
**Title:** Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data
**Date ingested:** 2026-04-28
**Type:** paper
**Authors:** Rishabh Ranjan, Valter Hudovernik, Mark Znidar, Charilaos Kanatsoulis, Roshan Upendra, Mahmoud Mohammadi, Joe Meyer, Tom Palczewski, Carlos Guestrin, Jure Leskovec
**Venue:** ICLR 2026
**Year:** 2025

## Summary

The Relational Transformer (RT) is the first architecture pretrained on diverse relational databases and capable of zero-shot generalization to unseen datasets and tasks without fine-tuning. The core challenge RT addresses is schema diversity: databases differ in table structure, column types, and relational topology, so tokenizing at the row level (as GNN/GT approaches do, including RelGT) makes cross-schema transfer impossible. RT instead tokenizes at the *cell* level — each (value, column, table) triple is a token — enabling schema-agnostic self-supervised learning via masked token prediction (MTP). Task specifications are delivered through *task table prompting*, where task-defining rows are appended as an ordinary relational table to the database, letting any forecasting or autocomplete task be expressed in the same input space as pretraining.

The central architectural innovation is *Relational Attention*: four specialized attention types replacing standard full attention in each Transformer block. Column attention aggregates within a column (modeling value distributions); feature attention mixes cells within a row and its F→P parent rows (equivalent to a joined-table row attention); neighbor attention propagates signals from P→F child rows (analogous to GNN message passing); and global full attention captures arbitrary pairwise interactions. Crucially, RT omits positional encodings entirely to preserve permutation invariance over tables, rows, and columns — a deliberate contrast with RelGT, which uses extensive positional encodings tailored to a fixed schema. These are implemented with FlexAttention/FlashAttention sparse kernels.

Pretrained on 6/7 RelBench databases (leave-one-DB-out), a 22M-parameter RT achieves on average 93% of supervised AUROC on binary classification tasks in zero-shot. For comparison, Gemma3-27B (27B params, 10⁵× more inference FLOPs) achieves only 84% of supervised AUROC. With continued pretraining on the target database (not the task), RT reaches 93.1% and fine-tunes 10–100× more sample-efficiently than baselines.

## Key Takeaways

- **Cell-level tokenization** enables schema-agnostic pretraining: unlike row-level GNN/GT tokenization, cell tokens are universal across any relational schema.
- **Relational Attention** (column + feature + neighbor + global) encodes relational structure directly in the attention masks, implemented efficiently with FlexAttention/FlashAttention.
- **No positional encodings** by design — preserves permutation invariance across tables/rows/columns, which is critical for zero-shot generalization.
- **Task table prompting** casts both forecasting and autocomplete tasks as masked token prediction, unifying the pretraining and fine-tuning objectives.
- **Outperforms 27B LLMs at 22M params** in zero-shot on relational binary classification (93% vs. 84% of supervised AUROC).
- **Interesting tension with RelGT**: RT's zero-shot strength comes from *avoiding* structural encodings; RelGT's supervised strength comes from *rich* positional encodings. The two papers represent complementary ends of the schema-specificity spectrum.

## Entities & Concepts

- [relational-foundation-model](relational-foundation-model.md)
- [relational-attention](relational-attention.md)
- [relational-deep-learning](relational-deep-learning.md)
- [relational-entity-graph](relational-entity-graph.md)
- [relbench](relbench.md)
- [autocomplete-tasks](autocomplete-tasks.md)
- [graph-transformer](graph-transformer.md)

## Relation to Other Wiki Pages

- [relational-graph-transformer](relational-graph-transformer.md): RT vs. RelGT is a key contrast — same problem (relational data), opposite philosophies (cell-level + no PE for zero-shot vs. row-level + rich PE for supervised).
- [graph-transformer](graph-transformer.md): RT is a cell-level alternative to row-level GTs; its neighbor attention is analogous to GNN message passing.
- [relbench](relbench.md): RT is pretrained and evaluated on RelBench; references RelBench v2 autocomplete tasks.
- [autocomplete-tasks](autocomplete-tasks.md): RT frames both autocomplete and forecasting uniformly as MTP.
