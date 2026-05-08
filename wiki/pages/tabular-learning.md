---
title: Tabular Learning
tags: [concept, tabular, foundation-model, in-context-learning]
sources: [hollmann2025tabpfnv2, qu2025tabicl, kim2024carte, hollmann2023tabpfnv1, somepalli2021saint, muller2022pfn]
updated: 2026-05-08
---

# Tabular Learning

## Description

Tabular learning refers to machine learning on structured data organized in rows and columns, where columns may have heterogeneous types (numerical, categorical, string). This is the most common data format in industry (databases, spreadsheets, CSV files), yet it has historically resisted the foundation model revolution that transformed images and text.

**The core challenge**: unlike images or text, tabular datasets vary wildly in schema — different tables have different columns, types, and semantics. A model pretrained on one table cannot directly apply to another with different columns, unless it can generalize across schemas.

**The GBDT dominance problem**: gradient-boosted decision trees (CatBoost, XGBoost) have dominated tabular benchmarks for years. Deep learning approaches underperform unless they bring something trees cannot — background knowledge, cross-table transfer, or zero-shot ICL.

**Foundation model approaches:**

| Approach | Method | Strengths | Limits |
|----------|--------|-----------|--------|
| Approach | Method | Strengths | Limits |
|----------|--------|-----------|--------|
| TabPFN v1 | ICL via SCM+BNN prior | Zero tuning; <1 sec; foundational PFN | N≤1000; ≤100 features; no missing |
| TabPFN v2 | ICL via synthetic prior; alternating row/col attention | Zero tuning; fast; SOTA small tables | N<10K; <10 classes |
| TabICL | 3-Transformer (col→row→ICL); Set Transformer embedding | Scalable to 500K; handles large tables | Classification only |
| CARTE | Graph per row + YAGO pretraining; edge = column name | Strings; no schema matching; transfer | Slow; needs fine-tuning |
| SAINT | Column attention + intersample (row) attention; contrastive pretrain | Beats boosted trees; semi-supervised | Supervised; no ICL |

**In-Context Learning (ICL) paradigm.** TabPFNv2 and TabICL both follow the Prior-Data Fitted Network (PFN) approach: pretrain a transformer on many synthetic datasets to approximate $p(y | x, \mathcal{D}_\text{train})$. At inference, training data is passed as context — single forward pass, no parameter updates. ICL eliminates the AutoML search process entirely.

## Appearances in Sources

- [muller2022pfn](muller2022pfn.md) — original PFN paper; proves Prior-Data NLL ↔ KL-divergence minimization; GP/BNN approximation + tabular; ICLR 2022
- [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md) — establishes the PFN paradigm for tabular classification; SCM+BNN prior; N≤1000 in-context classification; ICLR 2023
- [hollmann2025tabpfnv2](hollmann2025tabpfnv2.md) — extends v1 with 130M synthetic datasets, row/col attention, N≤10K; SOTA small tables; Nature 2025
- [qu2025tabicl](qu2025tabicl.md) — extends ICL to 500K samples via distribution-aware column embedding + patching analogy; outperforms TabPFNv2 on large tables; ICML 2025
- [kim2024carte](kim2024carte.md) — graph-based representation for cross-table pretraining without schema matching; YAGO knowledge pretraining; handles string-heavy tables; ICML 2024
- [somepalli2021saint](somepalli2021saint.md) — two-attention (column + intersample/row) Transformer + contrastive pretraining; outperforms boosted trees; 2021

## Related Concepts

- [relational-deep-learning](relational-deep-learning.md) — extends tabular learning to multi-table relational schemas via GNNs
- [relational-foundation-model](relational-foundation-model.md) — applies the foundation model paradigm to relational DBs (multi-table), not flat tables
- [graph-neural-network](graph-neural-network.md) — CARTE uses GAT-style edge-augmented attention; RDL baseline (HeteroGraphSAGE) adapts GNNs to relational tables
