---
title: "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training"
tags: [source, tabular, transformer, attention, self-supervised]
sources: [saint]
updated: 2026-04-29
---

# SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training

**Source:** https://arxiv.org/abs/2106.01342
**Title:** SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Somepalli, Goldblum, Schwarzschild, Bruss, Goldstein
**Venue:** arxiv 2021
**Year:** 2021

## Summary

Somepalli, Goldblum, Schwarzschild, Bruss, Goldstein (2021) introduce **SAINT** (Self-Attention and Intersample Attention Transformer), a deep learning architecture for tabular data that combines two complementary attention mechanisms: **self-attention** over features within a row and **intersample attention** (MISA) across rows in the batch.

**Architecture**: L identical stages, each with two sequential transformer blocks:
1. **MSA block** (Multi-head Self-Attention): standard column-wise Transformer within each row; attends over feature tokens
2. **MISA block** (Multi-head Intersample Attention): attends over *rows* in the batch — each row attends to all other rows in the batch simultaneously

MISA is equivalent to a GAT on a fully-connected graph of rows: each data point learns which other data points are most relevant for its prediction. This is analogous to nearest-neighbor reasoning but with learned distance metrics.

**Embedding**: all features (continuous and categorical) are projected into a shared d-dimensional space via separate per-feature linear layers. Unlike TabTransformer (which only embeds categoricals), SAINT embeds everything, enabling cross-feature-type attention.

**Contrastive pretraining**: CutMix + mixup augmentations create two views of each data point; contrastive loss (SimCLR-style) + denoising loss are used for self-supervised pretraining before fine-tuning. First paper to apply contrastive learning to tabular data.

SAINT outperforms XGBoost, LightGBM, CatBoost on average across 12+ benchmarks; consistent across data sizes and feature counts.

## Key Takeaways

- Stage = MSA (column attention within row) + MISA (row attention across batch); two sequential Transformer blocks
- MISA equation: `z_i^3 = LN(MISA({z_i^2}_{i=1}^b)) + z_i^2` — all rows in batch attend to each other
- Both continuous and categorical features embedded into shared d-dimensional space (unlike TabTransformer)
- CLS token per row → MLP for prediction
- Contrastive + denoising pretraining (CutMix, mixup) enables semi-supervised performance improvements
- Outperforms boosted trees on average across benchmarks; first consistent result in this direction

## Entities & Concepts

- [tabular-learning](tabular-learning.md) — SAINT's two-attention design is an important bridge between per-row and cross-row learning

## Relation to Other Wiki Pages

- [tabicl](tabicl.md): TabICL's 3-Transformer pipeline (TF_col → TF_row → TF_icl) directly builds on SAINT's insight of combining within-row and between-row attention. TF_col ≈ SAINT's MSA; TF_row ≈ SAINT's MISA. TabICL adds the ICL Transformer and scales to 500K.
- [tabpfnv1](tabpfnv1.md): TabPFN takes a different approach — PFN pretraining eliminates per-dataset training; SAINT requires supervised training. For the same scope, TabPFN is faster; SAINT can be fine-tuned on labeled data.
- [tabpfnv2](tabpfnv2.md): TabPFN v2's alternating row/column attention is architecturally similar to SAINT's MSA+MISA stages.
- [carte](carte.md): CARTE uses graph-per-row encoding with YAGO pretraining; SAINT uses row-and-column attention with contrastive pretraining. Both improve on flat tabular baselines but for different data regimes.
