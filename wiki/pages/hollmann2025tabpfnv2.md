---
title: "TabPFN v2: Accurate Predictions on Small Data with a Tabular Foundation Model"
tags: [source, tabular, in-context-learning, foundation-model, prior-fitted-network]
sources: [hollmann2025tabpfnv2]
updated: 2026-05-06
---

# TabPFN v2: Accurate Predictions on Small Data with a Tabular Foundation Model

**Source:** https://www.nature.com/articles/s41586-024-08328-6
**Title:** Accurate Predictions on Small Data with a Tabular Foundation Model
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Garg, Marc Hutter
**Venue:** Nature 2025

## Summary

- **What:** TabPFN v1 was limited to N≤1000, 100 features, and numerical-only inputs — too small for most real-world tabular tasks; it also required fixed column schemas, making zero-shot transfer across tables infeasible.
- **How:** TabPFN v2 introduces alternating row/column attention within a single Transformer, randomized attribute tokens that remove fixed feature embeddings, and pretraining on ~130M synthetic datasets — extending the ICL paradigm to N≤10K, mixed data types, and arbitrary column schemas.
- **So what:** SOTA on 261 classification/regression benchmarks; beats 4-hour AutoGluon in 2.8 seconds; the intra-table row/column attention design is the direct architectural ancestor of KumoRFM-2's Stage 1.

## Challenges & Novelty

TabPFN v1's scope was too narrow for production: N≤1000 excluded most real datasets, and its fixed feature embedding scheme required retraining for new column schemas. Scaling ICL to larger tables requires architectural changes — naive extension of v1 hits O(N²M) complexity from full attention over all (sample, feature) cells.

- **Scale cap at N=1000:** v1's architecture attended over all (sample, feature) pairs jointly — quadratic in both N and M; increasing the cap requires decoupling sample-level and feature-level attention.
- **Fixed feature embeddings block zero-shot transfer:** v1 learned position-specific column embeddings during pretraining; a new column schema at inference time had no pre-trained embedding.
- **Missing values and mixed types:** v1 required imputation and numeric-only inputs; real tables have categoricals, text, and missing cells.

## Relation to Prior Work

| Method | N limit | Schema-agnostic | Mixed types | Architecture |
|---|---|---|---|---|
| [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md) | 1000 | No | No | Full joint attention |
| **TabPFN v2** | 10K | Yes (random attr. tokens) | Yes | Alternating row/col attn |
| [qu2025tabicl](qu2025tabicl.md) | 500K | Yes | Yes | 3-stage (col→row→ICL) |

- [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md): v1 is the conceptual foundation — PFN objective, SCM prior, single forward pass ICL; v2 fixes scale, schema, and type limitations.
- [qu2025tabicl](qu2025tabicl.md): TabICL targets the remaining 10K→500K gap by decoupling column embedding from the ICL Transformer, trading v2's joint attention for lower complexity.

## Technical Details

> **Two pivots, one idea.** v1 baked fixed column identity into the network in two ways:
>
> 1. it flattened all $N \times M$ cells into one sequence and ran **full joint attention** over them — costing $O(N^2 M^2)$; and
> 2. it learned **fixed column embeddings** at pretraining time. v2 removes both: attention is **factorized** along the row and column axes, and column tokens are **randomized** at every inference call. Everything else (130M datasets, regression, mixed types, missing values) is the scale and engineering enabled by these two architectural changes.

**Alternating row/column attention.** A Transformer encoder alternates attention across *rows* (samples) and *columns* (features):
- *Row attention*: each sample attends to all other samples — captures which training examples are similar to the test instance.
- *Column attention*: each feature position attends to all other feature positions within a sample — captures feature correlations.

This factorization replaces v1's joint cell-attention: complexity drops from $O(N^2 M^2)$ to $O(N^2 M + N M^2)$, unlocking $N \le 10\text{K}$.

**Randomized attribute tokens.** v1 learned **fixed column embeddings** during pretraining, so a new table with a different schema had no matching embedding — v1 was not truly schema-agnostic. v2 instead **resamples attribute tokens randomly at every inference call**: the model must infer "what does column $j$ mean?" purely from the in-context training rows. One checkpoint, any schema, zero-shot — the property that makes v2 a *foundation model* rather than a per-schema PFN.

The two changes compose naturally: factorized column attention treats features as an *unordered set of tokens*, which only makes sense once those tokens carry no fixed identity.

**Pretraining.** Trained on ~130M synthetic tabular datasets (SCMs + BNNs + diverse activation functions), with varied sizes, feature counts, class imbalances, and generating mechanisms.

## Experiments

- Wins on 26% of 261 classification/regression benchmarks (vs. 12% for ModernNCA, 10% for TabM); outperforms 4-hour AutoGluon in 2.8 seconds.
- Robust to uninformative features and outliers; handles missing values via context-based inference.
- Degrades on class-imbalanced datasets and high dimensionality — performance on minority classes is TabICL's primary motivation for further scaling.

## Entities & Concepts

- [tabular-learning](tabular-learning.md)
- [relational-foundation-model](relational-foundation-model.md)
- [tabular-icl-lineage](tabular-icl-lineage.md) — comparison across the PFN → TabPFN → TabICL lineage
