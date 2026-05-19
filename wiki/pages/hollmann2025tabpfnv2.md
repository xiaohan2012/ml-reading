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

![[Pasted image 20260519093631.png]]
## Summary

- **What:** v1 collapsed each row into a single token via a `Linear` projection with **learned per-slot column weights** — capping it at small tables, fixed feature count, numeric-only inputs, classification only, and schema-invariant only on average (via rotation ensembling at inference).
- **How — the conceptual pivot:** v2 changes the *unit of tokenization* from row to **cell**. Per-(row, feature) tokens enable (i) **factorized alternating row/column attention** in place of v1's row-only attention, and (ii) **randomized attribute tokens** resampled every inference call — replacing v1's learned per-column weights with an architectural schema invariance. The same pivot makes regression, categoricals, and missingness first-class per-cell encoder steps rather than external pre-processing.
- **So what — the consequences:**
	- *Schema invariance by construction*, not by ensembling — one checkpoint, any schema, one forward pass; v2 is a *foundation model*, v1 was a per-schema PFN.
	- *Scope expansion*: larger tables, mixed types, NaNs, regression — all unlocked by the cell-token reframing.
	- *Cost*: per-layer attention complexity rises (row-only → factorized row + column); absorbed via KV caching + multiquery test attention, not a complexity drop.
	- *Empirics*: SOTA across hundreds of classification/regression benchmarks; the row/column attention block is the direct architectural ancestor of KumoRFM-2's Stage 1.

## Challenges & Novelty

TabPFN v1's scope was too narrow for production: N≤1000 excluded most real datasets, and its fixed feature embedding scheme required retraining for new column schemas. Scaling ICL to larger tables requires architectural changes — naive extension of v1 hits $O(N^2 M^2)$ complexity from full attention over all (sample, feature) cells.

- **Scale cap at N=1000:** v1 attended over rows only ($O(N^2)$) with all features collapsed into one row-token, which limits both expressivity (no per-feature reasoning) and schema flexibility; lifting these constraints means introducing per-feature tokens, which would naively cost $O((NM)^2)$ unless attention is factorized.
- **Fixed feature embeddings block zero-shot transfer:** v1 learned position-specific column embeddings during pretraining; a new column schema at inference time had no pre-trained embedding.
- **Missing values and mixed types:** v1 required imputation and numeric-only inputs; real tables have categoricals, text, and missing cells.

## Relation to Prior Work

| Method | N limit | Schema-agnostic | Mixed types | Architecture | Attention complexity |
|---|---|---|---|---|---|
| [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md) | 1000 | No | No | Full joint attention | $O(N^2)$ (one token per row, $M$ folded into $d$) |
| **TabPFN v2** | 10K | Yes (random attr. tokens) | Yes | Alternating row/col attn | $O(N^2 M + N M^2)$ |
| [qu2025tabicl](qu2025tabicl.md) | 500K | Yes | Yes | 3-stage (col→row→ICL) | $O(N M)$ (induced-point Set Transformer + factored stages) |

- [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md): v1 is the conceptual foundation — PFN objective, SCM prior, single forward pass ICL; v2 fixes scale, schema, and type limitations.
- [qu2025tabicl](qu2025tabicl.md): TabICL targets the remaining 10K→500K gap by decoupling column embedding from the ICL Transformer, trading v2's joint attention for lower complexity.

### v1 vs v2

v1 is schema-agnostic only on average — rotation ensembling masks learned per-slot biases at inference. v2 removes those biases architecturally; the four schema-shape limitations of v1 (see [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md) §Limitations) map directly onto v2 fixes:

| Issue                  | v1                                                               | v2                                                                                                   |
| ---------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Per-slot weights       | $W_x \in \mathbb{R}^{d \times 100}$, column $j$ uses $W_x[:, j]$ | Shared $W_x \in \mathbb{R}^{d \times g}$ ($g \in \{1, 2\}$), applied to every cell                   |
| Column identity        | Learned, position-specific; neutralized by rotation ensembling   | Random vector resampled per inference call (`feature_positional_embedding="subspace"`)               |
| Feature-count contract | Hard cap at $M_{\max} = 100$                                     | Pad to multiple of $g$; release config caps at 85–95 but no $M_{\max}$ in encoder                    |
| Cat / NaN              | External imputation                                              | First-class encoder steps (`NanHandlingEncoderStep`, `CategoricalInputEncoderPerFeatureEncoderStep`) |

Net effect: v2 makes schema invariance an *architectural property* rather than a statistical hope. The cost is the higher per-layer attention complexity, absorbed via KV caching and multiquery test attention. See [tabpfn-v2-code](tabpfn-v2-code.md) for code-level details.

## Technical Details

> **Two pivots, one idea.** v1 baked fixed column identity into the network in two ways:
>
> 1. it **collapsed all $M$ features of a row into a single token** via `Linear(M_max → d)` — one row, one token — and ran $O(N^2)$ row-attention. Features had no token of their own and no way to interact except through the row-level mixing.
> 2. it learned **fixed column embeddings** (the columns of $W_x$) at pretraining time, so a new schema had no matching weights. v2 removes both: tokens are **per (row, feature)** so attention can be **factorized** along the row and column axes, and column tokens are **randomized** at every inference call. Everything else (130M datasets, regression, mixed types, missing values) is the scale and engineering enabled by these two architectural changes.

**Alternating row/column attention.** A Transformer encoder alternates attention across *rows* (samples) and *columns* (features):
- *Row attention*: each sample attends to all other samples — captures which training examples are similar to the test instance.
- *Column attention*: each feature position attends to all other feature positions within a sample — captures feature correlations.

Per-layer attention cost is $O(N^2 M + N M^2)$ (row-attn over $N$ tokens for each of $M$ feature groups, plus feature-attn over $M$ tokens for each of $N$ rows). This is *higher* than v1's $O(N^2)$ row-only attention in absolute terms, but v2 needs it: per-feature tokens are what let column identity be randomized (and thus schema-agnostic) and what let regression / mixed types / NaNs be modeled per-cell. The $N \le 10\text{K}$ scale is unlocked by KV caching and multiquery test attention amortizing the constants — not by a complexity drop.

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
- [tabpfn-v2-code](tabpfn-v2-code.md) — code-grounded walkthrough of the v2 architecture
