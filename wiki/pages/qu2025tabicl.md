---
title: "TabICL: A Tabular Foundation Model for In-Context Learning on Large Data"
tags: [source, tabular, in-context-learning, foundation-model, transformer]
sources: [qu2025tabicl]
updated: 2026-05-06
---

# TabICL: A Tabular Foundation Model for In-Context Learning on Large Data

**Source:** https://arxiv.org/abs/2502.05564
**Title:** TabICL: A Tabular Foundation Model for In-Context Learning on Large Data
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Jingang Qu, Jiashuo Luo, Guo-Wei Yan, Taoran Sun, Chuanhui Hao, Deyuan Xue, Longbing Liu, Junfeng Chen, Yuantao Yin
**Venue:** ICML 2025

## Summary

- **What:** TabPFN v2's alternating row/column attention scales as O(N²M + NM²) — infeasible beyond N≈10K — leaving the large-table ICL regime unsolved.
- **How:** TabICL decouples feature embedding from ICL via a 3-stage Transformer stack: TF_col (Set Transformer over column values → per-cell embedding), TF_row (cross-feature attention per row → 512-dim row embedding), TF_icl (ICL Transformer over row embeddings only) — collapsing the column dimension before the quadratic ICL step.
- **So what:** Scales to N≤500K; matches TabPFN v2 overall and outperforms it on large datasets (N>10K); the 3-stage col→row→ICL pattern is the direct structural ancestor of KumoRFM-2's intra-table Stage 1 + cross-sample Stage 2.

## Challenges & Novelty

TabPFN v2's joint (row, column) attention avoids committing to a column-dimension collapse, preserving full cross-cell interactions — but this costs O(N²M) at the ICL stage, making N>10K intractable. Breaking this requires a design that embeds columns into fixed-size row vectors before the ICL step, without losing the inter-feature dependencies that make tabular ICL expressive.

- **O(N²M) complexity wall:** full attention over all (sample, feature) pairs is quadratic in both N and M; large real-world tables have both large N and large M.
- **Column-agnostic embedding collapses rows:** if all features are embedded identically (no column identity), rows with the same feature distribution become indistinguishable — representation collapse.
- **Curriculum generalization gap:** directly training on large synthetic tables fails to converge — small-to-large curriculum is essential for the model to learn multi-scale tabular patterns.

## Relation to Prior Work

| Method | N limit | Col embedding | Row embedding | ICL stage | Complexity |
|---|---|---|---|---|---|
| [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md) | 1K | Joint | Joint | Joint | O(N²M) |
| [hollmann2025tabpfnv2](hollmann2025tabpfnv2.md) | 10K | Alternating | Alternating | Alternating | O(N²M) |
| **TabICL** | 500K | TF_col | TF_row | TF_icl | O(N²) |

- [hollmann2025tabpfnv2](hollmann2025tabpfnv2.md): TabICL targets the remaining N>10K gap; same PFN/ICL paradigm, more scalable by separating column and ICL stages.
- [fey2025kumorfm2](fey2025kumorfm2.md): KumoRFM-2's hierarchical design mirrors TabICL's 3-stage logic — Stage 1 (col+row attention within a table) then Stage 2 (FK+cross-sample attention across tables), extending the pattern from flat tables to relational databases.

## Technical Details

**TF_col — column embedding (Set Transformer).** Each column $c_j \in \mathbb{R}^N$ is treated as an unordered set of scalar values. An ISAB (Induced Set Attention Block) with $k=128$ inducing vectors computes in $O(Nk)$. Output: per-cell weight $W_{ij}$ and bias $B_{ij}$, giving distribution-aware embedding $e_{ij} = W_{ij} \odot c_{ij} + B_{ij}$.

**TF_row — row interaction.** A 3-layer Transformer (8 heads) processes all feature embeddings per row, capturing cross-feature interactions. Four learnable [CLS] tokens are prepended; their outputs are concatenated into the row embedding $H \in \mathbb{R}^{N \times 512}$, collapsing the column dimension. RoPE (Rotary Positional Embedding) encodes column position to prevent representation collapse when features share the same distribution.

**TF_icl — ICL Transformer.** A 12-layer Transformer on row embeddings $H$. Training labels added via one-hot encoding. Test embeddings attend only to training embeddings (causal masking). Final MLP outputs class probabilities. Complexity: $O(N^2)$ — column dimension eliminated.

**Complexity.** TabPFN v2: $O(N^2M + NM^2)$. TabICL: $O(NMk + NM + N^2) \approx O(N^2)$ for large $N$.

**Pretraining.** Exclusively synthetic: SCMs (TabPFN v1 style) + tree-based SCMs (XGBoost-based generation). Curriculum: 1K → 40K → 60K samples over 3 stages, 20 days on 3×A100.

## Experiments

- On TALENT (200 classification datasets): TabICL matches TabPFN v2 overall and outperforms both TabPFN v2 and CatBoost on the 53 large datasets (N>10K).
- Up to 10× faster than TabPFN v2; column permutation ensembling partially restores permutation invariance lost from RoPE.
- Curriculum learning is critical — training directly on 60K-sample tables fails to converge.

## Entities & Concepts

- [tabular-learning](tabular-learning.md)
