---
title: Representation Similarity (Cosine / CKA)
tags: [interpretability, mechanistic-interpretability, representation-analysis]
sources: [balef2026onelayer]
updated: 2026-05-28
---

# Representation Similarity

Pairwise comparison of hidden-state tensors across model layers, used to find **redundant blocks** and stage transitions. Two complementary metrics are conventionally shown together.

## Two metrics

- **Cosine similarity** — point-wise: $\cos(h_i^{(t)}, h_j^{(t)})$ for each token $t$, averaged over tokens and datasets. Captures per-vector angular alignment; **survives non-isotropic scaling**.
- **Linear CKA** (Kornblith et al. 2019) — holistic:
    $$\mathrm{CKA}(X_i, X_j) = \frac{\|X_i^\top X_j\|_F^2}{\|X_i^\top X_i\|_F \cdot \|X_j^\top X_j\|_F + \varepsilon}$$
  on column-centered token matrices $X_i, X_j \in \mathbb{R}^{n \times d}$. Captures **global geometry**; invariant to isotropic scaling and orthogonal transforms.

## Diagnostic: cosine vs CKA

- **cosine ≈ CKA** → representations look similar at every level.
- **cosine ≫ CKA** → non-isotropic dimension stretching: vectors still angularly aligned, geometry distorted (typical signature of attention heads expanding particular subspaces).

## Visualization

A layer × layer heatmap per model, lower triangle = cosine, upper triangle = CKA:

- **Non-redundant** model → sharp diagonal, cold off-diagonal.
- **Redundant** model → thick red blocks along the diagonal (adjacent layers share representation space).

## Appearances in Sources

- [balef2026onelayer](balef2026onelayer.md) — heatmaps reveal middle/late depth-redundancy in TabPFN(2.5) and LimiX-16M; the cosine ≫ CKA discrepancy in early layers signals non-isotropic head expansion.
- [ferrando2024primer](ferrando2024primer.md) — complements representational analyses with the residual-stream perspective on layer-to-layer information flow.

## Related Concepts

- [probing-classifier](probing-classifier.md) — adjacent-layer block redundancy seen here bleeds into the cross-layer probe matrix.
