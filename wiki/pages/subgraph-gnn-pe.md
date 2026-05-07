---
title: Subgraph GNN PE (with Random Initialization)
tags: [concept, positional-encoding, graph-transformer, gnn]
sources: [dwivedi2025relgt]
updated: 2026-04-28
---

# Subgraph GNN PE (with Random Initialization)

## Description

Subgraph GNN PE is the positional encoding strategy used in [RelGT](dwivedi2025relgt.md) as the fifth element of its [multi-element tokenization](multi-element-tokenization.md). It applies a lightweight GNN to the sampled local subgraph (K nodes) using **randomly resampled node features** as input:

$$h_{\text{pe}}(v_j) = \text{GNN}(A_{\text{local}}, Z_{\text{random}})_j$$

where $A_{\text{local}} \in \mathbb{R}^{K \times K}$ is the subgraph adjacency and $Z_{\text{random}} \in \mathbb{R}^{K \times 1}$ is re-drawn independently at every training step.

**Why random initialization works**: Fixed random initialization would break permutation equivariance. Resampling at every step approximates equivariance in expectation while retaining the expressivity gains from randomness (breaking structural symmetries between topology and node attributes). This is a relaxed version of the learnable PE approach in Kanatsoulis & Leskovec (2025).

**Why it beats Laplacian PE on REGs** (LapPE originally from [dwivedi2020benchmarking](dwivedi2020benchmarking.md)):
- Laplacian PE requires separate computation per node type (heterogeneous graphs don't have a single Laplacian)
- Adds 1.02×–3.38× runtime overhead
- Consistently underperforms random initialization across tested tasks

This PE captures structural properties that other token elements miss: cycles, quasi-cliques, and parent-child relationships within the local subgraph.

## Appearances in Sources

- [dwivedi2025relgt](dwivedi2025relgt.md) — described in Section 3.1 and ablated in Table 2; most critical component in ablations

## Related Concepts

- [multi-element-tokenization](multi-element-tokenization.md) — the overall tokenization scheme this is part of
- [graph-transformer](graph-transformer.md) — architecture context
