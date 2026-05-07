---
title: "GraphiT: Encoding Graph Structure in Transformers"
tags: [source, graph-transformer, positional-encoding]
sources: [mialon2021graphit]
updated: 2026-05-07
---

# GraphiT: Encoding Graph Structure in Transformers

**Source:** https://arxiv.org/abs/2106.05667
**Title:** GraphiT: Encoding Graph Structure in Transformers
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Grégoire Mialon, Dexiong Chen, Margot Selosse, Julien Mairal
**Venue:** arXiv 2021 (Inria)

## Summary

- **What:** LapPE has sign ambiguity and poor cross-graph transferability; it adds absolute position to node features rather than encoding the *relative* structural relationship between node pairs.
- **How:** Modulate Transformer attention scores by the Gram matrix of a positive-definite graph kernel (diffusion, random-walk, adjacency), encoding relative structure as an attention bias rather than an absolute feature.
- **So what:** Competitive with GNNs on ZINC, MNIST, CIFAR10, MUTAG, PROTEINS; enables interpretable attention patterns that identify discriminative substructures.

## Challenges & Novelty

LapPE encodes the global position of each node independently — it doesn't directly encode the relationship between two nodes that are being attended to. Two nodes close together get similar LapPE vectors, but this is indirect; for attention, a relative encoding of *how similar* node $i$ and $j$ are structurally is more natural. Additionally, LapPE's sign ambiguity (eigenvectors can be negated) and $O(N^3)$ eigendecomposition limit applicability.

- **Kernel-based relative PE avoids sign ambiguity:** the kernel $\mathcal{K}(i, j)$ between nodes $i$ and $j$ depends on their structural relationship — there is no sign choice, no ambiguity. Different kernels encode different structural priors (local vs. global).
- **Kernel as attention score modulation:** multiplying (not adding) attention scores by $K_r$ before softmax is a structurally different integration than appending PE to node features — the structural signal controls *which pairs* attend to each other, not just *who* each node is.
- **GCKN enriches node features with local topology:** Graph Convolutional Kernel Network pre-processing enumerates short paths and embeds them with a kernel function, providing richer initial node features before the Transformer.

## Relation to Prior Work

| PE Type | Encoding | Relative/Absolute | Sign ambiguity | Cross-graph |
|---|---|---|---|---|
| LapPE ([dwivedi2020benchmarking](dwivedi2020benchmarking.md)) | Eigenvectors | Absolute | Yes | No |
| RWSE ([rampavsek2022graphgps](rampavsek2022graphgps.md)) | RW diagonals | Absolute | No | Yes |
| **GraphiT kernel PE** | Gram matrix | Relative | No | Yes |
| SPD bias ([ying2021graphormer](ying2021graphormer.md)) | Shortest path dist. | Relative | No | Yes |
| SignNet ([lim2022signnet](lim2022signnet.md)) | Invariant eigenvectors | Absolute (invariant) | Resolved | Yes |

- [dwivedi2020benchmarking](dwivedi2020benchmarking.md): GraphiT is directly motivated by LapPE's limitations (sign ambiguity, absolute encoding) — relative kernel PE is the complementary design choice.
- [ying2021graphormer](ying2021graphormer.md): both use relative structural encoding as attention biases; Graphormer uses discrete SPD, GraphiT uses continuous kernel — different inductive biases.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GraphGPS later encodes RWSE as a node feature (absolute), not a relative kernel — a practical simplification that loses the relative aspect but is cheaper.
- [kreuzer2021san](kreuzer2021san.md): SAN (same year, NeurIPS 2021) addresses LapPE via full Laplacian spectrum; GraphiT takes the complementary route via kernel modulation of attention.

## Technical Details

**Kernel-modulated attention.** Standard self-attention: $\text{Attn}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d}) V$. GraphiT replaces the softmax argument with an element-wise product with a graph kernel Gram matrix $K_r \in \mathbb{R}^{N \times N}$:

$$\text{PosAttn}(Q, K_r, V) = \text{normalize}\!\left(\exp\!\left(\frac{QQ^T}{\sqrt{d}}\right) \odot K_r\right) V$$

where $[K_r]_{ij} = \mathcal{K}(i, j)$ is a positive-definite kernel on the graph. Three kernel options:
- **Diffusion kernel:** $K_r = e^{-\beta L}$ — Gaussian with graph Laplacian as the metric. $\beta$ controls the attention span (larger $\beta$ = more local).
- **$p$-step random walk kernel:** $K_r = (I - \alpha A)^{-p}$ — captures $p$-hop structural similarity.
- **Adjacency kernel:** $K_r = A$ — restricts attention to 1-hop neighbors (equivalent to sparse GT).

**GCKN substructure encoding.** Pre-process nodes by enumerating short paths of length $l$ emanating from each node; embed each path with a kernel function (e.g., Gaussian on path feature sequences). Output: node feature matrix $X \in \mathbb{R}^{N \times d}$ with richer topological content than raw node attributes.

**Interpretability.** The attention score maps $A_{ij}^{(l)}$ after kernel modulation localize which node pairs drive the prediction — discriminative substructures appear as high-attention subgraphs.

## Experiments

- ZINC: GraphiT with diffusion kernel + GCKN encoding matches or outperforms GCN, GIN, and GAT; competitive with GatedGCN.
- MNIST/CIFAR10 (superpixel graphs): diffusion kernel PE provides consistent improvements over no-PE Transformer baseline.
- MUTAG/PROTEINS (molecular classification): GraphiT + GCKN outperforms GCN and GIN; attention maps identify discriminative ring structures.
- Kernel comparison: diffusion > RW > adjacency for global graph tasks; adjacency best for local tasks — kernel choice encodes the right inductive bias.

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-transformer](graph-transformer.md)
