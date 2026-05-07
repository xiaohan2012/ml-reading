---
title: "A Generalization of Transformer Networks to Graphs"
tags: [source, graph-transformer, positional-encoding, gnn]
sources: [dwivedi2021gt]
updated: 2026-05-07
---

# A Generalization of Transformer Networks to Graphs

**Source:** https://arxiv.org/abs/2012.09699
**Title:** A Generalization of Transformer Networks to Graphs
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Dwivedi, Bresson
**Venue:** AAAI 2021 Workshop (Deep Learning on Graphs)

## Summary

- **What:** Vanilla Transformers applied to graphs fail because (a) full all-pairs attention ignores the graph's structural inductive bias and (b) there is no canonical node ordering — nodes lack positional identity.
- **How:** Adapt Transformer to graphs with four modifications: sparse neighborhood attention, Laplacian eigenvector PE (LapPE), BatchNorm, and edge-feature injection into attention scores.
- **So what:** Establishes LapPE as the canonical graph positional encoding and sparse attention as the correct inductive bias; becomes the "GT baseline" that GraphGPS, SAN, and LSPE all compare against.

## Challenges & Novelty

NLP Transformers use sinusoidal PE for sequence position — but graphs have no canonical ordering and potentially millions of nodes. Prior GNNs (GCN, GAT) use edges as structural inductive bias but cannot reason about global position or propagate long-range information. Simply applying full Transformer attention to all node pairs loses edge semantics and doesn't scale.

- **Sparse neighborhood attention preserves structural inductive bias:** restricting attention to $\mathcal{N}(i)$ rather than all pairs encodes the graph's known connectivity as an inductive bias — sparse ablation directly outperforms full-graph attention on all three benchmark datasets.
- **LapPE as the NLP PE analogue for graphs:** Laplacian eigenvectors $\phi_1, \ldots, \phi_k$ of the normalized graph Laplacian encode global position (nearby nodes have similar eigenvector values) and generalize sinusoidal PE to arbitrary graph topology. Sign ambiguity handled by random sign-flipping during training.
- **Edge features multiply attention scores:** injecting edge attributes $\mathbf{e}_{ij}$ directly into the raw attention score ($\hat{w}_{ij} = (Q_i \cdot K_j / \sqrt{d}) \cdot E \cdot \mathbf{e}_{ij}$) is more effective than using them only in the value stream.

## Relation to Prior Work

| Model | Attention scope | Positional encoding | Edge features | Norm |
|---|---|---|---|---|
| Standard Transformer | Full (all-pairs) | Sinusoidal sequence PE | No | LayerNorm |
| GAT ([velickovic2018gat](velickovic2018gat.md)) | Sparse (1-hop) | None | No | BatchNorm |
| **GT (this paper)** | Sparse (1-hop) | LapPE | Yes (in attention) | BatchNorm |
| [kreuzer2021san](kreuzer2021san.md) | Full (all-pairs) | Full Laplacian spectrum | Yes | LayerNorm |
| [rampavsek2022graphgps](rampavsek2022graphgps.md) | Sparse + global (parallel) | RWSE/SignNet | Yes | — |

- [dwivedi2020benchmarking](dwivedi2020benchmarking.md): this paper formalizes the LapPE introduced in Benchmarking-GNNs into the GT architecture; same lead author and benchmark datasets.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GraphGPS builds directly on this GT baseline; categorizes LapPE as "global PE" and shows RWSE + SignNet outperforms it on molecular tasks.
- [kreuzer2021san](kreuzer2021san.md): SAN (same year) also uses full Laplacian spectrum as PE but in a full-graph attention context; GT's sparse attention is the complementary design choice.
- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md): HGT references this GT as its homogeneous counterpart; extends meta-relation triplet idea to handle multiple node/edge types.

![[Pasted image 20260507084511.png]]

## Technical Details

**Input embedding.** Raw node features $\alpha_i \in \mathbb{R}^{d_n}$ and edge features $\beta_{ij} \in \mathbb{R}^{d_e}$ are linearly projected to $d$-dimensional hidden vectors:

$$\hat{h}_i^0 = A^0 \alpha_i + a^0, \qquad e_{ij}^0 = B^0 \beta_{ij} + b^0$$

Pre-computed LapPE vectors $\lambda_i \in \mathbb{R}^k$ (the $k$ smallest non-trivial eigenvectors of the normalized graph Laplacian $L = I - D^{-1/2}AD^{-1/2}$) are linearly projected and **added to node features at the input layer only**:

$$\lambda_i^0 = C^0 \lambda_i + c^0, \qquad h_i^0 = \hat{h}_i^0 + \lambda_i^0$$

Sign ambiguity of eigenvectors is handled by random sign-flipping during training.

**Graph Transformer Layer (node-only).** Multi-head sparse attention restricted to 1-hop neighborhood $\mathcal{N}_i$, followed by residual connections and Norm (BatchNorm preferred over LayerNorm) + FFN:

$$\hat{h}_i^{\ell+1} = O_h^\ell \Big\|_{k=1}^{H} \!\left(\sum_{j \in \mathcal{N}_i} w_{ij}^{k,\ell}\, V^{k,\ell} h_j^\ell\right), \qquad w_{ij}^{k,\ell} = \operatorname{softmax}_j\!\left(\frac{Q^{k,\ell} h_i^\ell \cdot K^{k,\ell} h_j^\ell}{\sqrt{d_k}}\right)$$

$$\hat{\hat{h}}_i^{\ell+1} = \operatorname{Norm}(h_i^\ell + \hat{h}_i^{\ell+1}), \qquad h_i^{\ell+1} = \operatorname{Norm}\!\left(\hat{\hat{h}}_i^{\ell+1} + W_2^\ell \operatorname{ReLU}(W_1^\ell \hat{\hat{h}}_i^{\ell+1})\right)$$

Softmax logits are clamped to $[-5, +5]$ for numerical stability.

**Graph Transformer Layer with edge features.** For each head $k$, the raw attention score is a **$d_k$-dimensional vector** formed by element-wise product of the projected query and key:

$$\mathbf{a}_{ij}^{k,\ell} = \frac{Q^{k,\ell} h_i^\ell \odot K^{k,\ell} h_j^\ell}{\sqrt{d_k}} \in \mathbb{R}^{d_k}$$

The scalar attention weight is then the **dot product** of this vector with a projected edge embedding $E^{k,\ell} e_{ij}^\ell \in \mathbb{R}^{d_k}$:

$$\hat{w}_{ij}^{k,\ell} = \mathbf{a}_{ij}^{k,\ell} \cdot E^{k,\ell} e_{ij}^\ell, \qquad w_{ij}^{k,\ell} = \operatorname{softmax}_j\!\left(\hat{w}_{ij}^{k,\ell}\right)$$

Edge embeddings are **contextualized**: at $\ell = 0$ they come from raw features ($e_{ij}^0 = B^0\beta_{ij} + b^0$), but at $\ell > 0$ they are derived from the attention scalars of the previous layer — the $H$ per-head scalars $\{\hat{w}_{ij}^{k,\ell-1}\}_{k=1}^H$ are concatenated into an $H$-vector and projected:

$$\hat{e}_{ij}^{\ell} = O_e^{\ell-1} \Big\|_{k=1}^{H} \hat{w}_{ij}^{k,\ell-1}$$

This then passes through its own residual + Norm + FFN block, yielding updated $e_{ij}^{\ell}$ for use in the current layer's attention computation.

## Experiments

- ZINC (graph regression): GT with LapPE + BatchNorm outperforms GCN and GAT; edge-feature variant closes gap to GatedGCN.
- PATTERN/CLUSTER (node classification): LapPE provides the largest single boost — without PE, GT performs at random on PATTERN (which requires structural discrimination).
- Sparse attention vs. full attention: full-graph attention underperforms sparse (neighborhood) attention on all three benchmarks — validates sparsity as the correct inductive bias for graphs.
- BatchNorm > LayerNorm: consistent across all tasks; 2–4% improvement on structural tasks.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [positional-encoding](positional-encoding.md)
