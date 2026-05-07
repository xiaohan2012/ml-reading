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
**GT layer (sparse attention with edge features).**

Attention scores for edge $(i, j) \in \mathcal{E}$:

$$\hat{w}_{ij} = \frac{Q_i K_j^T}{\sqrt{d_k}} \cdot E \mathbf{e}_{ij}$$

$$w_{ij} = \frac{\exp(\hat{w}_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(\hat{w}_{ik})}$$

Node update:

$$\mathbf{h}'_i = \text{FFN}\!\left(O\!\left(\sum_{j \in \mathcal{N}_i} w_{ij} V_j\right)\right)$$

Edge update (residual):

$$\mathbf{e}'_{ij} = \text{FFN}(\mathbf{e}_{ij} + w_{ij} \cdot \mathbf{1})$$

**LapPE.** Compute the $k$ smallest non-trivial eigenvectors of the normalized graph Laplacian $L_\text{sym} = I - D^{-1/2}AD^{-1/2}$:

$$L_\text{sym} \phi_i = \lambda_i \phi_i, \quad 0 \leq \lambda_1 \leq \cdots \leq \lambda_k$$

LapPE of node $v$: $\text{PE}(v) = [\phi_1(v), \ldots, \phi_k(v)] \in \mathbb{R}^k$. Concatenated with node features as input. Sign ambiguity: randomly flip the sign of each eigenvector during training (augmentation) — SignNet later resolves this rigorously.

**Four modifications from vanilla Transformer:**
1. Sparse neighborhood attention (not full all-pairs)
2. LapPE input encoding
3. BatchNorm instead of LayerNorm (faster convergence on graph tasks)
4. Edge features injected into attention score computation

## Experiments

- ZINC (graph regression): GT with LapPE + BatchNorm outperforms GCN and GAT; edge-feature variant closes gap to GatedGCN.
- PATTERN/CLUSTER (node classification): LapPE provides the largest single boost — without PE, GT performs at random on PATTERN (which requires structural discrimination).
- Sparse attention vs. full attention: full-graph attention underperforms sparse (neighborhood) attention on all three benchmarks — validates sparsity as the correct inductive bias for graphs.
- BatchNorm > LayerNorm: consistent across all tasks; 2–4% improvement on structural tasks.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [positional-encoding](positional-encoding.md)
