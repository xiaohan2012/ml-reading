---
title: "GraphSAGE: Inductive Representation Learning on Large Graphs"
tags: [source, gnn, inductive-learning, neighborhood-sampling]
sources: [hamilton2017graphsage]
updated: 2026-05-07
---

# GraphSAGE: Inductive Representation Learning on Large Graphs

**Source:** https://arxiv.org/abs/1706.02216
**Title:** Inductive Representation Learning on Large Graphs
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Hamilton, Ying, Leskovec
**Venue:** NeurIPS 2017

## Summary

- **What:** Prior GNNs (DeepWalk, GCN) are transductive — they learn per-node embeddings for a fixed graph and cannot generalize to unseen nodes or new graphs.
- **How:** Learn aggregation functions over fixed-size sampled neighborhoods; concatenate each node's self-embedding with its aggregated neighborhood representation.
- **So what:** First inductive GNN framework; scales to 2.2M-node Reddit graphs and generalizes to completely unseen protein-interaction graphs; the direct ancestor of HeteroGraphSAGE used in all RDL baselines.

## Challenges & Novelty

DeepWalk and node2vec learn one embedding vector per node — they cannot embed nodes not seen during training, and retraining on a new graph is prohibitively expensive. GCN (Kipf 2017) requires the full graph adjacency matrix during training and inference, making it transductive by design.

- **Aggregation functions as the learned object:** instead of embedding vectors, GraphSAGE learns functions that generate embeddings from a node's neighborhood, enabling embedding any node with features — seen or unseen.
- **Neighborhood sampling enables mini-batch training:** sampling a fixed number of neighbors per layer bounds memory and compute regardless of node degree — critical for scaling to millions of nodes.
- **Concatenation preserves self-identity:** using $[h_v \| h_{\mathcal{N}(v)}]$ rather than pure aggregation ensures the node's own representation is not lost in the neighborhood average — a design choice later validated in many architectures.

## Relation to Prior Work

| Model | Inductive | Sampling | Aggregator | Scalable |
|---|---|---|---|---|
| DeepWalk / node2vec | No | Random walk | — | Yes |
| [kipf2017gcn](kipf2017gcn.md) | No | Full graph | Normalized mean | No (dense $\hat{A}$) |
| **GraphSAGE-mean** | Yes | Fixed-size | Mean | Yes |
| **GraphSAGE-max** | Yes | Fixed-size | Max-pool MLP | Yes |
| [velickovic2018gat](velickovic2018gat.md) | Yes | Full neighborhood | Attention-weighted | No (full $\mathcal{N}$) |

- [kipf2017gcn](kipf2017gcn.md): GraphSAGE directly addresses GCN's transductive limitation; GraphSAGE-mean approximates GCN without the renormalization trick.
- [velickovic2018gat](velickovic2018gat.md): GAT improves over GraphSAGE's fixed aggregation with learned neighbor importance, but requires the full neighborhood (no sampling).
- [xu2019gin](xu2019gin.md): GIN's expressiveness analysis shows GraphSAGE-mean is not injective (same limitation as GCN); max-pool variant is also strictly weaker than WL.
- [chen2025relgnn](chen2025relgnn.md): RelGNN's HeteroGraphSAGE baseline (the RDL benchmark reference) directly extends this with type-specific aggregators for heterogeneous FK/PK graphs.

## Technical Details

**Core update equations.**

$$h_{\mathcal{N}(v)}^k = \text{AGGREGATE}_k\!\left(\{h_u^{k-1} : u \in \mathcal{N}(v)\}\right)$$

$$h_v^k = \sigma\!\left(W^k \cdot \left[h_v^{k-1} \| h_{\mathcal{N}(v)}^k\right]\right)$$

**Aggregator variants:**

- *Mean:* $h_{\mathcal{N}(v)}^k = \text{mean}(\{h_u^{k-1}\})$ — element-wise mean; equivalent to GCN without degree renormalization.
- *LSTM:* apply an LSTM to a random permutation of neighbor embeddings — not permutation-invariant but empirically strong.
- *Max-pool:* $h_{\mathcal{N}(v)}^k = \max(\{\text{ReLU}(W_\text{pool} h_u^{k-1} + b)\})$ — element-wise max over MLP-transformed neighbors; captures the most salient representative feature.

**Neighborhood sampling.** At each layer $k$, sample $S_k$ neighbors uniformly at random (without replacement). For $K=2$ layers with $S_1=25, S_2=10$, each node aggregates at most 250 2-hop neighbors — memory is bounded regardless of actual degree.

**Unsupervised training.** When labels are unavailable, train with a graph-based loss: nearby nodes (co-occurring in random walks) should have similar embeddings; distant nodes should differ. This is a noise-contrastive approach:

$$J_\mathcal{G}(z_u) = -\log\sigma(z_u^\top z_v) - Q \cdot \mathbb{E}_{v_n \sim P_n} \log\sigma(-z_u^\top z_{v_n})$$

**HeteroGraphSAGE.** The standard RDL baseline adds type-specific weight matrices $W_r^k$ per relation type, making it a heterogeneous extension directly analogous to R-GCN's relation-specific formulation.

## Experiments

- Reddit (2.2M nodes, 11M edges): GraphSAGE-LSTM achieves F1=0.953 vs. DeepWalk 0.650 — inductive setting, 30% of nodes unseen during training.
- PPI (20 training graphs, 2 unseen test graphs): micro-F1=0.768 (unsupervised) to 0.612 (supervised LSTM) — demonstrates generalization to entirely new graphs.
- Max-pool aggregator consistently outperforms mean and LSTM on most benchmarks; concatenation outperforms mean-only aggregation by ~2–4% F1.
- 3–4× speedup over full-batch GCN on large graphs due to neighborhood sampling.

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md)
- [relational-deep-learning](relational-deep-learning.md)
