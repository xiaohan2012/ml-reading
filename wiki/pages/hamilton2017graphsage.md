---
title: "GraphSAGE: Inductive Representation Learning on Large Graphs"
tags: [source, gnn, inductive-learning, neighborhood-sampling]
sources: [hamilton2017graphsage]
updated: 2026-04-29
---

# GraphSAGE: Inductive Representation Learning on Large Graphs

**Source:** https://arxiv.org/abs/1706.02216
**Title:** Inductive Representation Learning on Large Graphs
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Hamilton, Ying, Leskovec
**Venue:** NeurIPS 2017
**Year:** 2017

## Summary

Hamilton, Ying, and Leskovec (Stanford) introduce GraphSAGE (SAmple and aggreGatE), the first *inductive* framework for node representation learning. The key problem with prior methods (DeepWalk, GCN): they are transductive — they learn embeddings for fixed nodes in a fixed graph. GraphSAGE instead learns an *aggregation function* that generates embeddings from a node's sampled neighborhood, enabling generalization to unseen nodes and new graphs.

**Core idea.** Instead of learning a single embedding per node, GraphSAGE learns a sequence of aggregator functions that sample and aggregate neighborhood features:

$$h_{\mathcal{N}(v)}^k = \text{AGGREGATE}_k(\{h_u^{k-1} : u \in \mathcal{N}(v)\})$$
$$h_v^k = \sigma(W^k \cdot [h_v^{k-1} \| h_{\mathcal{N}(v)}^k])$$

where neighborhood $\mathcal{N}(v)$ is *sampled* to a fixed size at each layer — making mini-batch training possible on graphs with millions of nodes.

**Aggregator variants:**
- *Mean*: element-wise mean of neighbor features + self (≈ GCN with no renormalization)
- *LSTM*: LSTM applied to a random permutation of neighbors — not permutation-invariant but empirically works
- *Max pooling*: element-wise max over MLP-transformed neighbor features — most expressive; captures representative features

**Inductive setting.** After training, embedding a new node requires only its features and neighborhood — no re-training. Demonstrated on: (1) citation networks (PPI protein interaction), (2) Reddit (2.2M posts), (3) completely unseen protein-protein interaction graphs.

**HeteroGraphSAGE.** The standard RDL baseline (used in RelBench) is a heterogeneous extension of GraphSAGE that handles multiple node/edge types — directly derived from this paper.

## Key Takeaways

- **Inductive vs. transductive**: GraphSAGE learns *functions* not embeddings — the single most important distinction from prior work; enables deployment on evolving graphs and new nodes.
- **Neighborhood sampling enables scalability**: fixed-size neighborhood samples at each layer give bounded memory and compute — critical for Reddit (2.2M nodes) and production deployment.
- **Concatenation over summation**: `[h_v || h_N(v)]` concatenation preserves the self vs. neighbor distinction; pure aggregation loses this.
- **Max pooling is the best aggregator**: empirically outperforms mean and LSTM aggregators; captures "representative" neighbors rather than averaging all.
- **Foundation for RDL baseline**: HeteroGraphSAGE (used in RelBench) directly extends this with type-specific aggregator weights — the canonical heterogeneous GNN baseline that all RDL methods are measured against.

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md)

## Relation to Other Wiki Pages

- [graph-neural-network](graph-neural-network.md): GraphSAGE introduces the inductive learning paradigm that enables GNNs to scale to large and evolving graphs.
- [relational-deep-learning](relational-deep-learning.md): HeteroGraphSAGE (the RDL baseline) is a direct heterogeneous extension — the benchmark all RDL papers beat.
- [kipf2017gcn](kipf2017gcn.md): GraphSAGE addresses GCN's transductive limitation; mean aggregator ≈ GCN without renormalization.
