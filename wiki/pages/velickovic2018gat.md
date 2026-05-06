---
title: "Graph Attention Networks (GAT)"
tags: [source, gnn, attention, node-classification]
sources: [velickovic2018gat]
updated: 2026-04-29
---

# Graph Attention Networks (GAT)

**Source:** https://arxiv.org/abs/1710.10903
**Title:** Graph Attention Networks
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Veličković, Cucurull, Casanova, Romero, Liò, Bengio
**Venue:** ICLR 2018
**Year:** 2018

## Summary

Veličković et al. introduce Graph Attention Networks (GAT), which replace GCN's isotropic (degree-normalized) aggregation with learned attention weights over neighbors. The key insight: different neighbors should contribute differently to a node's representation, but the importance should be computed from features alone — no global graph structure required upfront.

**Attention mechanism.** For each edge (i, j), a scalar attention coefficient is computed by a shared single-layer feedforward network applied to the concatenation of the two nodes' (linearly transformed) features, normalized via softmax:

$$\alpha_{ij} = \frac{\exp\!\left(\text{LeakyReLU}\!\left(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]\right)\right)}{\sum_{k\in\mathcal{N}_i} \exp\!\left(\text{LeakyReLU}\!\left(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_k]\right)\right)}$$

The output for node $i$ is then:

$$\mathbf{h}'_i = \sigma\!\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}\mathbf{h}_j\right)$$

**Multi-head attention.** $K$ independent attention mechanisms are run in parallel; their outputs are concatenated at intermediate layers and averaged at the final layer (before classification softmax).

**Complexity.** O(|V|FF' + |E|F') per head — same order as GCN. Fully parallelizable across edges and nodes.

**Inductive capability.** Because attention is computed locally from node features (not global graph structure), GAT can be applied to unseen graphs at test time — demonstrated on the PPI dataset (20 training graphs, 2 unseen test graphs).

**Results.** Transductive: Cora 83.0%, Citeseer 72.5%, Pubmed 79.0% (all SOTA or tied). Inductive: PPI micro-F1 = 0.973 (vs. GraphSAGE-LSTM 0.612; vs. Const-GAT 0.934), establishing that the attention mechanism provides real benefit beyond architecture.

## Key Takeaways

- **Attention replaces degree normalization**: GCN weights every neighbor by degree alone; GAT learns neighbor importance from features — this is the single most important distinction.
- **No global structure required**: attention is edge-local, making GAT applicable inductive and on heterogeneous/directed graphs without upfront Laplacian computation.
- **Multi-head stabilizes learning**: single-head attention is noisy; $K=8$ heads for transductive, $K=4$ intermediate + $K=6$ final for inductive.
- **Dropout on attention coefficients**: regularizing the $\alpha_{ij}$ values (not just node features) is critical for small training sets — each node sees a stochastically sampled neighborhood per iteration.
- **GIN perspective**: GAT's attention-weighted mean is still not an injective multiset function — attention improves over GCN but does not achieve WL-level expressive power (per GIN's theoretical analysis).

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md)

## Relation to Other Wiki Pages

- [graph-neural-network](graph-neural-network.md): GAT is one of the four foundational GNN variants; introduces learned per-neighbor attention weights.
- [kipf2017gcn](kipf2017gcn.md): GAT directly replaces GCN's isotropic degree normalization with feature-based attention; same O(|E|) complexity.
- [hamilton2017graphsage](hamilton2017graphsage.md): both handle inductive learning; GAT uses the full neighborhood while GraphSAGE samples a fixed-size subset.
- [xu2019gin](xu2019gin.md): GIN's theory shows attention-weighted mean (GAT) is not injective — GAT improves capacity over GCN but is still < WL in expressive power.
- [hu2020hgt](hu2020hgt.md): HGT extends GAT-style attention to heterogeneous graphs via type-specific key/query matrices per meta-relation triplet.
