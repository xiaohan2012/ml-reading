---
title: "GCN: Semi-Supervised Classification with Graph Convolutional Networks"
tags: [source, gnn, graph-convolution, spectral-gnn]
sources: [gcn]
updated: 2026-04-29
---

# GCN: Semi-Supervised Classification with Graph Convolutional Networks

**Source:** https://arxiv.org/abs/1609.02907
**Title:** Semi-Supervised Classification with Graph Convolutional Networks
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Kipf, Welling
**Venue:** ICLR 2017
**Year:** 2017

## Summary

Kipf and Welling (University of Amsterdam) derive a scalable graph neural network architecture from a first-order approximation of spectral graph convolutions. The motivation: standard graph Laplacian regularization enforces label smoothness but limits modeling capacity by assuming edges encode similarity. GCN instead conditions the model on both node features X and adjacency matrix A, letting gradient flow through the graph structure.

**Derivation.** Spectral graph convolutions (Chebyshev polynomials) are simplified to K=1 (linear in L), then approximated with λ_max≈2, and renormalized via the *renormalization trick* to avoid gradient explosion:

$$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-\frac{1}{2}} \tilde{A}\tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$

where $\tilde{A} = A + I_N$ (self-loops), $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$. Each layer is a normalized, symmetrically weighted mean aggregation over direct neighbors + self. Complexity: O(|E|·C·F) per layer — linear in edges.

**Two-layer GCN for semi-supervised classification:**
$$Z = \text{softmax}(\hat{A} \cdot \text{ReLU}(\hat{A} X W^{(0)}) \cdot W^{(1)})$$

where $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ is precomputed once. Trained with cross-entropy on labeled nodes only; gradient flows to all nodes via graph structure.

**Results.** Evaluated on Cora, Citeseer, Pubmed (citation networks, semi-supervised node classification) and NELL (knowledge graph). Outperforms graph embedding methods (DeepWalk, Planetoid) and prior spectral GNNs. Strong performance with only 20 labels per class.

## Key Takeaways

- **Renormalization trick prevents gradient explosion**: adding self-loops before normalization keeps eigenvalues in [0,1] instead of [0,2], enabling stable deep training.
- **Isotropic aggregation**: same weight for all neighbors (weighted only by degree normalization) — no attention, no edge features. This is both the simplicity and the limitation.
- **Transductive semi-supervised**: requires full graph during training; node embeddings for unseen nodes require re-running inference — addressed by GraphSAGE.
- **Spectral justification**: K=1 Chebyshev approximation means each layer aggregates exactly 1-hop neighbors; K layers aggregate K-hop neighborhoods.
- **WL connection**: authors note equivalence to 1-WL refinement in Appendix — each GCN layer is a 1-WL step; GIN later proves mean aggregation is not injective and sum is needed for WL-level power.

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md)

## Relation to Other Wiki Pages

- [graph-neural-network](graph-neural-network.md): GCN is the foundational spectral GNN; the field's starting point for node classification.
- [graphgps](graphgps.md): GraphGPS reports GCN on all benchmarks as the standard GNN baseline.
- [relational-graph-transformer](relational-graph-transformer.md): RelGT's RDL baseline is HeteroGraphSAGE (a heterogeneous extension of GraphSAGE/GCN ideas) — GCN itself is not applicable to heterogeneous graphs.
