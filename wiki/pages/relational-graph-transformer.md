---
title: Relational Graph Transformer (RelGT)
tags: [source, graph-transformer, relational-deep-learning, gnn, heterogeneous-graph]
sources: [relational-graph-transformer]
updated: 2026-04-28
---

# Relational Graph Transformer (RelGT)

**Source:** https://arxiv.org/abs/2505.10960
**Title:** Relational Graph Transformer
**Date ingested:** 2026-04-28
**Type:** paper
**Authors:** Unknown
**Venue:** Unknown
**Year:** 2025

## Summary

This paper introduces RelGT, the first Graph Transformer (GT) architecture designed specifically for Relational Deep Learning (RDL). RDL represents multi-table relational databases as heterogeneous temporal graphs called Relational Entity Graphs (REGs), and prior work used standard GNNs (e.g., HeteroGraphSAGE) for prediction. Standard GNNs suffer from limited expressiveness and an inability to model long-range dependencies, while existing GTs (HGT, GraphGPS) were designed for static, homogeneous, or small-scale graphs and fail to handle REGs' combination of heterogeneity, temporality, and schema-defined structure.

RelGT's core contribution is a **5-element tokenization** scheme that decomposes each node into: (1) multimodal node features, (2) node type embedding, (3) hop distance encoding, (4) relative time encoding, and (5) a subgraph GNN-based positional encoding with randomly resampled node initializations. This replaces the standard 2-element (features + global PE) approach used in existing GTs and avoids expensive global precomputation entirely. The transformer then combines local all-pair attention over K=300 sampled subgraph tokens with global attention to B=4096 EMA-updated learnable centroid tokens.

On the RelBench benchmark across 21 entity classification and regression tasks, RelGT consistently matches or outperforms the HeteroGNN RDL baseline (up to 18.43% gain on `rel-trial site-success`), outperforms HGT even with Laplacian PE, and beats the Griffin relational GNN foundation model on 8/10 tasks — establishing Graph Transformers as a strong architecture for relational data.

## Key Takeaways

- The subgraph GNN PE with **random node initialization** (resampled each training step) is the single most critical component; removing it causes the worst ablation drops and it is cheaper and better than Laplacian PE (up to 3.38× faster, consistently better accuracy).
- The 5-element tokenization is a principled decomposition — each element captures a distinct relational property without overlap or expensive precomputation.
- The **global attention module is task-dependent**: it provides large gains for some tasks (`rel-trial site-success`: −19% without it) but can hurt others (`rel-avito user-clicks`: +7.85% without it).
- Traditional PEs (Laplacian, random walk, node2vec) are either prohibitively expensive or fail to generalize to massive heterogeneous temporal REGs.
- HGT, as a GT baseline for heterogeneous graphs, consistently underperforms the HeteroGNN RDL baseline and adds high overhead when augmented with Laplacian PE (up to 8.62× slowdown).

## Entities & Concepts

- [relational-deep-learning](relational-deep-learning.md)
- [relational-entity-graph](relational-entity-graph.md)
- [graph-transformer](graph-transformer.md)
- [multi-element-tokenization](multi-element-tokenization.md)
- [subgraph-gnn-pe](subgraph-gnn-pe.md)
- [relbench](relbench.md)
- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md)

## Relation to Other Wiki Pages

- [relational-transformer](relational-transformer.md): RT vs. RelGT is a key contrast — same RDL problem, opposite design philosophies. RelGT uses rich schema-specific positional encodings for best supervised performance; RT omits all PEs for zero-shot generalization.
- [relbench](relbench.md): RelGT evaluated on v1; [relbench-v2](relbench-v2.md) is the expanded benchmark.
- [relational-foundation-model](relational-foundation-model.md): RelGT is not a foundation model (schema-specific), but is the strongest supervised GT for relational data.
