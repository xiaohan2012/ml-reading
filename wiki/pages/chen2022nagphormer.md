---
title: "NAGphormer: A Tokenized Graph Transformer for Node Classification in Large Graphs"
tags: [source, graph-transformer, scalability]
sources: [chen2022nagphormer]
updated: 2026-05-04
---

# NAGphormer: A Tokenized Graph Transformer for Node Classification in Large Graphs

**Source:** https://arxiv.org/abs/2206.04910
**Title:** NAGphormer: A Tokenized Graph Transformer for Node Classification in Large Graphs
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Jinsong Chen, Kaiyuan Gao, Gaichao Li, Kun He
**Venue:** ICLR 2023

## Summary

NAGphormer reformulates the scalability problem for Graph Transformers: instead of applying attention over all N nodes (quadratic), each node is represented as a **sequence of hop-aggregated tokens**. The **Hop2Token** module pre-aggregates neighborhood features at each hop independently (similar to SGC / SIGN), producing a fixed-length sequence $[h_v^{(0)}, h_v^{(1)}, ..., h_v^{(K)}]$ where $h_v^{(k)}$ is the mean of $k$-hop neighborhood features. A Transformer then processes this $K$-token sequence for each node independently, enabling mini-batch training.

This is fundamentally different from ego-graph approaches (Gophormer): NAGphormer doesn't sample a neighborhood subgraph; it pre-computes multi-hop aggregate features and treats each node in isolation. The Transformer captures interactions *between different hops* for a single node, not interactions between neighboring nodes. This gives multi-scale receptive field information without any graph sampling or $O(N^2)$ attention.

## Key Takeaways

- **Hop2Token**: pre-aggregate features at each hop → per-node sequence of $K$ tokens; enables mini-batch training over nodes independently.
- Transformer processes interaction *between hops* (within a node), not between nodes — fundamentally different from standard GT designs.
- Multi-hop aggregation is pre-computed (no graph access at training time) — fast and memory-efficient.
- Outperforms decoupled GNNs (SIGN, SGC) by capturing cross-hop interactions via attention.
- Applicable to very large graphs (millions of nodes) since node mini-batches are independent.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)

## Relation to Other Wiki Pages

- [graph-transformer](graph-transformer.md): NAGphormer exemplifies the hop-tokenization approach — alternative to ego-graph sampling and global attention.
- [zhao2021gophormer](zhao2021gophormer.md): Gophormer samples an ego-graph and attends over nodes; NAGphormer pre-aggregates hops and attends over hop levels — complementary designs.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT's multi-hop neighborhood sampling is closer to Gophormer; NAGphormer's hop-token idea is a possible future direction for REGs.
