---
title: "Gophormer: Ego-Graph Transformer for Node Classification"
tags: [source, graph-transformer, scalability]
sources: [zhao2021gophormer]
updated: 2026-05-04
---

# Gophormer: Ego-Graph Transformer for Node Classification

**Source:** https://arxiv.org/abs/2110.13094
**Title:** Gophormer: Ego-Graph Transformer for Node Classification
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Jianan Zhao, Chaozhuo Li, Qianlong Wen, Yiqi Wang, Yuming Liu, Hao Sun, Xing Xie, Yanfang Ye
**Venue:** arXiv 2021

## Summary

Gophormer addresses the scalability problem of Graph Transformers for **node classification** on large graphs. Full-graph Transformers require $O(N^2)$ attention and cannot be mini-batched like GNNs. The solution: sample an **ego-graph** (local neighborhood subgraph) for each target node and apply a Transformer only within that ego-graph. This enables mini-batch training with standard GPU memory budgets.

The ego-graph Transformer uses the full node features and edge structure within each sampled subgraph and applies multi-head self-attention to all nodes within it. A **progressive label propagation** strategy is also introduced to leverage label information across the graph. The model is additionally augmented with a **consistency regularization** term to reduce variance from stochastic ego-graph sampling.

Gophormer is directly analogous to what RelGT does for relational entity graphs: sample a fixed-size subgraph around the target node, then apply Transformer attention within that subgraph.

## Key Takeaways

- **Ego-graph sampling**: apply full GT attention within a small sampled subgraph per node — scales to large graphs via mini-batching.
- Progressive label propagation enriches node representations with label context across the graph.
- Consistency regularization reduces variance from stochastic subgraph sampling.
- Establishes the ego-graph (local subgraph) paradigm for scalable GT node classification — directly used in RelGT.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)

## Relation to Other Wiki Pages

- [graph-transformer](graph-transformer.md): Gophormer exemplifies the neighborhood sampling scalability strategy for GTs.
- [relational-graph-transformer](relational-graph-transformer.md): RelGT's K=300 subgraph sampling design is the RDL analog of Gophormer's ego-graph approach; extended to heterogeneous temporal REGs.
- [chen2022nagphormer](chen2022nagphormer.md): NAGphormer takes a different approach — instead of sampling the subgraph, it aggregates hop-level features into a per-node sequence.
