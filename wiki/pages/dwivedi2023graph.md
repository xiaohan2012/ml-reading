---
title: "Graph Transformers for Large Graphs"
tags: [source, graph-transformer, scalability, benchmark]
sources: [dwivedi2023graph]
updated: 2026-05-04
---

# Graph Transformers for Large Graphs

**Source:** https://arxiv.org/abs/2312.11109
**Title:** Graph Transformers for Large Graphs
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Vijay Prakash Dwivedi, Yozen Liu, Anh Tuan Luu, Xavier Bresson, Neil Shah, Tong Zhao
**Venue:** arXiv 2023

## Summary

This paper (from the same lead author as Benchmarking-GNNs and LSPE) systematically studies the challenge of scaling Graph Transformers from small graphs (< 100 nodes, where global attention is feasible) to large graphs (millions to billions of nodes). It introduces new large-scale graph benchmarks and evaluates major scalability strategies: (1) sparse attention (Exphormer-style), (2) hierarchical clustering, (3) neighborhood sampling (Gophormer, NAGphormer), and (4) linear attention approximations.

The paper documents the accuracy-scalability tradeoffs across strategies, showing that no single approach dominates: sparse attention methods preserve more expressiveness but can still be expensive; sampling methods scale to billions of nodes but lose global context; hierarchical methods capture multi-scale structure but add architectural complexity. The work serves as a benchmark and landscape paper for the community working on large-scale GTs.

## Key Takeaways

- Scalable GT survey + new large-scale benchmarks covering millions/billions of nodes.
- Four scalability strategies evaluated: sparse attention, hierarchical, sampling, linear attention.
- No single winner: accuracy-scalability tradeoff is task- and graph-dependent.
- Fills the gap between small-graph GT benchmarks (Benchmarking-GNNs) and real-world large-graph needs.
- From the same Dwivedi/Bresson group that authored Benchmarking-GNNs, LSPE, and GT generalization.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)

## Relation to Other Wiki Pages

- [graph-transformer](graph-transformer.md): complements the GT landscape with a scalability-focused taxonomy.
- [dwivedi2020benchmarking](dwivedi2020benchmarking.md): same lead author; extends benchmark focus from small to large graphs.
- [shirzad2023exphormer](shirzad2023exphormer.md), [zhang2022hierarchical](zhang2022hierarchical.md), [zhao2021gophormer](zhao2021gophormer.md), [chen2022nagphormer](chen2022nagphormer.md): the scalable GT methods evaluated in this paper.
