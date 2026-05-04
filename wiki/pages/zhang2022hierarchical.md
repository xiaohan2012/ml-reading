---
title: "Hierarchical Graph Transformer with Adaptive Node Sampling"
tags: [source, graph-transformer, scalability]
sources: [zhang2022hierarchical]
updated: 2026-05-04
---

# Hierarchical Graph Transformer with Adaptive Node Sampling

**Source:** https://arxiv.org/abs/2210.03930
**Title:** Hierarchical Graph Transformer with Adaptive Node Sampling
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Zaixi Zhang, Qi Liu, Qingyong Hu, Chee-Kong Lee
**Venue:** NeurIPS 2022

## Summary

This paper identifies two deficiencies in existing Graph Transformers that limit performance on large graphs: **(1) sampling strategies are agnostic to graph characteristics and the training process** — most methods use static, random, or topology-only sampling without considering node importance from a learning perspective; **(2) most sampling strategies focus only on local neighbors** and miss long-range dependencies. To address both, the paper proposes a **hierarchical GT with adaptive node sampling**.

The model operates at two levels: a **local level** where standard GT attention is applied to an adaptive neighborhood, and a **global level** where a hierarchical clustering approach coarsens the graph to capture long-range dependencies efficiently. The node sampling policy adapts dynamically during training — nodes contributing more to the training objective are sampled more frequently — combining the benefits of learned importance and hierarchical structure.

## Key Takeaways

- Adaptive sampling guided by training-time node importance — not static topology-only heuristics.
- Hierarchical design: local GT attention + global cluster-level aggregation captures both short and long-range dependencies.
- Competitive with or outperforms prior GTs on node classification benchmarks while scaling to larger graphs.
- Combines sampling (for scalability) with hierarchical structure (for long-range).

## Entities & Concepts

- [graph-transformer](graph-transformer.md)

## Relation to Other Wiki Pages

- [graph-transformer](graph-transformer.md): representative of the hierarchical clustering scalability strategy.
- [alon2020bottleneck](alon2020bottleneck.md): hierarchical aggregation is one solution to the long-range over-squashing problem.
- [chen2022nagphormer](chen2022nagphormer.md): NAGphormer addresses scalability via hop-aggregated tokens; complementary sampling-based approach.
