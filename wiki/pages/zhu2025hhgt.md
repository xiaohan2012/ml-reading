---
title: "HHGT: Hierarchical Heterogeneous Graph Transformer"
tags: [source, graph-transformer, heterogeneous-graph, scalability]
sources: [zhu2025hhgt]
updated: 2026-05-04
---

# HHGT: Hierarchical Heterogeneous Graph Transformer

**Source:** https://arxiv.org/abs/2407.13158
**Title:** HHGT: Hierarchical Heterogeneous Graph Transformer for Heterogeneous Graph Representation Learning
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Qiuyu Zhu, Liang Zhang, Qianxiong Xu, Kaijun Liu, Cheng Long, Xiaoyang Wang
**Venue:** CIKM 2024

## Summary

HHGT addresses two limitations of existing GT approaches on Heterogeneous Information Networks (HINs): **(1) semantics at different distances differ** — a direct neighbor carries different semantic meaning than an indirect neighbor, so they shouldn't be aggregated uniformly; **(2) type-level interactions are insufficiently modeled** — existing heterogeneous GTs aggregate nodes of different types without explicitly modeling inter-type relationships.

HHGT proposes a **two-level hierarchical Transformer**: a **Type-level Transformer** that aggregates nodes of different types within each $k$-ring neighborhood (modeling inter-type relationships at each distance), followed by a **Ring-level Transformer** that aggregates the $k$-ring representations hierarchically (capturing distance-dependent semantics). This two-level design processes both type heterogeneity and structural hierarchy simultaneously.

Results show up to 24.75% improvement in NMI and 29.25% in ARI for node clustering on ACM compared to prior baselines.

## Key Takeaways

- **Two-level hierarchy**: Type-level Transformer (inter-type, within ring) → Ring-level Transformer (inter-ring, across distances).
- Each ring distance is processed separately before aggregation — preserves semantic distinctions between near and distant neighbors.
- Addresses both heterogeneity (type-level attention) and scale/structure (ring-level hierarchy).
- Strong results on node clustering tasks (NMI/ARI) on HIN benchmarks.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md)

## Relation to Other Wiki Pages

- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md): HHGT is a more structured heterogeneous GT compared to HGT's meta-relation parameterization.
- [mao2023hinormer](mao2023hinormer.md): HINormer uses local+global split; HHGT uses ring-level hierarchy — both address heterogeneous GT but with different structural decompositions.
- [zhu2023hierarchical](zhu2023hierarchical.md): same hierarchical + heterogeneous motivation; HHGT adds explicit ring-distance semantics.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT also handles heterogeneity + temporality but in the relational database (REG) setting, not general HINs.
