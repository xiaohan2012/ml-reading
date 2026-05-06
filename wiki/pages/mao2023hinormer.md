---
title: "HINormer: Representation Learning on Heterogeneous Information Networks with Graph Transformer"
tags: [source, graph-transformer, heterogeneous-graph]
sources: [mao2023hinormer]
updated: 2026-05-04
---

# HINormer: Representation Learning on Heterogeneous Information Networks with Graph Transformer

**Source:** https://arxiv.org/abs/2302.11329
**Title:** HINormer: Representation Learning On Heterogeneous Information Networks with Graph Transformer
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Qiheng Mao, Zemin Liu, Chenghao Liu, Jianling Sun
**Venue:** WWW 2023

## Summary

HINormer addresses limitations of message-passing HGNNs (limited expressiveness, over-smoothing, over-squashing) on heterogeneous information networks (HINs) by introducing a Graph Transformer architecture that combines two modules: a **local structure encoder** capturing heterogeneous neighborhood structure via type-aware message passing within a local window, and a **heterogeneous relation encoder** that processes global cross-type node interactions via Transformer attention.

The local encoder handles the structural heterogeneity (different node/edge types, meta-path relationships) while the global encoder captures long-range cross-type dependencies that HGNN message passing cannot reach. The combination provides both structural and semantic richness for node representation learning on HINs.

## Key Takeaways

- Combines local HIN message passing (structure) + global Transformer attention (long-range cross-type) for heterogeneous graph representation.
- Addresses over-squashing in HGNNs via global attention range; addresses over-smoothing via type-aware aggregation.
- Local + global design mirrors GraphGPS's MPNN ∥ GlobalAttn philosophy, adapted for heterogeneous graphs.
- Evaluated on HIN node classification benchmarks (ACM, DBLP, Freebase).

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md)

## Relation to Other Wiki Pages

- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md): HINormer is an alternative GT for heterogeneous graphs; HGT uses meta-relation parameterization; HINormer uses local+global encoder split.
- [graph-transformer](graph-transformer.md): represents the heterogeneous GT design space.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): same local+global architecture philosophy applied to heterogeneous graphs.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT subsumes heterogeneous GT design but additionally handles temporality and schema constraints that HINormer ignores.
