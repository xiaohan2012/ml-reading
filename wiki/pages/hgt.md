---
title: "Heterogeneous Graph Transformer (HGT)"
tags: [source, graph-transformer, heterogeneous-graph, temporal-graph]
sources: [hgt]
updated: 2026-04-29
---

# Heterogeneous Graph Transformer (HGT)

**Source:** https://arxiv.org/abs/2003.01332
**Title:** Heterogeneous Graph Transformer
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Hu, Dong, Wang, Sun
**Venue:** WWW 2020
**Year:** 2020

## Summary

Hu, Dong, Wang, and Sun (UCLA / Microsoft Research) introduce the Heterogeneous Graph Transformer (HGT) for modeling large-scale heterogeneous and dynamic graphs. The core challenge: standard GNNs assume all nodes share the same feature distribution, but heterogeneous graphs have nodes and edges of different semantic types requiring type-specific treatment.

**Meta-Relation Triplet Parameterization.** HGT's key design principle: parameterize all weight matrices by the meta-relation triplet `<τ(s), φ(e), τ(t)>` — source node type, edge type, target node type. This enables a parameter-sharing tradeoff: separate matrices per type handle distribution differences, but the triplet structure shares parameters more efficiently than keeping a fully distinct matrix per relation.

**Three-Step HGT Layer:**
1. *Heterogeneous Mutual Attention*: type-specific K/Q projections per node type; edge-type-specific W^ATT matrix in attention computation; meta-relation prior µ scales attention by relation importance:
$$ATT\text{-}head^i(s,e,t) = \Big(K^i_{\tau(s)}(s)\ W^{ATT}_{\phi(e)}\ Q^i_{\tau(t)}(t)^T\Big) \cdot \frac{\mu_{\langle \tau(s), \phi(e), \tau(t) \rangle}}{\sqrt{d}}$$
2. *Heterogeneous Message Passing*: type-specific M-Linear projection per source type, then W^MSG per edge type
3. *Target-Specific Aggregation*: attention-weighted sum of messages → A-Linear per target type → residual connection

**Relative Temporal Encoding (RTE).** For dynamic graphs: keep all edges with their timestamps instead of time-slicing. Compute relative time gap ΔT(t,s) = T(t) − T(s), encode with sinusoid basis + tunable T-Linear projection, add to source node representation before attention. Handles arbitrary and unseen time gaps (sinusoids generalize).

**HGSampling.** Heterogeneous mini-batch graph sampling algorithm for training on web-scale data (178M nodes, 2B edges on Open Academic Graph). Samples maintain the heterogeneous graph structure and relative temporal ordering.

**Results.** Evaluated on Open Academic Graph (OAG: 179M nodes, 2.2B edges) on Paper-Field prediction, Paper-Venue prediction, and Author Disambiguation. HGT outperforms all GNN baselines by 9%–21%.

## Key Takeaways

- **Meta-relation triplet is the core design**: `<τ(s), φ(e), τ(t)>` enables type-specific parameterization with parameter sharing — each weight matrix is indexed by source type, edge type, or target type rather than their full cross-product.
- **Type-specific K/Q for attention + W^ATT per edge type**: different node types have different feature distributions; same meta-relation produces different interactions even for the same node-type pair if edge type differs.
- **RTE over time-slicing**: keeping a single temporal graph with relative time encoding avoids structural information loss that occurs when splitting by time windows.
- **Limitation on PK-FK relational data**: HGT assumes semantically meaningful edge types (the meta-relation semantics enable type-specific learning). PK-FK edges in relational databases carry no intrinsic semantics — they're schema constraints — so HGT's key inductive bias breaks down. This is why HGT underperforms on RelBench (see [RelGNN](relgnn.md) for the formal analysis).
- **Scale**: first GNN demonstrated at 179M nodes / 2B edges scale via heterogeneous mini-batch sampling.

## Entities & Concepts

- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md)
- [graph-transformer](graph-transformer.md)
- [temporal-graph](temporal-graph.md)

## Relation to Other Wiki Pages

- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md): this is the source page for the HGT entity page.
- [relgnn](relgnn.md): RelGNN's bridge/hub analysis specifically explains *why* HGT underperforms on RDL — meta-relation semantics don't hold for PK-FK links; HGT's type-specific attention assumptions cause redundant and imbalanced information flow.
- [relational-graph-transformer](relational-graph-transformer.md): RelGT uses HGT as the primary GT baseline and shows it underperforms HeteroGNN on 19/21 RelBench tasks even with +PE.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GPS is the homogeneous counterpart — hybrid MPNN+GlobalAttn; HGT is the heterogeneous-graph-specific GT.
- [temporal-graph](temporal-graph.md): HGT's RTE is one approach to dynamic graph learning; contrasts with CTDG/DTDG paradigms in [TGM](tgm-temporal-graph-modelling.md).
