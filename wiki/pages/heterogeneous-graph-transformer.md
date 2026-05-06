---
title: Heterogeneous Graph Transformer (HGT)
tags:
  - entity
  - graph-transformer
  - heterogeneous-graph
sources:
  - hu2020hgt
  - dwivedi2025relgt
  - chen2025relgnn
updated: 2026-04-29
---

# Heterogeneous Graph Transformer (HGT)

## Description

HGT (Hu et al., WWW 2020) is a Graph Transformer designed for heterogeneous graphs. The core mechanism: parameterize all weight matrices by the **meta-relation triplet** `<τ(s), φ(e), τ(t)>` (source node type, edge type, target node type), enabling type-specific K/Q/V projections and W^ATT/W^MSG matrices while sharing parameters across meta-relations with few occurrences. It additionally supports dynamic graphs via **Relative Temporal Encoding (RTE)**: compute relative time gap ΔT(t,s) = T(t) − T(s), encode with sinusoid basis + tunable projection, add to source node rep before attention. HGT scales to 179M nodes / 2B edges via its HGSampling algorithm. Implemented in PyTorch Geometric as `HGTConv`.

**Core limitation for PK-FK graphs**: HGT's meta-relation inductive bias assumes semantically meaningful edge types — a valid assumption for knowledge graphs and academic citation networks, but not for relational databases where PK-FK edges are schema constraints carrying no inherent semantics. [RelGNN](chen2025relgnn.md) formally characterizes this mismatch: bridge/hub table topology causes redundant and imbalanced information flow that HGT's design does not address.

**Performance on RelBench**: HGT underperforms the HeteroGNN RDL baseline on 19/21 tasks. Adding Laplacian PE (HGT+PE) improves results on 8/21 tasks but requires computing Laplacian PE per node type (since REGs are heterogeneous), adding 1.8×–8.62× runtime overhead per epoch.

[RelGT](dwivedi2025relgt.md) consistently outperforms both HGT and HGT+PE across RelBench while avoiding the costly PE precomputation — using its [subgraph GNN PE](subgraph-gnn-pe.md) instead.

## Appearances in Sources

- [hu2020hgt](hu2020hgt.md) — original paper; meta-relation triplet parameterization, RTE, HGSampling; 9%–21% gains on OAG
- [dwivedi2025relgt](dwivedi2025relgt.md) — used as the primary GT baseline; shown to be inadequate for RDL
- [chen2025relgnn](chen2025relgnn.md) — explains *why* standard heterogeneous GNNs underperform: semantic-edge assumptions don't hold for PK-FK links; bridge/hub topology causes redundant and imbalanced message passing

## Related Concepts

- [graph-transformer](graph-transformer.md) — parent concept
- [relational-entity-graph](relational-entity-graph.md) — the graph type HGT is applied to
- [subgraph-gnn-pe](subgraph-gnn-pe.md) — RelGT's alternative to HGT's Laplacian PE
