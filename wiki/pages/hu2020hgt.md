---
title: "Heterogeneous Graph Transformer (HGT)"
tags: [source, graph-transformer, heterogeneous-graph, temporal-graph]
sources: [hu2020hgt]
updated: 2026-05-06
---

# Heterogeneous Graph Transformer (HGT)

**Source:** https://arxiv.org/abs/2003.01332
**Title:** Heterogeneous Graph Transformer
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Ziniu Hu, Yuxiao Dong, Kuansan Wang, Yizhou Sun
**Venue:** WWW 2020

## Summary

- **What:** Standard GNNs assume all nodes share the same feature distribution, failing on heterogeneous graphs where nodes and edges have different semantic types.
- **How:** HGT parameterizes all weight matrices by the meta-relation triplet `<τ(s), φ(e), τ(t)>` (source type, edge type, target type), enabling type-specific attention and message passing with structured parameter sharing, plus a relative temporal encoding for dynamic graphs.
- **So what:** HGT outperforms GNN baselines by 9–21% on Open Academic Graph (179M nodes, 2.2B edges), demonstrating scalable heterogeneous graph learning at web scale.

## Challenges & Novelty

Prior GNNs (GCN, GAT, GraphSAGE) treat all nodes and edges as homogeneous, losing the semantic distinctions between entity types that are essential in knowledge graphs, academic networks, and relational data. HGT introduces type-aware parameterization and temporal encoding that preserves these distinctions while remaining scalable via heterogeneous mini-batch sampling.

- **Type-agnostic message passing:** Standard GNNs apply the same weight matrices across all nodes, failing to model distribution differences between types (e.g., papers vs. authors vs. venues have incompatible feature spaces).
- **Time-slicing for dynamic graphs:** Prior approaches discretize temporal graphs into snapshots, losing fine-grained temporal ordering and creating arbitrary window boundaries; HGT instead keeps all edges with timestamps and encodes relative time gaps.
- **Scalability to web-scale heterogeneous graphs:** Existing methods lacked a sampling strategy that preserves heterogeneous structure at billion-edge scale; HGT introduces HGSampling for this.

## Relation to Prior Work

| Model | Heterogeneous | Temporal | Scale |
|---|---|---|---|
| GCN / GAT / GraphSAGE | No | No | Medium |
| RGCN ([schlichtkrull2018rgcn](schlichtkrull2018rgcn.md)) | Yes | No | Medium |
| **HGT** | **Yes** | **Yes (RTE)** | **Web-scale** |

- [schlichtkrull2018rgcn](schlichtkrull2018rgcn.md): RGCN handles relation types but uses per-relation matrices without the meta-relation triplet structure, leading to parameter explosion or basis decomposition as a workaround.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GPS is the homogeneous counterpart — hybrid MPNN+GlobalAttn; HGT is the heterogeneous-graph-specific GT operating without global attention.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT uses HGT as the primary GT baseline and shows it underperforms HeteroGNN on 19/21 RelBench tasks even when augmented with Laplacian PE.
- [chen2025relgnn](chen2025relgnn.md): RelGNN's bridge/hub analysis explains *why* HGT underperforms on RDL — PK-FK edges carry no semantic type meaning, so HGT's meta-relation inductive bias breaks down.

## Technical Details

**Meta-Relation Triplet.** All weight matrices are indexed by `<τ(s), φ(e), τ(t)>` — source node type, edge type, target node type. This enables type-specific parameterization while sharing parameters more efficiently than a fully distinct matrix per relation.

**Three-Step HGT Layer:**

1. *Heterogeneous Mutual Attention* — type-specific K/Q projections per node type; W^ATT per edge type; meta-relation prior µ scales attention by relation importance:
$$\text{ATT-head}^i(s,e,t) = \Big(K^i_{\tau(s)}(s)\ W^{ATT}_{\phi(e)}\ Q^i_{\tau(t)}(t)^\top\Big) \cdot \frac{\mu_{\langle \tau(s), \phi(e), \tau(t) \rangle}}{\sqrt{d}}$$

2. *Heterogeneous Message Passing* — type-specific M-Linear per source type, then W^MSG per edge type.

3. *Target-Specific Aggregation* — attention-weighted sum of messages → A-Linear per target type → residual connection.

**Relative Temporal Encoding (RTE).** Compute relative time gap $\Delta T(t,s) = T(t) - T(s)$, encode with sinusoid basis + tunable T-Linear projection, added to source representation before attention. Sinusoids generalize to unseen time gaps; avoids information loss from time-window slicing.

**HGSampling.** Heterogeneous mini-batch sampling that preserves graph structure and temporal ordering, enabling training on 179M nodes / 2.2B edges (Open Academic Graph).

## Experiments

- HGT outperforms all GNN baselines by 9–21% on Paper-Field prediction, Paper-Venue prediction, and Author Disambiguation on OAG.
- On RelBench, HGT underperforms the HeteroGNN baseline on most tasks; adding Laplacian PE (HGT+PE) costs up to 8.62× per-epoch overhead with inconsistent gains (per [dwivedi2025relgt](dwivedi2025relgt.md)).

## Entities & Concepts

- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md)
- [graph-transformer](graph-transformer.md)
- [temporal-graph](temporal-graph.md)
