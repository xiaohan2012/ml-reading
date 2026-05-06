---
title: "R-GCN: Modeling Relational Data with Graph Convolutional Networks"
tags: [source, gnn, heterogeneous-graph, knowledge-graph, relational]
sources: [schlichtkrull2018rgcn]
updated: 2026-04-29
---

# R-GCN: Modeling Relational Data with Graph Convolutional Networks

**Source:** https://arxiv.org/abs/1703.06103
**Title:** Modeling Relational Data with Graph Convolutional Networks
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Schlichtkrull, Kipf, Bloem, van den Berg, Titov, Welling
**Venue:** ESWC 2018
**Year:** 2018

## Summary

Schlichtkrull, Kipf, et al. (ESWC 2018) extend GCN to heterogeneous relational data by introducing **relation-specific weight matrices**. The core motivation: standard GCNs use a single shared weight matrix for all edges, which is inappropriate when edges have different semantic types (e.g., *born_in*, *works_at*, *married_to* in a knowledge graph). R-GCN gives each relation type its own transformation matrix.

**R-GCN layer update:**

```
h_i^{l+1} = sigma( sum_{r in R} sum_{j in N_r(i)} (1/c_{i,r}) W_r^l h_j^l + W_0^l h_i^l )
```

where `N_r(i)` is the set of neighbors of node `i` under relation `r`, `c_{i,r}` is a normalization constant, and `W_0^l h_i^l` is a self-loop term.

**Parameter explosion problem**: with `|R|` relation types, each requiring its own `W_r`, parameter count scales linearly with relations. Two regularization strategies address this: (1) **Basis decomposition**: `W_r = sum_b a_{rb} V_b` — shared basis matrices `V_b` with relation-specific coefficients `a_{rb}`; (2) **Block-diagonal decomposition**: `W_r = diag(Q_{1r}, ..., Q_{Br})` — sparse block structure. Basis decomposition enables parameter sharing between rare and frequent relations.

Applied to two knowledge graph completion tasks: **entity classification** (R-GCN as encoder alone) and **link prediction** (R-GCN encoder + DistMult decoder). On FB15k-237, achieves 29.8% MRR improvement over decoder-only DistMult.

## Key Takeaways

- Core idea: relation-type-specific weight matrices `W_r^l` per layer — extends GCN to heterogeneous multi-relational graphs
- Layer equation: `h_i^{l+1} = sigma( sum_r sum_{j in N_r(i)} (1/c_{ir}) W_r^l h_j^l + W_0^l h_i^l )`
- Basis decomposition: `W_r = sum a_{rb} V_b` — controls parameter explosion, improves rare-relation generalization
- Block-diagonal decomposition: `W_r = diag(Q_{1r}, ..., Q_{Br})` — sparsity constraint per relation
- Combines with factorization decoders (DistMult) for link prediction via encoder-decoder architecture
- First GCN model for knowledge graphs; precursor to HGT's meta-relation triplet parameterization

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md) — R-GCN extends GCN to multi-relational graphs
- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md) — HGT's type-parameterized weights generalize R-GCN's relation-specific matrices
- [relational-entity-graph](relational-entity-graph.md) — REGs are heterogeneous graphs; R-GCN's relation-typing applies directly

## Relation to Other Wiki Pages

- [hu2020hgt](hu2020hgt.md): HGT's meta-relation triplet (source_type, edge_type, target_type) is a generalization of R-GCN's single-relation typing — HGT distinguishes not just edge type but also source and target node types
- [kipf2017gcn](kipf2017gcn.md): R-GCN extends GCN's `c_vw W h_w` message to `(1/c_{ir}) W_r^l h_j^l` — adding relation specificity
- [fey2024rdlposition](fey2024rdlposition.md): RDL's blueprint identifies R-GCN as an ancestor; HeteroGraphSAGE in RelBench baseline generalizes the same multi-relational idea to inductive settings
- [chen2025relgnn](chen2025relgnn.md): RelGNN's FUSE operation on PK-FK graphs can be viewed as a successor to R-GCN's relation-specific aggregation, but specialized for database-style topology
