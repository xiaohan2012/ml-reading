---
title: Positional and Structural Encodings (PE/SE)
tags: [concept, positional-encoding, graph-transformer]
sources: [graphgps, relational-graph-transformer, san, graphormer]
updated: 2026-04-29
---

# Positional and Structural Encodings (PE/SE)

## Description

Positional encodings (PE) and structural encodings (SE) are additional node/edge features that inject topological awareness into GNNs and Graph Transformers, which otherwise cannot distinguish nodes without unique input features (1-WL limitation).

**PE vs SE distinction** (GraphGPS taxonomy):
- **PE**: captures *position* — close nodes in the graph have similar PEs. Enables node identifiability for global attention.
- **SE**: captures *structural similarity* — nodes in similar subgraphs have similar SEs. Increases expressivity beyond 1-WL.

**Three levels for each:**
- **Local** (node features): position/structure within a neighborhood cluster
- **Global** (node features): position/structure within the whole graph
- **Relative** (edge features): pairwise distance or structural difference between nodes

**Key variants:**
- *LapPE*: Laplacian eigenvectors — global PE; expensive O(N³) precomputation; sign-ambiguous (addressed by SignNet)
- *RWSE* (Random-Walk SE): diagonal of m-step random walk matrix — local SE; cheap; captures cycle membership; most robust in practice
- *SignNet*: invariant aggregation of Laplacian eigenvectors via DeepSets/MLP; best single encoding in GPS ablations at moderate cost
- *Relative PE*: pairwise shortest-path or heat-kernel distances as edge features; enables pair-specific attention biases

**In the relational setting** ([RelGT](relational-graph-transformer.md)): standard PEs (Laplacian, RWSE) cannot be directly applied to heterogeneous temporal graphs. RelGT replaces them with schema-aware encodings: node type, hop distance from seed entity, temporal distance, and subgraph GNN PE.

## Appearances in Sources

- [graphgps](graphgps.md) — introduces the local/global/relative PE/SE taxonomy; ablation shows RWSE > LapPE for molecular tasks; MPNN is essential alongside PE/SE
- [relational-graph-transformer](relational-graph-transformer.md) — 5-element tokenization adapts PE/SE concepts to the RDL setting; [subgraph GNN PE](subgraph-gnn-pe.md) is the RDL analog of RWSE
- [san](san.md) — Kreuzer et al. (NeurIPS 2021): full Laplacian spectrum as LPE; eigenvectors as graph-domain sine functions; addresses eigenvalue multiplicity and variable eigenvector count; motivates GPS's choice to move to RWSE over LapPE
- [graphormer](graphormer.md) — Ying et al. (NeurIPS 2021): spatial encoding via shortest-path distance (SPD-indexed learnable bias in attention) + centrality encoding (degree-indexed learnable vectors); relative PE in attention directly

## Related Concepts

- [graph-transformer](graph-transformer.md) — GTs that incorporate PE/SE
- [subgraph-gnn-pe](subgraph-gnn-pe.md) — RelGT's schema-adapted PE for relational entity graphs
- [multi-element-tokenization](multi-element-tokenization.md) — RelGT's tokenization that includes PE/SE as elements
