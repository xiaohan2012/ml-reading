---
title: "Supervised Learning on Relational Databases with Graph Neural Networks"
tags: [source, relational-deep-learning, gnn, database]
sources: [cvitkovic2020rdb]
updated: 2026-05-06
---

# Supervised Learning on Relational Databases with Graph Neural Networks

**Source:** https://arxiv.org/abs/2002.02046
**Title:** Supervised Learning on Relational Databases with Graph Neural Networks
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Cvitkovic
**Venue:** arXiv 2020

## Summary

- **What:** No principled, general method existed for supervised learning directly on relational databases — practitioners relied on manual feature engineering or flat-table AutoML.
- **How:** Maps an RDB to a heterogeneous directed graph (rows = nodes, FK references = directed edges) and trains a GNN via subgraph batching with the RDBToGraph algorithm.
- **So what:** First explicit RDB-as-GNN formulation; beats Deep Feature Synthesis on 2/3 datasets; directly precedes the RDL framework of Fey et al. (2024).

## Challenges & Novelty

Before this work, supervised learning on relational databases required either manual feature engineering (expensive, domain-specific) or automated feature synthesis tools like DFS (featuretools), which operate on flat tables and discard the relational graph structure. No prior work had framed the problem as node classification on a heterogeneous graph.

- **RDB structure is naturally a heterogeneous graph:** rows are nodes typed by their table; foreign key references are typed directed edges — the correspondence is 1:1 and loses no structural information.
- **Subgraph batching via RDBToGraph:** for each prediction target (row $i$ in table $k$), the algorithm traverses FK chains outward (ancestors and their descendants) to collect a connected subgraph containing all relational context — no domain knowledge required.
- **No temporal handling:** the paper ignores seed-time constraints; any future row reachable via FK chains can leak into the subgraph, a limitation later addressed by the RDL Training Table abstraction.

## Relation to Prior Work

| Method | Uses relational structure | No feature engineering | Temporal leakage prevention |
|---|---|---|---|
| Manual features + XGBoost | Partial (manual) | No | Manual |
| DFS (featuretools) | Yes | Yes (automatic) | No |
| **Cvitkovic (2020)** | Yes (as graph) | Yes | No |
| [fey2024rdlposition](fey2024rdlposition.md) (RDL) | Yes | Yes | Yes (Training Table + seed time) |

- [fey2024rdlposition](fey2024rdlposition.md): the RDL blueprint cites this paper as foundational; adds temporal ordering, seed-time gating, and a formal Training Table abstraction to Cvitkovic's graph interpretation.
- [chen2025relgnn](chen2025relgnn.md): RelGNN's bridge/hub analysis builds on the PK-FK graph structure first characterized here; adds structural awareness Cvitkovic's homogeneous GNN lacked.

## Technical Details

**RDB → Graph mapping:**

| Relational Database | Graph |
|---|---|
| Row | Node |
| Table | Node type |
| FK column | Edge type |
| Non-FK column | Node feature |
| FK reference $A \to B$ | Directed edge $A \to B$ |
| Target column value | Node label |

Supervised learning on an RDB reduces to **node classification on a heterogeneous directed graph**.

**RDBToGraph algorithm.** Given database $D$, target table $k$, and target row $i$:
1. Start from row $i$ (the prediction target node)
2. Traverse all FK chains outward — add ancestor rows (rows that $i$ references transitively) and descendants of those ancestors
3. Return the induced connected subgraph

This produces a subgraph containing all relational context reachable from the target row. Complexity: $O(|V_\text{sub}| + |E_\text{sub}|)$.

**Model.** A standard GNN $g_\theta$ is applied to `RDBToGraph(D, i, k)`, outputting a node embedding for row $i$ used for prediction. Trained by SGD on labeled rows.

## Experiments

- Outperforms Deep Feature Synthesis (featuretools + XGBoost) on 2 out of 3 datasets.
- Performance gap is largest on tasks where relational context spans multiple hops — confirming that graph structure carries signal not captured by flat aggregation.

## Entities & Concepts

- [relational-deep-learning](relational-deep-learning.md)
- [relational-entity-graph](relational-entity-graph.md)
- [graph-neural-network](graph-neural-network.md)
- [training-table](training-table.md)
