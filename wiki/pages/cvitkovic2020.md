---
title: "Supervised Learning on Relational Databases with Graph Neural Networks"
tags: [source, relational-deep-learning, gnn, database]
sources: [cvitkovic2020]
updated: 2026-04-29
---

# Supervised Learning on Relational Databases with Graph Neural Networks

**Source:** https://arxiv.org/abs/2002.02046
**Title:** Supervised Learning on Relational Databases with Graph Neural Networks
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Cvitkovic
**Venue:** arxiv 2020
**Year:** 2020

## Summary

Cvitkovic (Amazon AWS, 2020) is the first paper to explicitly frame supervised learning on relational databases as a GNN problem, formalizing the RDB-as-graph interpretation that the RDL blueprint (Fey et al., 2024) would later build on. The core contribution is the observation that the correspondence between RDB concepts and graph concepts is direct and natural:

| Relational Database | Graph |
|---|---|
| Row | Node |
| Table | Node type |
| Foreign key column | Edge type |
| Non-FK column | Node feature |
| FK reference A→B | Directed edge A→B |
| Target column value | Node label |

Supervised learning on an RDB then reduces to **node classification on a heterogeneous directed graph**.

The **RDBToGraph algorithm** deterministically constructs a per-prediction subgraph: starting from the target row, it selects all ancestor rows (rows referenced via FK chains) and all descendants of those ancestors. This produces a connected subgraph containing all relational context relevant to a prediction. A GNN `g_theta(RDBToGraph(D, i, k))` is then trained by SGD on the resulting node classification problem.

The method outperforms Deep Feature Synthesis (DFS, automatic feature engineering) on 2 out of 3 datasets. Key limitations: no temporal awareness (ignores seed_time), naive subgraph selection (ancestors only), fixed heuristic rather than learned neighbor sampling.

## Key Takeaways

- First explicit formulation: RDB supervised learning = node classification on heterogeneous graph
- RDB → Graph mapping: rows=nodes, tables=node types, FK columns=edge types, non-FK columns=features
- RDBToGraph: deterministic subgraph selection via ancestor + descendant traversal; O(|V|+|E|)
- GNN `g_theta(RDBToGraph(D, i, k))` trained by SGD; no feature engineering required
- Beats Deep Feature Synthesis on 2/3 datasets
- Limitations: no temporal handling, no learned neighbor sampling, naive topology

## Entities & Concepts

- [relational-deep-learning](relational-deep-learning.md) — this paper establishes the conceptual foundation
- [relational-entity-graph](relational-entity-graph.md) — REG is a temporal, directed extension of Cvitkovic's RDB-as-graph
- [graph-neural-network](graph-neural-network.md) — GNNs applied to heterogeneous graphs from RDBs

## Relation to Other Wiki Pages

- [relational-deep-learning-position](relational-deep-learning-position.md): The RDL blueprint (Fey et al., 2024) cites this paper as foundational; REG formalizes and extends Cvitkovic's graph interpretation with temporal edges, heterogeneous node/edge features, and a principled training table abstraction
- [relational-entity-graph](relational-entity-graph.md): REG adds temporal ordering and formal event-stream semantics; Cvitkovic's graph is the static, non-temporal precursor
- [training-table](training-table.md): Cvitkovic's formulation ignores temporal leakage — the Training Table abstraction in the RDL blueprint directly addresses this gap
- [relgnn](relgnn.md): RelGNN's bridge/hub topology analysis is informed by the PK-FK structure Cvitkovic first characterized as a graph; RelGNN can be seen as a principled successor with temporal + structural awareness
