---
title: Relational Deep Learning (RDL)
tags: [concept, relational-deep-learning, gnn, database]
sources: [relational-deep-learning-position, relbench-v1, relational-graph-transformer, relbench-v2, relational-transformer, relgnn, griffin, cvitkovic2020]
updated: 2026-04-29
---

# Relational Deep Learning (RDL)

## Description

Relational Deep Learning is an end-to-end learning framework that converts multi-table relational databases into graph structures, enabling neural networks to learn directly from relational data without manual feature engineering. A relational database $(T, R)$ is represented as a [Relational Entity Graph](relational-entity-graph.md), where nodes are table rows (entities) and edges encode primary-foreign key relationships.

**RDL pipeline** (Fey et al. 2024): (1) convert database to [Relational Entity Graph](relational-entity-graph.md); (2) multimodal column encoders → initial node embeddings; (3) heterogeneous temporal message-passing GNN; (4) task-specific head. GNN message passing is mathematically equivalent to SQL `JOIN`+`AGGREGATE` — making GNNs a natural, differentiable replacement for manual feature engineering. Tasks are specified via a [Training Table](training-table.md) of (EntityID, seed\_time, label) triples; temporal leakage is prevented by time-consistent computation graphs that filter neighbors to $\tau(w) \leq t$.

**Standard baseline**: HeteroGraphSAGE backbone + temporal neighbor sampling + PyTorch Frame multimodal encoders. **Empirical baseline**: RDL outperforms a data-scientist (manual feature engineering + LightGBM) on all 30 RelBench v1 classification/regression tasks with 96% fewer human hours and 94% fewer lines of code (Robinson et al. 2024).

**Task types**: entity classification, entity regression, recommendation (forecasting tasks); and [autocomplete tasks](autocomplete-tasks.md) (predict missing attribute values in existing columns — introduced in RelBench v2).

**Architecture landscape** (as of 2026):
- *Supervised GNNs*: HeteroGraphSAGE (baseline), [RelGNN](relgnn.md) (composite message passing, SOTA 27/30 RelBench tasks), ContextGNN (recommendation)
- *Graph Transformers*: [RelGT](relational-graph-transformer.md) — 5-element tokenization, schema-specific, best supervised results
- *Foundation models*: [Relational Transformer (RT)](relational-transformer.md) — cell-level tokenization, schema-agnostic, zero-shot generalization; [Griffin](griffin.md) — GNN + unified tabular encoders, pretrained across 150M+ nodes


**Key insight from the literature**: the relational structure (multi-table, primary-foreign key links) consistently outperforms single-table baselines (LightGBM) across all task types, justifying the RDL paradigm.

The field is moving toward [Relational Foundation Models](relational-foundation-model.md) — pretrained models that generalize across schemas and tasks with zero-shot or few-shot prompting.

## Appearances in Sources

- [relational-deep-learning-position](relational-deep-learning-position.md) — founding blueprint; defines REG, Training Table, temporal neighbor sampling, GNN-as-SQL insight
- [relbench-v1](relbench-v1.md) — empirical validation at scale; 7 databases, 30 tasks, data scientist comparison
- [relational-graph-transformer](relational-graph-transformer.md) — RelGT as a GT solution within the RDL framework
- [relbench-v2](relbench-v2.md) — benchmark expansion; confirms RDL outperforms single-table baselines across all new task types
- [relational-transformer](relational-transformer.md) — RT as a foundation model approach to RDL
- [relgnn](relgnn.md) — RelGNN: bridge/hub topology analysis + composite message passing; SOTA 27/30 RelBench v1 tasks
- [griffin](griffin.md) — Griffin: first pretrained foundation model for RDBs; unified text+float encoders + hierarchical aggregation
- [cvitkovic2020](cvitkovic2020.md) — Cvitkovic (2020): first paper to frame RDB supervised learning as a GNN node classification problem; RDBToGraph algorithm; precursor to RDL blueprint

## Related Concepts

- [relational-entity-graph](relational-entity-graph.md) — the graph representation used in RDL
- [training-table](training-table.md) — the task specification abstraction; (EntityID, seed\_time, label) triples
- [relbench](relbench.md) — the benchmark for evaluating RDL methods
- [graph-transformer](graph-transformer.md) — architecture family applied in RDL
- [relational-foundation-model](relational-foundation-model.md) — the frontier of RDL research
- [autocomplete-tasks](autocomplete-tasks.md) — new RDL task type for attribute inference
