---
title: "RelBench: A Benchmark for Deep Learning on Relational Databases"
tags: [source, benchmark, relational-deep-learning, gnn]
sources: [relbench-v1]
updated: 2026-04-29
---

# RelBench: A Benchmark for Deep Learning on Relational Databases

**Source:** https://arxiv.org/abs/2407.20060
**Title:** RelBench: A Benchmark for Deep Learning on Relational Databases
**Date ingested:** 2026-04-29
**Type:** paper (benchmark / datasets)
**Authors:** Robinson, Ranjan, Hu, Huang, Han, Dobles, Fey, Lenssen, Das, Sun, Leskovec
**Venue:** NeurIPS 2024 (Datasets & Benchmarks Track)
**Year:** 2024

## Summary

Robinson, Ranjan, Hu, Huang, Han, Dobles, Fey, Lenssen, Yuan, Zhang, He, and Leskovec (Stanford / Kumo.AI) present the first comprehensive benchmark for Relational Deep Learning (RDL). Where the [RDL blueprint paper](relational-deep-learning-position.md) introduced the framework with a 2-database beta, this paper delivers the full infrastructure: 7 real-world relational databases, 30 predictive tasks, an open-source implementation, a public leaderboard, and the first empirical validation that RDL outperforms the current gold-standard of manual feature engineering.

The seven databases span e-commerce (rel-amazon, rel-avito, rel-hm), social platforms (rel-event, rel-stack), sports (rel-f1), and medical (rel-trial), with entity counts ranging from 74K to 41M and time spans from 2 weeks to 55 years. Tasks cover three types: entity classification (AUROC), entity regression (MAE), and recommendation (MAP@K). All datasets use temporal splits — training on rows up to `val_timestamp`, validated up to `test_timestamp`, tested after — with future data strictly hidden at inference time.

The central empirical finding comes from a **data scientist baseline**: an experienced practitioner manually engineered features and trained LightGBM on each task. RDL matches or outperforms this baseline on all classification and regression tasks while requiring **96% fewer human hours** and **94% fewer lines of code**. This is the first demonstration that end-to-end relational deep learning delivers on its core promise. The only notable weak point is recommendation tasks, where the gap between RDL and simple popularity baselines is smaller.

The open-source package is built on PyTorch Frame (multimodal tabular encoding) + PyTorch Geometric (GNN training), and provides database loading, training table construction, temporal data splitting, and standardized evaluation. The design prioritizes research flexibility: shared `val_timestamp`/`test_timestamp` across tasks in the same database explicitly enables multi-task learning and cross-task pretraining experiments.

## Key Takeaways

- **RDL vs. data scientist baseline**: end-to-end RDL matches or beats a human expert who manually feature-engineered + trained LightGBM, on all 30 classification/regression tasks — with 96% fewer human hours and 94% fewer lines of code. First empirical proof of the RDL promise.
- **7 databases, 30 tasks**: largest relational ML benchmark at publication time; databases vary by 3 orders of magnitude in size (74K–41M entities), cover 5 domains, and range from 3 to 15 tables.
- **LightGBM single-table baseline**: RDL consistently outperforms single-table LightGBM (only entity features, no cross-table joins); gap narrows for tasks with rich single-table features (e.g., rel-trial study-outcome has 28 columns).
- **Temporal split discipline**: all datasets use strict temporal splits; data after `test_timestamp` is hidden at inference. Temporal leakage during training is handled by the time-consistent computation graph from [Fey et al. 2024](relational-deep-learning-position.md).
- **Research flexibility by design**: shared timestamps across tasks enable future multi-task learning and pretraining research; raw tables are exposed to allow alternative graph constructions.

## Entities & Concepts

- [relbench](relbench.md)
- [relational-deep-learning](relational-deep-learning.md)
- [relational-entity-graph](relational-entity-graph.md)
- [training-table](training-table.md)
- [temporal-graph](temporal-graph.md)

## Relation to Other Wiki Pages

- [relational-deep-learning-position](relational-deep-learning-position.md): this paper is the empirical companion to the blueprint; it implements and validates the RDL pipeline at scale.
- [relbench](relbench.md): this paper defines RelBench v1 (7 datasets, 30 tasks); [relbench-v2](relbench-v2.md) later expands to 11 datasets + autocomplete tasks.
- [relational-graph-transformer](relational-graph-transformer.md): RelGT is evaluated on 21 tasks from this benchmark, outperforming HeteroGNN by up to 18.43%.
- [relational-transformer](relational-transformer.md): RT is pretrained on 6 of the 7 v1 databases (leave-one-out); achieves 93% of supervised AUROC zero-shot.
