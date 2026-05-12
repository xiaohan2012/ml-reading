---
title: "RelBench: A Benchmark for Deep Learning on Relational Databases"
tags: [source, benchmark, relational-deep-learning, gnn]
sources: [robinson2024relbench]
updated: 2026-05-06
---

# RelBench: A Benchmark for Deep Learning on Relational Databases

**Source:** https://arxiv.org/abs/2407.20060
**Title:** RelBench: A Benchmark for Deep Learning on Relational Databases
**Date ingested:** 2026-04-29
**Type:** paper (benchmark / datasets)
**Authors:** Joshua Robinson, Rishabh Ranjan, Weihua Hu, Kexin Huang, Jiaqi Han, Alejandro Dobles, Matthias Fey, Jan E. Lenssen, Yuan Zhang, Zecheng Zhang, Xinwei He, Jure Leskovec
**Venue:** NeurIPS 2024 (Datasets & Benchmarks Track)

## Summary

- **What:** The RDL blueprint paper shipped only a 2-database beta; no comprehensive benchmark existed to evaluate and compare RDL methods on diverse realistic relational databases.
- **How:** RelBench provides 7 real-world relational databases, 30 tasks (classification, regression, recommendation), strict temporal splits, an open-source package, and a public leaderboard — with a data scientist baseline for calibration.
- **So what:** First empirical proof that end-to-end RDL matches or beats expert manual feature engineering across all classification and regression tasks, at 96% fewer human hours and 94% fewer lines of code.

## Challenges & Novelty

A benchmark for relational ML must cover database diversity (size, domain, table count), enforce temporal consistency without leakage, support multiple task types, and provide a meaningful human baseline to calibrate progress. No such benchmark existed prior to RelBench — prior GNN benchmarks (OGB, TGB) used homogeneous or simple graph structures, not multi-table relational schemas.

- **No realistic multi-table benchmark:** OGB/TGB evaluate on homogeneous or knowledge graphs; no benchmark tested end-to-end learning across PK-FK-linked heterogeneous tables.
- **Temporal leakage risk:** standard ML practice on time-series data does not generalize to multi-hop relational joins, where a neighbor's row may have been created after the seed time.
- **Calibrating "good enough":** without a human expert baseline, it's impossible to know whether RDL results are practically useful or just better than a weak strawman.

## Relation to Prior Work

| Benchmark                                | Graph type             | Temporal | Multi-table | Human baseline       |
| ---------------------------------------- | ---------------------- | -------- | ----------- | -------------------- |
| OGB                                      | Homogeneous / KG       | No       | No          | No                   |
| TGB                                      | Temporal homogeneous   | Yes      | No          | No                   |
| **RelBench v1**                          | Heterogeneous temporal | Yes      | Yes         | Yes (data scientist) |
| [gu2026relbenchv2](gu2026relbenchv2.md) (v2) | Heterogeneous temporal | Yes      | Yes         | Yes                  |

- [fey2024rdlposition](fey2024rdlposition.md): this paper is the empirical companion — implements and validates the RDL pipeline defined in the blueprint at scale.
- [gu2026relbenchv2](gu2026relbenchv2.md): v2 expands to 11 datasets, adds autocomplete tasks, and integrates TGB and 4DBInfer.

## Technical Details

**Databases.** Seven real-world databases across 5 domains: e-commerce (rel-amazon, rel-avito, rel-hm), social (rel-event, rel-stack), sports (rel-f1), medical (rel-trial). Entity counts range from 74K (rel-f1) to 41M (rel-event). Table counts range from 3 to 15.

**Task types.** Three task types, each measured by a standard metric:
- *Entity classification* — AUROC
- *Entity regression* — MAE (normalized)
- *Recommendation* — MAP@K

**Temporal splits.** All datasets use strict temporal splits: train on rows up to `val_timestamp`, validate up to `test_timestamp`, test on rows after. Future data is hidden at inference. The time-consistent computation graph from [fey2024rdlposition](fey2024rdlposition.md) enforces leakage-free neighbor sampling.

**Shared timestamps across tasks.** Tasks within the same database share `val_timestamp`/`test_timestamp`, enabling future multi-task learning and cross-task pretraining experiments.

**Data scientist baseline.** An experienced practitioner manually engineered task-specific temporal aggregations and multi-hop joins for each task, then trained LightGBM — representing the current gold standard in production.

## Experiments

- HeteroGraphSAGE (RDL) matches or outperforms the data scientist LightGBM baseline on all 30 classification and regression tasks with 96% fewer human hours and 94% fewer lines of code.
- Single-table LightGBM (entity features only, no cross-table joins) is consistently weaker than RDL, gap narrows for tasks with rich single-table features (e.g., rel-trial with 28 columns).
- Recommendation tasks show a smaller RDL advantage — simple popularity baselines remain competitive, suggesting recommendation over relational graphs is an open problem.

## Useful Resources

- [RelBench getting-started page](https://relbench.stanford.edu/start/) — install, datasets, task API overview.
  - [Tutorial: load_data.ipynb](https://colab.research.google.com/github/snap-stanford/relbench/blob/main/tutorials/load_data.ipynb) — load a database, inspect tables, materialize a training table.
  - [Tutorial: train_model.ipynb](https://colab.research.google.com/github/snap-stanford/relbench/blob/main/tutorials/train_model.ipynb) — end-to-end HeteroGraphSAGE training on a RelBench task.

## Entities & Concepts

- [relbench](relbench.md)
- [relational-deep-learning](relational-deep-learning.md)
- [relational-entity-graph](relational-entity-graph.md)
- [training-table](training-table.md)
- [temporal-graph](temporal-graph.md)
