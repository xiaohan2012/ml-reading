---
title: RelBench
tags: [entity, benchmark, relational-deep-learning]
sources: [relational-deep-learning-position, relbench-v1, relational-graph-transformer, relbench-v2, relational-transformer, relgnn, griffin]
updated: 2026-04-29
---

# RelBench

## Description

RelBench (Robinson et al., 2025) is the standard benchmark for [Relational Deep Learning](relational-deep-learning.md). **RelBench v1** comprised 7 datasets and 30 prediction tasks (entity classification, entity regression, recommendation). **[RelBench v2](relbench-v2.md)** expands to 11 datasets, 29 tables, 22M+ rows, and adds autocomplete tasks and external benchmark integrations.

### v1 Datasets (7)
- `rel-amazon`: Amazon product/user/review data
- `rel-avito`: Avito marketplace ads and users
- `rel-event`: Hangtime social event app
- `rel-f1`: Formula 1 racing records since 1950
- `rel-hm`: H&M customer-product interactions
- `rel-stack`: Stack Exchange Q&A activity
- `rel-trial`: AACT clinical trials database

### v2 New Datasets (4)
- `rel-arxiv`: arXiv physics papers (222K papers, 1.5M citation links, 143K authors)
- `rel-salt`: SAP enterprise ERP sales order data
- `rel-ratebeer`: 20+ years of beer ratings/reviews
- `rel-mimic`: MIMIC-IV ICU electronic health records (requires PhysioNet credentials)

### Task Types
- **Forecasting tasks** (v1+v2): predict future outcomes via SQL-constructed labels — entity classification, entity regression, recommendation.
- **[Autocomplete tasks](autocomplete-tasks.md)** (v2 new): predict missing attribute values in existing columns — binary classification (AUC), multiclass (accuracy), regression (R²).

### External Integrations (v2)
- **TGB**: temporal graph benchmark datasets translated into relational schemas for unified temporal-relational evaluation
- **ReDeLEx**: uniform access to 70+ real-world relational databases for pretraining
- **4DBInfer**: multi-table benchmarking toolbox

The standard RDL pipeline uses HeteroGraphSAGE + temporal-aware neighbor sampling + PyTorch Frame multimodal encoders.

## Appearances in Sources

- [relational-deep-learning-position](relational-deep-learning-position.md) — beta release (2 databases, 4 tasks) accompanying the RDL blueprint paper
- [relbench-v1](relbench-v1.md) — defines v1: 7 databases, 30 tasks, data scientist baseline, open-source implementation
- [relational-graph-transformer](relational-graph-transformer.md) — evaluated on 21 v1 tasks; RelGT outperforms HeteroGNN on 10/21
- [relbench-v2](relbench-v2.md) — v2 expansion paper
- [relational-transformer](relational-transformer.md) — RT pretrained on 6/7 v1 databases (leave-one-out); strong zero-shot on forecasting tasks
- [relgnn](relgnn.md) — evaluated on all 30 v1 tasks; SOTA on 27/30 with up to 25% improvement
- [griffin](griffin.md) — evaluated on RelBench v1 and 4DBInfer; Griffin-RDB-SFT most beneficial in low-data regimes

## Related Concepts

- [relational-deep-learning](relational-deep-learning.md) — the learning paradigm RelBench evaluates
- [relational-entity-graph](relational-entity-graph.md) — graph representation each dataset is converted to
- [autocomplete-tasks](autocomplete-tasks.md) — new task type introduced in v2
- [relational-foundation-model](relational-foundation-model.md) — v2's scale designed to support foundation model pretraining
