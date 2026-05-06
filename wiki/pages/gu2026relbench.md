---
title: "RelBench v2: A Large-Scale Benchmark and Repository for Relational Data"
tags: [source, benchmark, relational-deep-learning]
sources: [gu2026relbench]
updated: 2026-05-06
---

# RelBench v2: A Large-Scale Benchmark and Repository for Relational Data

**Source:** https://arxiv.org/abs/2602.12606
**Title:** RelBench v2: A Large-Scale Benchmark and Repository for Relational Data
**Date ingested:** 2026-04-28
**Type:** paper
**Authors:** Justin Gu, Rishabh Ranjan, Charilaos Kanatsoulis, Haiming Tang, Martin Jurkovic, Valter Hudovernik, Mark Znidar, Pranshu Chaturvedi, Parth Shroff, Fengyu Li, Jure Leskovec
**Venue:** ICLR 2026

## Summary

- **What:** RelBench v1's 7 datasets and forecasting-only task type were insufficient to benchmark foundation models that need scale, domain diversity, and non-forecasting tasks.
- **How:** Adds 4 new real-world datasets (22M+ new rows), 23 autocomplete tasks (predict existing column values rather than future events), and integrates TGB/ReDeLEx/4DBInfer as external data sources.
- **So what:** 11 datasets, 29 tables, broader task coverage; positioned as an ecosystem hub for relational ML research and foundation model pretraining.

## Challenges & Novelty

RelBench v1 covered only 7 databases and one task paradigm (forecasting: predict a future value from historical events). As relational foundation models scale up, two gaps emerge: (1) more diverse pretraining data is needed, and (2) the autocomplete use case (predicting missing attribute values in existing records — the dominant enterprise ML use case) has no benchmark representation.

- **Autocomplete as a task type:** unlike forecasting tasks that predict a future outcome, autocomplete predicts an existing but masked/missing column value at inference time (e.g., SAP S/4HANA sales order completion). Autocomplete has no intrinsic temporal structure but still benefits from relational context — RDL outperforms tabular baselines because the signal spans multiple tables.
- **Scale gap:** v1 datasets had at most ~3M rows; v2 adds rel-arxiv (1.5M papers + citation network), rel-mimic (MIMIC-IV ICU records), rel-salt (enterprise ERP), and rel-ratebeer (20+ years of reviews) — pushing the benchmark into the range where foundation model pretraining is meaningful.
- **Ecosystem integration:** rather than collecting new data from scratch, v2 reuses existing datasets (TGB temporal interactions, ReDeLEx's 70+ databases) by translating them into the relational schema format, enabling cross-community benchmarking.

## Relation to Prior Work

| Benchmark | Datasets | Task types | Temporal leakage prevention | Foundation model pretraining data |
|---|---|---|---|---|
| OGB | 9 | Classification/regression on static graphs | N/A | No |
| TGB ([huang2023tgb](huang2023tgb.md)) | 7 | Temporal link pred + node pred | Yes | No |
| RelBench v1 ([robinson2024relbench](robinson2024relbench.md)) | 7 | Forecasting | Yes | No |
| **RelBench v2** | 11 | Forecasting + autocomplete | Yes | Yes (ReDeLEx) |

- [robinson2024relbench](robinson2024relbench.md): v2 is the direct successor; all v1 tasks remain; v2 adds autocomplete and dataset diversity.
- [huang2023tgb](huang2023tgb.md): TGB datasets are integrated into v2 via relational schema translation, bridging temporal graph and RDL evaluation.
- [fey2025kumorfm2](fey2025kumorfm2.md): KumoRFM-2 uses ReDeLEx (integrated via v2) for pretraining; v2's scale is a direct enabler for foundation model research.

## Technical Details

**New datasets:**
- `rel-arxiv`: 222K physics papers + 1.5M citation links; tasks include paper acceptance prediction, citation count forecasting
- `rel-salt`: SAP enterprise sales order ERP data; autocomplete tasks for sales order field prediction
- `rel-ratebeer`: 20+ years of beer reviews; user preference forecasting and review attribute autocomplete
- `rel-mimic`: MIMIC-IV ICU EHR data; clinical outcome prediction and diagnosis code autocomplete

**Autocomplete task construction.** A row's target column is masked at a seed time $t_v$; the model predicts the column value using only relational context from rows with $\tau \leq t_v$. Unlike forecasting, the label comes from the row itself (not a future aggregation), so no time-window SQL is needed — but temporal leakage prevention still applies to neighbor sampling.

![Autocomplete task: mask a cell value, predict from relational context at seed time](assets/gu2026relbench-autocomplete.png)

**External integrations:**
- *TGB*: temporal interaction datasets translated to relational schema — each interaction becomes a row in an events table with FK references to node tables
- *ReDeLEx*: uniform API to 70+ real-world relational databases; used as pretraining corpus for foundation models
- *4DBInfer*: multi-table benchmarking tools with additional evaluation utilities

**Evaluation metrics.** Forecasting: AUROC (classification), MAE/RMSE (regression). Autocomplete: AUC (binary), accuracy (multiclass), R² (regression).

## Experiments

- HeteroGraphSAGE outperforms LightGBM on all autocomplete task types, confirming relational structure is informative for attribute inference beyond what a single-table model can capture.
- RDL gains are largest on tasks where the target column is correlated with information in linked tables (not visible in the target row alone).

## Entities & Concepts

- [relbench](relbench.md)
- [autocomplete-tasks](autocomplete-tasks.md)
- [relational-deep-learning](relational-deep-learning.md)
- [temporal-graph](temporal-graph.md)
- [relational-foundation-model](relational-foundation-model.md)
