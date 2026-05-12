---
title: "RDL Benchmark Scales"
tags: [query, analysis, relational-deep-learning, benchmark]
sources: [robinson2024relbench, gu2026relbenchv2, huang2023tgb, wang2025griffin, hollmann2025tabpfnv2, qu2025tabicl]
updated: 2026-04-29
---

# RDL Benchmark Scales

## RelBench v1 ([robinson2024relbench](robinson2024relbench.md))

- **7 databases, 30 tasks** across entity classification, regression, and recommendation
- Databases include Amazon (books/products), Stack Exchange, and others
- The wiki page focuses on task count and comparison with data scientists rather than row counts — exact row/node counts per database not captured during ingestion

## RelBench v2 ([gu2026relbenchv2](gu2026relbenchv2.md))

- **11 databases**, 22M+ rows across 29 tables (from 4 new datasets alone)
- **23 autocomplete tasks** (plus v1 forecasting tasks)
- New datasets:
  - **rel-arxiv**: 222K physics papers, 1.5M citation links
  - **rel-salt**: SAP enterprise ERP sales orders
  - **rel-ratebeer**: 20+ years of beer reviews
  - **rel-mimic**: MIMIC-IV ICU EHR — one of the largest public clinical datasets

## Temporal Graph Benchmark ([huang2023tgb](huang2023tgb.md))

- **4M – 67M edges** per dataset — largest temporal graph benchmarks in the field
- Multiple domains: Wikipedia edits, Reddit posts, financial transactions, transport networks
- TGB datasets are now integrated into RelBench v2's relational schemas

## Foundation Model Pretraining Scale

- **Griffin** ([wang2025griffin](wang2025griffin.md)): pretrained on **150M+ nodes** across heterogeneous and temporal graphs

## Tabular Baselines for Context

| Model | Max scale |
|---|---|
| TabPFN v2 ([hollmann2025tabpfnv2](hollmann2025tabpfnv2.md)) | ~10K rows |
| TabICL ([qu2025tabicl](qu2025tabicl.md)) | 500K rows |

## Summary Table

| System | Scale |
|---|---|
| RelBench v1 | 7 DBs, 30 tasks |
| RelBench v2 | 11 DBs, 22M+ rows, 29 tables |
| TGB (integrated into v2) | 4M–67M edges per dataset |
| Griffin pretraining | 150M+ nodes |
| TabPFN v2 | Up to ~10K rows |
| TabICL | Up to 500K rows |

## Gap

Exact row/node counts for individual RelBench v1 databases are not captured in the wiki. Ingesting the RelBench v1 paper's data tables would fill that gap.

## Related Pages

- [relbench](relbench.md) — benchmark overview (v1 + v2)
- [relational-deep-learning](relational-deep-learning.md) — the RDL framework these benchmarks evaluate
- [temporal-graph](temporal-graph.md) — TGB lives here
