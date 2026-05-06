---
title: "RelBench v2: A Large-Scale Benchmark and Repository for Relational Data"
tags: [source, benchmark, relational-deep-learning]
sources: [gu2026relbench]
updated: 2026-04-28
---

# RelBench v2: A Large-Scale Benchmark and Repository for Relational Data

**Source:** https://arxiv.org/abs/2602.12606
**Title:** RelBench v2: A Large-Scale Benchmark and Repository for Relational Data
**Date ingested:** 2026-04-28
**Type:** paper
**Authors:** Justin Gu, Rishabh Ranjan, Charilaos Kanatsoulis, Haiming Tang, Martin Jurkovic, Valter Hudovernik, Mark Znidar, Pranshu Chaturvedi, Parth Shroff, Fengyu Li, Jure Leskovec
**Venue:** ICLR 2026
**Year:** 2026

## Summary

RelBench v2 is a major expansion of the RelBench benchmark for Relational Deep Learning, growing from 7 to 11 datasets and introducing a new category of predictive tasks. The four new datasets span scholarly publications (rel-arxiv: 222K physics papers, 1.5M citation links), enterprise ERP (rel-salt: SAP sales order data), consumer platforms (rel-ratebeer: 20+ years of beer reviews), and clinical records (rel-mimic: MIMIC-IV ICU EHR data). Together these add over 22 million rows across 29 tables, significantly increasing the scale and domain diversity of the benchmark.

The most conceptually significant addition is **autocomplete tasks**: predicting missing or masked attribute values in *existing* relational table columns, as opposed to forecasting tasks which predict future outcomes constructed via SQL queries over future rows. Autocomplete is inherently temporal (a seed time prevents future-data leakage) and requires models to understand relational context rather than just temporal trends — directly inspired by SAP's S/4HANA sales order autocomplete UI. RelBench v2 introduces 23 autocomplete tasks across both new and existing datasets, using AUC (binary classification), accuracy (multiclass), or R² (regression) as metrics. RDL (HeteroGraphSAGE) consistently outperforms LightGBM on autocomplete tasks, validating that relational structure is informative.

In addition to new native datasets and tasks, RelBench v2 integrates three external frameworks: TGB temporal interaction datasets (translated into relational schemas for unified temporal-relational evaluation), ReDeLEx (uniform access to 70+ real-world relational databases for pretraining), and 4DBInfer (multi-table benchmarking tools). This positions RelBench v2 not just as a task benchmark but as a broader ecosystem hub for relational machine learning research.

## Key Takeaways

- **4 new datasets**: rel-arxiv, rel-salt, rel-ratebeer, rel-mimic — bringing total to 11 datasets, 29 tables, 22M+ rows.
- **Autocomplete tasks** (23 new): predict existing column values at a seed time — a new paradigm that demands relational context understanding, not just forecasting.
- **External integrations**: TGB (temporal ↔ relational bridge), ReDeLEx (70+ pretraining databases), 4DBInfer (multi-table tooling).
- **RDL vs. baselines**: HeteroGraphSAGE beats LightGBM on all autocomplete task types, confirming relational structure matters for attribute inference.
- **Directly enables foundation model research**: scale + diversity of v2, combined with ReDeLEx pretraining data, are motivated by the trend toward relational foundation models (Griffin, RT, RelGT).

## Entities & Concepts

- [relbench](relbench.md)
- [autocomplete-tasks](autocomplete-tasks.md)
- [relational-deep-learning](relational-deep-learning.md)
- [temporal-graph](temporal-graph.md)
- [relational-foundation-model](relational-foundation-model.md)

## Relation to Other Wiki Pages

- [relbench](relbench.md): this paper is the v2 expansion; relbench.md updated to reflect new datasets and task types.
- [ranjan2025relationaltr](ranjan2025relationaltr.md): RT uses RelBench for pretraining; references v2 autocomplete tasks.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT was evaluated on RelBench v1; v2 is the expanded successor benchmark.
- [chmura2026tgm](chmura2026tgm.md): TGB datasets integrated into v2 via relational schema translation.
