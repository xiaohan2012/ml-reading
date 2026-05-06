---
title: "TGB: Temporal Graph Benchmark for Machine Learning on Temporal Graphs"
tags: [source, benchmark, temporal-graph, evaluation]
sources: [huang2023tgb]
updated: 2026-04-29
---

# TGB: Temporal Graph Benchmark for Machine Learning on Temporal Graphs

**Source:** https://arxiv.org/abs/2307.01026
**Title:** Temporal Graph Benchmark for Machine Learning on Temporal Graphs
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Huang, Poursafaei, Danovitch, Fey, Hu, Rossi, Leskovec, Bronstein, Wolf, Rabbany, Gauthier, Rolnick, Derr
**Venue:** NeurIPS 2023
**Year:** 2023

## Summary

Huang et al. (NeurIPS 2023) introduce the **Temporal Graph Benchmark (TGB)**, a large-scale collection of benchmark datasets and evaluation protocols specifically designed for temporal graph learning. TGB addresses three critical limitations in prior temporal graph evaluation: (1) datasets are too small (most <2M edges); (2) negative sampling is unrealistic (1 negative per positive edge); (3) inconsistent evaluation metrics and pipelines.

TGB datasets span years in duration, range from 4M to 67M edges, and cover diverse domains: social networks, international trade, financial transactions, and transportation. Both node-level (dynamic node property prediction) and edge-level (temporal link prediction) tasks are included. The evaluation protocol for link prediction samples hundreds of negative edges per positive, using **temporal negative sampling** — negatives are drawn from historically-observed but currently-absent edges, creating a harder and more realistic setting.

A key empirical finding: **on dynamic node property prediction tasks, simple baselines frequently outperform complex temporal GNN methods** (TGAT, TGN, DyGFormer). This finding directly challenges assumptions in the field and opens questions about when temporal GNN complexity is warranted.

## Key Takeaways

- Datasets: 4M–67M edges (orders of magnitude larger than prior CTDG benchmarks like Wikipedia/Reddit with ~0.1M edges)
- Tasks: temporal link prediction + dynamic node property prediction across 7+ domains
- Evaluation: hundreds of negatives per positive using historical temporal patterns (much harder than random negatives)
- Key finding: simple methods often beat TGAT/TGN/DyGFormer on node prediction — structural complexity ≠ better performance
- Automated pipeline: data loading, experiment setup, leaderboard at tgb.complexdatalab.com
- Referenced by RelBench v2 as an integrated dataset source alongside TGB datasets

## Entities & Concepts

- [temporal-graph](temporal-graph.md) — TGB provides the standard evaluation benchmark for CTDG methods
- [relbench](relbench.md) — RelBench v2 integrates TGB datasets, bridging CTDG and RDL communities

## Relation to Other Wiki Pages

- [xu2020tgat](xu2020tgat.md): TGAT is one of the baselines evaluated on TGB; TGB's findings show TGAT underperforms simple baselines on node prediction
- [rossi2020tgn](rossi2020tgn.md): TGN is another TGB baseline; same finding applies
- [yu2023dygformer](yu2023dygformer.md): DyGFormer is evaluated on TGB; the benchmark's harder negative sampling and larger scale exposes limitations not visible on smaller datasets
- [gu2026relbench](gu2026relbench.md): RelBench v2 explicitly cites TGB integration as part of its dataset expansion strategy
- [chmura2026tgm](chmura2026tgm.md): TGM library supports TGB dataset loading as part of its unified temporal graph interface
