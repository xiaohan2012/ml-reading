---
title: "TGB: Temporal Graph Benchmark for Machine Learning on Temporal Graphs"
tags: [source, benchmark, temporal-graph, evaluation]
sources: [huang2023tgb]
updated: 2026-05-06
---

# TGB: Temporal Graph Benchmark for Machine Learning on Temporal Graphs

**Source:** https://arxiv.org/abs/2307.01026
**Title:** Temporal Graph Benchmark for Machine Learning on Temporal Graphs
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Huang, Poursafaei, Danovitch, Fey, Hu, Rossi, Leskovec, Bronstein, Wolf, Rabbany, Gauthier, Rolnick, Derr
**Venue:** NeurIPS 2023

## Summary

- **What:** Prior temporal graph benchmarks (Wikipedia, Reddit) are too small (<2M edges), use unrealistic random negative sampling, and have inconsistent evaluation pipelines — making published results incomparable.
- **How:** 7 datasets spanning 4M–67M edges with temporal negative sampling (hundreds of negatives per positive drawn from historically-observed edges), two task types (link prediction and node property prediction), and a standardized automated pipeline.
- **So what:** Reveals that simple baselines frequently outperform TGAT, TGN, and DyGFormer on node property prediction tasks, challenging the field's assumptions about when temporal GNN complexity is warranted.

## Challenges & Novelty

The dominant CTDG benchmarks at NeurIPS 2023 were Wikipedia and Reddit datasets with ~0.1M edges. Results on these small datasets don't generalize — models that overfit on small graphs may saturate on larger ones, and negative sampling strategies that were unrealistic (1 random negative per positive) made edge prediction appear too easy.

- **Scale gap:** 4M–67M edges across 7 datasets in diverse domains (social networks, trade, finance, transportation) — 40–700× larger than Wikipedia/Reddit.
- **Temporal negative sampling is harder and more realistic:** negatives are sampled from historically-observed but currently-absent edges, forcing models to distinguish between active and dormant relationships rather than arbitrary node pairs.
- **Simple baselines often win on node prediction:** a finding that directly motivates rethinking when complex temporal GNN architectures add value — the complexity may be warranted for link prediction but not for node property prediction where label smoothness matters more.

## Relation to Prior Work

| Benchmark | Edges | Neg. sampling | Tasks | Cross-domain |
|---|---|---|---|---|
| Wikipedia/Reddit | ~0.1M | 1 random | Link pred | No |
| OGB-LSC | Up to 86M | N/A | Static | Yes |
| **TGB** | 4M–67M | Hundreds (temporal) | Link + node | Yes |
| [robinson2024relbench](robinson2024relbench.md) (RelBench v1) | — | — | Forecasting | Yes |

- [xu2020tgat](xu2020tgat.md): TGAT is a baseline on TGB; TGB's node prediction results show it underperforms simple methods on non-link tasks.
- [rossi2020tgn](rossi2020tgn.md): TGN is another TGB baseline with the same pattern — strong on link prediction, not dominant on node prediction.
- [yu2023dygformer](yu2023dygformer.md): DyGFormer is evaluated on TGB; harder negative sampling and larger scale expose limitations invisible on Wikipedia/Reddit.
- [gu2026relbenchv2](gu2026relbenchv2.md): RelBench v2 integrates TGB datasets via relational schema translation, enabling unified CTDG + RDL evaluation.
- [chmura2026tgm](chmura2026tgm.md): TGM implements TGB evaluation protocol natively and supports TGB datasets as first-class citizens.

## Technical Details

**Datasets.** 7 datasets across domains: social (tgbl-wiki, tgbn-reddit), trade (tgbl-trade, tgbn-trade), financial (tgbl-coin), transportation (tgbl-flight), and review (tgbn-genre). Sizes: 4M–67M edges, months to years in duration.

**Evaluation for temporal link prediction.** For each positive edge $(u, v, t)$, sample $K$ negative edges $(u, v_i^-, t)$ where $v_i^-$ are historically-observed destination nodes that $u$ has not interacted with at time $t$. Rank the positive edge among all $K+1$ candidates; report Mean Reciprocal Rank (MRR) or Hits@K.

**Evaluation for dynamic node property prediction.** Predict a node attribute (e.g., tag of a post, sector of a trade flow) from its temporal neighborhood at each time step. Metric: accuracy or F1.

**Automated pipeline.** Data loading, train/val/test split (chronological), negative sampler, and leaderboard submission available via the `tgb` Python package.

## Experiments

- Simple baselines (e.g., persistent prediction, degree-based heuristics) outperform TGAT and TGN on node property prediction tasks — structured temporal complexity is not universally beneficial.
- On link prediction, TGAT/TGN/DyGFormer retain advantages over simple baselines under temporal negative sampling, confirming the link prediction use case.
- Performance gaps between methods increase substantially under temporal vs. random negative sampling, validating that random NS is an insufficient discriminator.

## Entities & Concepts

- [temporal-graph](temporal-graph.md)
- [relbench](relbench.md)
