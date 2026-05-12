---
title: "Open-Source RDL Software"
tags: [query, analysis, relational-deep-learning, software]
sources: [robinson2024relbench, gu2026relbenchv2, chmura2026tgm, fey2024rdlposition, fey2025kumorfm2]
updated: 2026-04-29
---

# Open-Source RDL Software

## 1. RelBench — Core Research Scaffold

**`pip install relbench`** | [robinson2024relbench](robinson2024relbench.md), [gu2026relbenchv2](gu2026relbenchv2.md)

Built on PyTorch Frame (multi-modal tabular encoding) + PyTorch Geometric (GNN training). Provides:
- Database loading (7 → 11 real-world DBs across e-commerce, social, medical, academic, enterprise domains)
- Automatic training table construction — generates `(EntityID, seed_time, label)` triples from the DB via SQL, with strict temporal leakage prevention
- Temporal splits with shared `val_timestamp`/`test_timestamp` across tasks (enabling multi-task learning experiments)
- Standardized evaluation: AUROC (classification), MAE (regression), MAP@k (recommendation)
- Public leaderboard

Data scientist baseline notebooks (878 LoC per task) available at `github.com/snap-stanford/relbench-user-study` — useful as a strong feature-engineering reference point.

## 2. TGM — Temporal Graph Library

**Open-source** | [chmura2026tgm](chmura2026tgm.md)

The only library unifying CTDG and DTDG under a single framework. Treats both paradigms as different iteration strategies over the same event stream. Supports:
- CTDG models: TGAT, TGN, DyGFormer, TPNet (SOTA on TGB as of Sep 2025)
- DTDG models: GCLSTM, T-GCN, GCN
- TGB evaluation protocol natively
- Dynamic graph-level property prediction (not supported by any prior library)

7.8× faster end-to-end training than DyGLib; up to 433× faster on graph discretization. Hook-based modular architecture for rapid prototyping. TGB datasets are integrated into RelBench v2, making TGM directly relevant to RDL pipelines.

## 3. PyTorch Frame — Multi-Modal Tabular Encoding

Referenced by [fey2024rdlposition](fey2024rdlposition.md) and [robinson2024relbench](robinson2024relbench.md) as the node encoder backbone. Handles: numerical, categorical, multi-categorical, text (sentence-BERT), image, timestamp, and embedding columns — the layer that converts heterogeneous table rows into initial node embeddings for downstream GNNs.

## 4. PyTorch Geometric (PyG) — GNN Training

The underlying GNN framework all RDL papers build on. RelGNN, HeteroGraphSAGE, HGT, and RelGT are implemented via PyG's heterogeneous graph APIs (`HeteroData`, `to_hetero`, relation-typed message passing).

## 5. KumoRFM Python SDK

**`kumorfm.ai`** (commercial, publicly accessible) | [fey2025kumorfm](fey2025kumorfm.md), [fey2025kumorfm2](fey2025kumorfm2.md)

End-to-end RFM system with SQL database connectors and a memory-mapped graph engine for billion-scale datasets. Benchmark scripts at `github.com/kumo-ai/kumo-rfm`. LLM agent integration skills at `github.com/kumo-ai/kumo-coding-agent`.

## Summary Table

| Tool | Role | Key Feature | Source |
|---|---|---|---|
| RelBench | Benchmark + data pipeline | 11 DBs, 30+ tasks, temporal splits, leaderboard | [robinson2024relbench](robinson2024relbench.md), [gu2026relbenchv2](gu2026relbenchv2.md) |
| TGM | Temporal graph training | CTDG+DTDG unified, 7.8× faster than DyGLib | [chmura2026tgm](chmura2026tgm.md) |
| PyTorch Frame | Multi-modal node encoding | Handles text/image/numerical/categorical cells | [fey2024rdlposition](fey2024rdlposition.md) |
| PyTorch Geometric | GNN backbone | HeteroData, relation-typed message passing | [fey2024rdlposition](fey2024rdlposition.md) |
| KumoRFM SDK | End-to-end RFM system | PQL interface, SQL connectors, billion-scale | [fey2025kumorfm2](fey2025kumorfm2.md) |

## Gaps Not Covered by Wiki

- **DyGLib** — predecessor temporal graph library (superseded by TGM, but still referenced in benchmarks)
- **4DBInfer** — benchmark package for the additional datasets used in KumoRFM-2 experiments
- **PyTorch Geometric Temporal** — DTDG models (T-GCN, GCLSTM) before TGM
- Standalone code releases for RelGNN or RelGT (papers don't explicitly mention public repos)

## Related Pages

- [relational-deep-learning](relational-deep-learning.md) — the RDL paradigm all tools implement
- [relbench](relbench.md) — the benchmark all tools integrate with
- [temporal-graph](temporal-graph.md) — TGM's domain
- [rdl-approaches](rdl-approaches.md) — the models these tools support
