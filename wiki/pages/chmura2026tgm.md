---
title: "TGM: A Modular and Efficient Library for Machine Learning on Temporal Graphs"
tags: [source, temporal-graph, library, software]
sources: [chmura2026tgm]
updated: 2026-05-06
---

# TGM: A Modular and Efficient Library for Machine Learning on Temporal Graphs

**Source:** https://arxiv.org/abs/2510.07586
**Title:** TGM: A Modular and Efficient Library for Machine Learning on Temporal Graphs
**Date ingested:** 2026-04-28
**Type:** paper (library/systems)
**Authors:** Jacob Chmura, Shenyang Huang, Tran Gia Bao Ngo, Ali Parviz, Farimah Poursafaei, Jure Leskovec, Michael Bronstein, Guillaume Rabusseau, Matthias Fey, Reihaneh Rabbany
**Venue:** ICLR 2026

## Summary

- **What:** Prior temporal graph libraries (DyGLib, TGL, TGLite) are siloed around either CTDG or DTDG, blocking cross-paradigm comparison and making graph-level tasks impractical.
- **How:** TGM unifies CTDG and DTDG as two iteration strategies over the same event stream, with a typed hook system for composable transformations and a fully vectorized PyTorch-native graph discretization backend.
- **So what:** 7.8× faster end-to-end training than DyGLib; up to 433× faster graph discretization; first library to natively support CTDG Transformers, DTDG models, and TGB evaluation together.

## Challenges & Novelty

The temporal graph ecosystem is fragmented: DyGLib supports CTDG methods (TGAT, TGN, DyGFormer), while T-GCN and GCLSTM are DTDG-only and have no standard library. This split prevents direct comparison and makes hybrid research (CTDG methods on DTDG tasks, or vice versa) impractical. Additionally, dynamic *graph-level* property prediction requires time-based iteration that no prior CTDG library supports.

- **CTDG and DTDG are iteration strategies, not fundamentally different paradigms:** CTDG iterates event-by-event at granularity $\tau_\text{event}$; DTDG iterates at coarser time windows $\hat{\tau} \geq \tau$. Both views operate on the same underlying event stream — TGM exploits this to unify them.
- **Hook system formalizes composition:** each hook declares required and produced batch attributes; a dependency graph ensures correct ordering. Invalid recipes (missing attributes) are caught at initialization rather than runtime.
- **Graph discretization bottleneck:** converting event streams to snapshots (for DTDG) was previously slow (CPU-only, loop-based); TGM's PyTorch-native COO backend achieves 175× average and up to 433× speedup (LastFM dataset).

## Relation to Prior Work

| Library | CTDG | DTDG | Graph-level tasks | TGB eval | Speedup vs. DyGLib |
|---|---|---|---|---|---|
| DyGLib | Yes | No | No | Partial | baseline |
| TGL | Yes | No | No | No | — |
| TGLite | Partial | No | No | No | — |
| **TGM** | Yes | Yes | Yes | Yes | 7.8× |

- [yu2023dygformer](yu2023dygformer.md): DyGLib was developed alongside DyGFormer; TGM supersedes DyGLib with broader model coverage and faster execution.
- [rossi2020tgn](rossi2020tgn.md): TGN's modular memory/embedding design (5 composable components) inspired TGM's hook-based modularity.
- [gu2026relbenchv2](gu2026relbenchv2.md): TGB datasets are integrated into RelBench v2 via relational schema translation, creating a bridge between TGM and the RDL ecosystem.

## Technical Details

**Unified event stream.** All temporal graph data is stored as a sorted event list $(u_i, v_i, t_i, \mathbf{e}_i)$. Iteration mode selects the granularity:
- CTDG: one event per step
- DTDG: all events in $[t_\text{start}, t_\text{start} + \hat{\tau})$ per step

**Hook system.** A hook is a callable `(batch: Batch) -> Batch` with declared `requires: Set[str]` and `produces: Set[str]`. The TGM runtime builds a DAG from all registered hooks; if the DAG contains a cycle or a required attribute has no producer, initialization fails. Built-in hooks cover neighbor sampling (most-recent, uniform), GPU transfer, TGB evaluation, and analytics.

**Graph discretization.** Events within a window are aggregated into a COO sparse matrix using `torch.scatter_add`. No Python loops — the full discretization is a vectorized PyTorch operation. Supports weighted edges (sum/mean/last), self-loops, and directed/undirected modes.

**Supported models.** CTDG: TGAT, TGN, DyGFormer, TPNet. DTDG: GCLSTM, T-GCN, GCN on snapshots. All share the same training loop — only the iteration strategy and model class differ.

## Experiments

- 7.8× faster training than DyGLib averaged across models and datasets.
- 175× average and up to 433× (LastFM) speedup on graph discretization vs. UTG baseline.
- Only library to natively run TPNet (TGB SOTA as of Sep 2025) alongside CTDG Transformers and DTDG models.
- Dynamic graph-level property prediction (forecasting network-level statistics) demonstrated as a new task class enabled by time-based iteration.

## Entities & Concepts

- [temporal-graph](temporal-graph.md)
- [relational-deep-learning](relational-deep-learning.md)
