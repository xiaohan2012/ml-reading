---
title: "TGM: A Modular and Efficient Library for Machine Learning on Temporal Graphs"
tags: [source, temporal-graph, library, software]
sources: [tgm-temporal-graph-modelling]
updated: 2026-04-28
---

# TGM: A Modular and Efficient Library for Machine Learning on Temporal Graphs

**Source:** https://arxiv.org/abs/2510.07586
**Title:** TGM: A Modular and Efficient Library for Machine Learning on Temporal Graphs
**Date ingested:** 2026-04-28
**Type:** paper (library/systems)
**Authors:** Unknown
**Venue:** Unknown
**Year:** 2025

## Summary

Temporal Graph Modelling (TGM) is a research-oriented open-source library that, for the first time, unifies continuous-time dynamic graph (CTDG) and discrete-time dynamic graph (DTDG) methods under a single framework. Prior libraries (DyGLib, TGL, TGLite) are each siloed around a single paradigm — CTDG or DTDG — and most implement only a narrow family of architectures, blocking fair cross-paradigm comparison. TGM resolves this by treating CTDG and DTDG as different *iteration strategies* over the same underlying event stream: CTDG iterates event-by-event (using event-ordered granularity $\tau_\text{event}$), while DTDG iterates by fixed time windows at a coarser granularity $\hat{\tau} \geq \tau$. Graph discretization — converting a CTDG stream to snapshots — is implemented with a fully vectorized PyTorch-native COO backend, achieving up to 433× speedup over the UTG baseline.

The library's modularity comes from a typed *hook* system: each hook declares what batch attributes it requires and produces, and a dependency graph ensures correct composition order. Hooks cover neighbor sampling, evaluation, GPU transfer, and analytics. This design enables rapid prototyping and unlocks research questions previously impractical to study — most notably, dynamic *graph-level* property prediction (e.g., forecasting whether a transaction network snapshot will grow), which requires time-based iteration and is not supported by any prior library.

Empirically, TGM achieves 7.8× faster end-to-end training than DyGLib across models and datasets, while being the only library to natively support TPNet (SOTA on TGB as of Sep 2025), CTDG transformer models (DyGFormer), and DTDG models (GCLSTM, T-GCN, GCN) together. It also integrates TGB evaluation protocol natively, enabling consistent benchmarking.

## Key Takeaways

- First library to unify CTDG and DTDG as two iteration views of the same event stream — breaks the long-standing ecosystem split.
- Hook mechanism formalizes TG transformations as typed contracts; recipes are valid when their dependency graph is acyclic.
- 7.8× average speedup over DyGLib; 175× average speedup on graph discretization; up to 433× on LastFM.
- Enables dynamic graph-level property prediction as a first-class task — previously impractical without time-based iteration.
- Connects to the RDL ecosystem: TGB datasets are translated into relational schemas in RelBench v2, making TGM relevant to [Relational Deep Learning](relational-deep-learning.md).

## Entities & Concepts

- [temporal-graph](temporal-graph.md)
- [relational-deep-learning](relational-deep-learning.md)
- [relbench-v2](relbench-v2.md)

## Relation to Other Wiki Pages

- [temporal-graph](temporal-graph.md): TGM is the primary software infrastructure for this concept.
- [tgat](tgat.md): TGM natively supports TGAT as one of its CTDG model implementations.
- [tgn](tgn.md): TGM natively supports TGN; TGN's modular memory/embedding design parallels TGM's hook-based architecture.
- [dygformer](dygformer.md): TGM natively supports DyGFormer (CTDG Transformer); DyGLib (DyGFormer's companion library) is a predecessor that TGM supersedes.
- [relbench-v2](relbench-v2.md): TGB datasets are integrated into RelBench v2 via relational schema translation, creating a bridge between TGM's temporal graph domain and the RDL benchmark ecosystem.
