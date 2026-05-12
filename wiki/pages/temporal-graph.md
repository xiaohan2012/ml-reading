---
title: Temporal Graph
tags: [concept, temporal-graph]
sources: [fey2024rdlposition, chmura2026tgm, gu2026relbenchv2, xu2020tgat, rossi2020tgn, yu2023dygformer, huang2023tgb, trivedi2019dyrep, cong2023graphmixer]
updated: 2026-04-29
---

# Temporal Graph

## Description

A temporal graph is a sequence of time-ordered events $\mathcal{G} = \{e_0, \ldots, e_T\}$, where each event is either an **edge event** (interaction between two nodes at time $t$) or a **node event** (arrival of new features at a node at time $t$). This formulation unifies the two main research paradigms:

- **CTDG (Continuous-Time Dynamic Graphs)**: iterate event-by-event; each batch contains a fixed number of events regardless of elapsed real time. Models: TGAT, TGN, DyGFormer, TPNet.
- **DTDG (Discrete-Time Dynamic Graphs)**: iterate by fixed time windows at a coarser granularity $\hat{\tau}$, producing graph snapshots. Models: GCN, T-GCN, GCLSTM.

Both are views of the same underlying event stream. [TGM](chmura2026tgm.md) is the first library to unify them.

**Temporal graph learning tasks**: link property prediction, node property prediction, and graph property prediction (predicting global properties of future graph snapshots — enabled by TGM's time-based iteration).

**Connection to Relational Deep Learning**: [Relational Entity Graphs](relational-entity-graph.md) are a special class of temporal heterogeneous graph defined by primary-foreign key schema constraints. [RelBench v2](gu2026relbenchv2.md) integrates TGB temporal graph datasets into relational schemas, creating a unified relational-temporal evaluation framework.

## Appearances in Sources

- [fey2024rdlposition](fey2024rdlposition.md) — time-consistent computation graph (Algorithm 1): temporal neighbor sampling filters to $\tau(w) \leq t$; three strategies (uniform, ordered, biased); applied to heterogeneous REGs
- [chmura2026tgm](chmura2026tgm.md) — defines the formal model; TGM is the software infrastructure
- [gu2026relbenchv2](gu2026relbenchv2.md) — TGB temporal interaction datasets translated into relational schemas for unified evaluation
- [xu2020tgat](xu2020tgat.md) — TGAT: first transformer-based CTDG encoder; Bochner functional time encoding; inductive single forward pass; ICLR 2020
- [rossi2020tgn](rossi2020tgn.md) — TGN: memory + temporal graph attention; unifies TGAT/Jodie/DyRep; 30× faster; SOTA 2020
- [yu2023dygformer](yu2023dygformer.md) — DyGFormer: first-hop Transformer with patching + neighbor co-occurrence; eliminates RNN/multi-hop; NeurIPS 2023 SOTA
- [huang2023tgb](huang2023tgb.md) — Temporal Graph Benchmark: large-scale (4M-67M edges), temporal negative sampling, diverse domains; finds simple baselines often beat TGAT/TGN/DyGFormer on node prediction
- [trivedi2019dyrep](trivedi2019dyrep.md) — DyRep (ICLR 2019): two-process framework (Association/Communication); recurrent + structural + temporal embedding update; intensity-based attention via stochastic matrix S; temporal point process model
- [cong2023graphmixer](cong2023graphmixer.md) — GraphMixer (ICLR 2023): MLP link-encoder + mean-pooling node-encoder + MLP classifier; no RNN/attention; matches or beats TGAT/TGN/DyGFormer with 2–5× faster training

## Related Concepts

- [relational-entity-graph](relational-entity-graph.md) — special case: temporal heterogeneous graph with schema-defined structure
- [chmura2026tgm](chmura2026tgm.md) — the library implementing temporal graph ML
