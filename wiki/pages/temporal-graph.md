---
title: Temporal Graph
tags: [concept, temporal-graph]
sources: [relational-deep-learning-position, tgm-temporal-graph-modelling, relbench-v2, tgat, tgn, dygformer, tgb, dyrep, graphmixer]
updated: 2026-04-29
---

# Temporal Graph

## Description

A temporal graph is a sequence of time-ordered events $\mathcal{G} = \{e_0, \ldots, e_T\}$, where each event is either an **edge event** (interaction between two nodes at time $t$) or a **node event** (arrival of new features at a node at time $t$). This formulation unifies the two main research paradigms:

- **CTDG (Continuous-Time Dynamic Graphs)**: iterate event-by-event; each batch contains a fixed number of events regardless of elapsed real time. Models: TGAT, TGN, DyGFormer, TPNet.
- **DTDG (Discrete-Time Dynamic Graphs)**: iterate by fixed time windows at a coarser granularity $\hat{\tau}$, producing graph snapshots. Models: GCN, T-GCN, GCLSTM.

Both are views of the same underlying event stream. [TGM](tgm-temporal-graph-modelling.md) is the first library to unify them.

**Temporal graph learning tasks**: link property prediction, node property prediction, and graph property prediction (predicting global properties of future graph snapshots — enabled by TGM's time-based iteration).

**Connection to Relational Deep Learning**: [Relational Entity Graphs](relational-entity-graph.md) are a special class of temporal heterogeneous graph defined by primary-foreign key schema constraints. [RelBench v2](relbench-v2.md) integrates TGB temporal graph datasets into relational schemas, creating a unified relational-temporal evaluation framework.

## Appearances in Sources

- [relational-deep-learning-position](relational-deep-learning-position.md) — time-consistent computation graph (Algorithm 1): temporal neighbor sampling filters to $\tau(w) \leq t$; three strategies (uniform, ordered, biased); applied to heterogeneous REGs
- [tgm-temporal-graph-modelling](tgm-temporal-graph-modelling.md) — defines the formal model; TGM is the software infrastructure
- [relbench-v2](relbench-v2.md) — TGB temporal interaction datasets translated into relational schemas for unified evaluation
- [tgat](tgat.md) — TGAT: first transformer-based CTDG encoder; Bochner functional time encoding; inductive single forward pass; ICLR 2020
- [tgn](tgn.md) — TGN: memory + temporal graph attention; unifies TGAT/Jodie/DyRep; 30× faster; SOTA 2020
- [dygformer](dygformer.md) — DyGFormer: first-hop Transformer with patching + neighbor co-occurrence; eliminates RNN/multi-hop; NeurIPS 2023 SOTA
- [tgb](tgb.md) — Temporal Graph Benchmark: large-scale (4M-67M edges), temporal negative sampling, diverse domains; finds simple baselines often beat TGAT/TGN/DyGFormer on node prediction
- [dyrep](dyrep.md) — DyRep (ICLR 2019): two-process framework (Association/Communication); recurrent + structural + temporal embedding update; intensity-based attention via stochastic matrix S; temporal point process model
- [graphmixer](graphmixer.md) — GraphMixer (ICLR 2023): MLP link-encoder + mean-pooling node-encoder + MLP classifier; no RNN/attention; matches or beats TGAT/TGN/DyGFormer with 2–5× faster training

## Related Concepts

- [relational-entity-graph](relational-entity-graph.md) — special case: temporal heterogeneous graph with schema-defined structure
- [tgm-temporal-graph-modelling](tgm-temporal-graph-modelling.md) — the library implementing temporal graph ML
