---
title: "KumoRFM-2 Reading Plan: Predecessor Papers and Techniques"
tags: [query, analysis, relational-deep-learning, foundation-model, in-context-learning]
sources: [fey2025kumorfm2, fey2025kumorfm, fey2024rdlposition, robinson2024relbench, muller2022pfn, hollmann2023tabpfnv1, hollmann2025tabpfnv2, qu2025tabicl, rampavsek2022graphgps, dwivedi2025relgt]
updated: 2026-05-08
---

# KumoRFM-2 Reading Plan: Predecessor Papers and Techniques

KumoRFM-2 sits at the intersection of three lines of work. You need all three.

## Thread 1 — Relational Deep Learning (the problem setting)

**Start here to understand what KumoRFM-2 is even trying to solve.**

1. [fey2024rdlposition](fey2024rdlposition.md) — The RDL blueprint. Defines the Relational Entity Graph (REG): model a relational database as a heterogeneous temporal graph where rows = nodes, PK-FK links = edges. Every RDL/RFM paper assumes this framing.
2. [robinson2024relbench](robinson2024relbench.md) — RelBench v1: the benchmark all methods are evaluated on. Understand what a "task table" is and why temporal leakage is the key engineering constraint.

## Thread 2 — Tabular In-Context Learning (the ICL paradigm)

**KumoRFM-2's most distinctive idea — injecting labels as context for a frozen model — comes directly from tabular ICL.**

3. [muller2022pfn](muller2022pfn.md) — PFN (Prior-Data Fitted Networks): the theoretical foundation. Proves that training a Transformer on samples from a prior minimizes KL to the exact Bayesian PPD. Establishes the core idea that *any* Bayesian model can be amortized into a single forward pass given a sampable prior. Read this before TabPFN to understand *why* the ICL paradigm works, not just that it does.
4. [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md) — TabPFN v1: PFNs applied to tabular classification. A prior-fitted network trained on SCM+BNN synthetic data predicts any table in a single forward pass by treating training rows as context. This is the paradigm KumoRFM-2 extends to multi-table relational data.
5. [hollmann2025tabpfnv2](hollmann2025tabpfnv2.md) — TabPFN v2: adds alternating row/column attention within a table — this is exactly Stage 1 of KumoRFM-2's hierarchical design (column attention + row attention within a table, before FK/cross-sample attention across tables).
6. [qu2025tabicl](qu2025tabicl.md) — TabICL: 3-Transformer stack (column → row → ICL Transformer). KumoRFM-2 is essentially TabICL + a relational graph layer inserted between Stage 1 (intra-table) and Stage 2 (inter-table/cross-sample).

## Thread 3 — Graph Transformers for relational structure (the graph layer)

**KumoRFM-2's FK-level and cross-sample attention over the relational entity graph builds on GT work.**

7. [rampavsek2022graphgps](rampavsek2022graphgps.md) — GraphGPS: the architectural backbone. MPNN ∥ GlobalAttn in parallel; PE/SE taxonomy. KumoRFM-1 used RelGT (which is built on GPS) as its table-wise interaction module.
8. [dwivedi2025relgt](dwivedi2025relgt.md) — RelGT: the direct supervised predecessor to KumoRFM's relational GT module. 5-element tokenization (features, type, hop, time, subgraph GNN PE); hybrid local+global attention on the REG. KumoRFM-1 used RelGT explicitly as Stage 2; KumoRFM-2 replaces it with hierarchical 4-axis attention but the concepts carry over.
9. [fey2025kumorfm](fey2025kumorfm.md) — KumoRFM-1: the direct predecessor. Three stages: table-invariant encoder → RelGT → ICL module. V2 flattens this into a single hierarchical 4-axis attention model with early label injection.

## How Threads 2 and 3 Relate

They solve different sub-problems that KumoRFM-2 stacks into one pipeline:

- **Thread 2 (Tabular ICL)** handles what's *inside* a single table — column + row attention produce a row embedding from raw cell values. This is a data-representation problem: how to turn a heterogeneous, variable-width row into a fixed-size vector.
- **Thread 3 (Graph Transformers)** handles what's *across* tables — FK attention + cross-sample attention aggregate those row embeddings over the relational graph. This is a relational-structure problem: how to propagate information across PK-FK links.

KumoRFM-2 sequences them: Thread 2 runs first (lightweight intra-table network → row embeddings), then Thread 3 runs on those embeddings (larger inter-table network → final predictions).

The seam between them is exactly where single-table tabular models break down. TabPFN/TabICL stop after Thread 2 — they have no FK-attention layer because they only see one table. Prior Graph Transformers (GPS, RelGT) skip Thread 2 — they assume you already have a node feature vector from some upstream encoder. KumoRFM-2 is the first model to learn both jointly end-to-end, with task labels injected before Thread 2 so both stages are task-conditioned.

## Shortest Critical Path (4 papers)

| Order | Paper | Why |
|---|---|---|
| 1 | [fey2024rdlposition](fey2024rdlposition.md) | Understand REGs and the RDL task formulation |
| 2 | [muller2022pfn](muller2022pfn.md) | Understand *why* PFN training = Bayesian inference (the theory) |
| 2b | [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md) | Understand ICL applied to tabular data (the practice) |
| 3 | [dwivedi2025relgt](dwivedi2025relgt.md) | Understand how GTs handle heterogeneous temporal relational graphs |
| 4 | [fey2025kumorfm](fey2025kumorfm.md) | Understand V1; V2 is then a legible architectural evolution |

## Gap

The wiki has no page on the synthetic data generation techniques (Structural Causal Models, PluRel) that KumoRFM-2 uses for pretraining — needed to understand *how it was trained*, not just the architecture.
