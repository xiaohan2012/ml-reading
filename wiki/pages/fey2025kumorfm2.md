---
title: "KumoRFM-2: Scaling Foundation Models for Relational Learning"
tags: [source, relational-deep-learning, foundation-model, in-context-learning, graph-transformer]
sources: [fey2025kumorfm2]
updated: 2026-05-06
---

# KumoRFM-2: Scaling Foundation Models for Relational Learning

**Source:** https://arxiv.org/abs/2604.12596
**Title:** KumoRFM-2: Scaling Foundation Models for Relational Learning
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Matthias Fey, Vida Kocijan, Nhat Hoang, Jonathan Lehmann, Jan E. Lenssen, Jure Leskovec
**Venue:** NeurIPS 2025 (preprint)

## Summary

- **What:** Tabular FMs applied to relational data via fixed-function flattening (DFS-style column-wise aggregation) suffer a fundamental expressivity gap — they cannot capture row-level conjunctions — and KumoRFM-1 was limited to in-memory datasets.
- **How:** KumoRFM-2 uses a hierarchical 4-axis attention scheme (column+row within tables, FK+cross-sample across tables) with early task-label injection, pretrained on synthetic and real relational data, and scales to 500B+ rows via SQL pushdown or a memory-mapped graph engine.
- **So what:** First few-shot foundation model to surpass supervised relational approaches (RelGNN) on RelBenchV1 (avg AUROC 79.60 vs. 78.06), with performance further improved ~16% by fine-tuning.

## Challenges & Novelty

Relational databases require aggregating signals across multiple connected tables while preserving temporal consistency — something tabular FMs avoid by flattening to a single row. Fixed-function, column-wise encoders like DFS fail on tasks requiring row-level alignment (e.g., conjunction: AUROC 0.5 vs. KumoRFM-2's 1.0). KumoRFM-1 addressed the multi-table setting but was constrained to in-memory data and used a less expressive single-stage architecture.

- **Expressivity gap of flat encoders:** column-wise aggregation cannot detect that two binary features co-occur in the same child row — a fundamental limitation any fixed-function flattener shares.
- **Scale barrier:** real enterprise databases have 500B+ rows; prior RFMs required the entire graph in memory.
- **Task-agnostic feature extraction:** without injecting the prediction target early, the model aggregates equally across task-irrelevant columns, adding noise that degrades ICL performance.

## Relation to Prior Work

| Model | Tokenization | Multi-table | Few-shot | Scale |
|---|---|---|---|---|
| DFS+AutoGluon | Column-wise flat | Yes (fixed-fn) | No | Large |
| [ranjan2025relationaltr](ranjan2025relationaltr.md) (RT) | Cell | Yes (4 attn masks) | Yes (zero-shot) | Moderate |
| KumoRFM-1 | Row (subgraph) | Yes | Yes | In-memory only |
| [dwivedi2025relgt](dwivedi2025relgt.md) | Row (5-element) | Yes | No (supervised) | Moderate |
| **KumoRFM-2** | Row (subgraph) | Yes (hierarchical) | Yes | 500B+ rows |

- [ranjan2025relationaltr](ranjan2025relationaltr.md): RT tokenizes at cell level for schema-agnostic zero-shot transfer; KumoRFM-2 stays row-level but gains expressivity through task-conditioned hierarchical attention rather than PE-free generality.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT is the strongest supervised baseline KumoRFM-2 surpasses; both operate on relational entity graphs but RelGT requires task-specific training while KumoRFM-2 uses ICL.
- KumoRFM-1: same RFM paradigm; v2 adds hierarchical attention, task-conditioned label injection, scale support, and fine-tuning.

## Technical Details

**Four-axis hierarchical attention.** Processing occurs at two stages:

*Stage 1 — intra-table (lightweight network):*
- **Column attention** — across columns within a row (multi-modal attribute understanding)
- **Row attention** — across rows within a table (entity-level relationships)

*Stage 2 — inter-table (larger network):*
- **Foreign-key attention** — across tables via PK-FK edges (relational topology)
- **Cross-sample attention** — across context examples (in-context learning signal)

Context targets $y_i$ are injected at the input stage so all four axes are task-conditioned throughout. Prediction:

$$\hat{y}_n = \mathrm{RFM}_{\bm{\theta}}^{\text{frozen}}\!\left(\mathcal{G}^{\leq t_n}[v_n],\; \{(\mathcal{G}^{\leq t_i}[v_i], y_i)\}_{i=1}^{n-1}\right)$$

![KumoRFM-2 architecture: intra-table column/row attention followed by inter-table FK/cross-sample attention](assets/fey2025kumorfm2-arch.png)

**Context selection.** Structured combination of *local* context (prediction entity's own lagged subgraph history) and *global* context (most recent database snapshots from other entities). At most 10k context examples used; covers as little as 0.2% of available training data on large databases.

**Expressivity.** Row-level attention solves the conjunction task (predict $y=1$ iff features $A$ and $B$ co-occur in at least one child row): column-wise encoders achieve AUROC 0.5; KumoRFM-2 achieves AUROC 1.0.

**Scale.** Three operating modes: (1) in-memory, (2) SQL pushdown via recursive metapath queries — no intermediate graph, (3) memory-mapped graph engine — 500B+ rows, 5 GB/sec bandwidth, 20M lookups/sec.

## Experiments

- KumoRFM-2 achieves avg AUROC 79.60 on RelBenchV1 classification, surpassing RelGNN (78.06) and KumoRFM-1 (76.71) without any task-specific training.
- On RelBenchV2, avg AUROC 79.96 vs. GraphSAGE supervised baseline 78.19.
- On SAP SALT (8 enterprise multi-class tasks), beats AutoGluon ensembles by ~8% and TabICLv2 by ~25% in MRR.
- Fine-tuning adds ~16% improvement beyond ICL-only performance, with reduced dependence on context selection quality.
- Robust under extreme cold-start (10 context examples), feature sparsity, and structural noise — performance degrades gracefully rather than catastrophically.

## Entities & Concepts

- [relational-foundation-model](relational-foundation-model.md)
- [relational-deep-learning](relational-deep-learning.md)
- [relbench](relbench.md)
- [relational-entity-graph](relational-entity-graph.md)
- [graph-transformer](graph-transformer.md)
- [tabular-learning](tabular-learning.md)
