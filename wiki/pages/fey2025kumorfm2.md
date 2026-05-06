---
title: "KumoRFM-2: Scaling Foundation Models for Relational Learning"
tags: [source, paper, relational-deep-learning, foundation-model, in-context-learning]
sources: [fey2025kumorfm2]
updated: 2026-04-29
---

# KumoRFM-2: Scaling Foundation Models for Relational Learning

**Source:** https://arxiv.org/abs/2604.12596
**Title:** KumoRFM-2: Scaling Foundation Models for Relational Learning
**Type:** paper
**Authors:** Matthias Fey, Vida Kocijan, Nhat Hoang, Jonathan Lehmann, Jan E. Lenssen, Jure Leskovec (Kumo AI / Stanford)
**Venue:** arXiv preprint (NeurIPS 2025 submission)
**Year:** 2025

## Summary

KumoRFM-2 is the second iteration of Kumo AI's Relational Foundation Model, designed to perform in-context learning (ICL) directly on multi-table relational databases without manual feature engineering or task-specific training. Where v1 was limited to in-memory datasets, v2 scales to 500B+ rows via SQL pushdown or a memory-mapped graph engine. It is the first few-shot foundation model demonstrated to surpass supervised relational approaches (including RelGNN) on standard benchmarks.

The architecture processes relational data at three scales simultaneously: a lightweight network handles row/column attention within individual tables (task-conditioned by early label injection), then a larger network applies foreign-key attention across tables and cross-sample attention across context examples. This hierarchical design avoids quadratic complexity while preserving full relational expressivity.

The key advance over v1 is task-conditioned processing throughout — labels from context examples are injected early, enabling the model to filter task-relevant columns and suppress noise before aggregating across tables. Combined with structured context selection (local prediction-entity history + global recent snapshots), KumoRFM-2 achieves strong performance with as few as 10k context examples, even when that represents 0.2% of available training data.

## Key Takeaways

- **First few-shot RFM to surpass supervised SOTA**: avg AUROC 79.60 on RelBench v1 vs. RelGNN's 78.06 — without any task-specific training
- **Hierarchical 4-scale attention**: row→column (within table) then FK→cross-sample (across tables); task-conditioned from the start
- **Expressivity advantage over fixed-function encoders**: column-wise DFS-style encoders (AUROC 0.5) fail on row-level conjunction tasks; KumoRFM-2 achieves AUROC 1.0
- **Scales to 500B+ rows**: SQL pushdown mode (no intermediate graph) or memory-mapped graph engine (5GB/sec bandwidth, 20M lookups/sec)
- **SALT benchmark**: beats AutoGluon ensembles by ~8%, TabICL v2 by ~25% on SAP enterprise ERP multi-class tasks
- **Fine-tuning adds ~16%** beyond ICL performance; ICL alone already beats supervised methods
- **System features**: Predictive Query Language (PQL) interface, explainability outputs, embeddings, stateless multi-tenant serving for agentic workflows

## Architecture Details

**Four attention axes:**
1. **Column attention** — across columns within a row (understand multi-modal attributes)
2. **Row attention** — across rows within a table (relate entities)
3. **Foreign-key attention** — across tables via PK-FK edges (capture relational topology)
4. **Cross-sample attention** — across context examples (in-context learning signal)

Labels are injected at the input stage so all four attention axes are task-conditioned.

**Context selection strategy:**
- *Local*: subgraph history of the prediction entity itself (lagged targets)
- *Global*: most recent database snapshots from other entities

**Training data:** mix of synthetic (Structural Causal Models) and real-world relational databases; multi-stage pretraining from single tables to complex relational structures.

## Experimental Results

| Benchmark | KumoRFM-2 | Best Supervised | Best Prior FM |
|---|---|---|---|
| RelBench v1 (class.) | **79.60** AUROC | 78.06 (RelGNN) | 76.71 (KumoRFM-1) |
| RelBench v2 (class.) | **79.96** AUROC | 78.19 (GraphSAGE) | 76.22 (KumoRFM-1) |
| SALT (multi-class) | SOTA (MRR) | +8% vs AutoGluon | +25% vs TabICL v2 |
| 4DBInfer | **79.96** AUROC | 79.94 (HGT) | — |

## Entities & Concepts

- [relational-foundation-model](relational-foundation-model.md)
- [relational-deep-learning](relational-deep-learning.md)
- [relbench](relbench.md)
- [relational-entity-graph](relational-entity-graph.md)
- [tabular-learning](tabular-learning.md)

## Relation to Other Wiki Pages

- Supersedes [fey2025kumorfm](fey2025kumorfm.md) — v2 adds hierarchical attention, task-conditioned processing, billion-scale support, and fine-tuning
- Outperforms [chen2025relgnn](chen2025relgnn.md) (78.06) on RelBench v1 without supervised training — first RFM to do so
- Shares ICL paradigm with [ranjan2025relationaltr](ranjan2025relationaltr.md) (RT) but differs: KumoRFM uses row-level tokens + FK graph attention vs. RT's cell-level tokens + Relational Attention masks
- Contrasts with [wang2025griffin](wang2025griffin.md) which also pretrained on relational data but uses meta-learning rather than ICL
- Extends [tabular-learning](tabular-learning.md) context: outperforms TabPFN v2 and TabICL on relational tasks where multi-table structure matters
