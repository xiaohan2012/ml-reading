---
title: "KumoRFM: A Foundation Model for In-Context Learning on Relational Data"
tags: [source, relational-deep-learning, foundation-model, in-context-learning, explainability]
sources: [fey2025kumorfm]
updated: 2026-05-06
---

# KumoRFM: A Foundation Model for In-Context Learning on Relational Data

**Source:** https://kumo.ai/research/kumo_relational_foundation_model.pdf
**Title:** KumoRFM: A Foundation Model for In-Context Learning on Relational Data
**Date ingested:** 2026-04-29
**Type:** paper (whitepaper)
**Authors:** Matthias Fey, Vid Kocijan, Federico Lopez, Jan Eric Lenssen, Jure Leskovec
**Venue:** Kumo AI technical whitepaper (2025)

## Summary

- **What:** No foundation model existed for multi-table relational databases — all prior RDL methods required task-specific supervised training per database and task.
- **How:** KumoRFM (v1) combines a table-invariant row encoder, RelGT for cross-table attention, and an ICL Transformer that conditions on context subgraphs and their labels at inference; task labels are generated on-the-fly from historical database states via PQL.
- **So what:** First RFM — achieves 76.71 avg AUROC ICL-only on RelBench v1 (beating supervised HeteroGraphSAGE at 75.83), fine-tunes to 81.14; predictions in ~1 second with 1 line of code vs. hours and hundreds of lines for supervised methods.

## Challenges & Novelty

Extending ICL from flat tables (TabPFN) to multi-table relational databases requires solving two new problems: (1) how to aggregate information across PK-FK-linked tables without task-specific training, and (2) how to generate context examples (input-label pairs) on-the-fly from a database where labels are not pre-annotated but derivable from historical states.

- **No relational ICL:** TabPFN/TabICL operate on single flat tables; extending ICL to multi-table graphs requires a cross-table attention module that is schema-agnostic (works on any database structure).
- **Label generation from historical data:** in temporal databases, labels like "did this user churn in 30 days?" must be computed from future database events; KumoRFM uses PQL queries evaluated at past timestamps to automatically construct context examples without human annotation.
- **Dual ICL granularity:** within-subgraph context (an entity's own historical labels and relational neighbors) and cross-subgraph context (subgraph-level attention across many context examples) serve different roles — combining both improves ICL signal quality.

## Relation to Prior Work

| Model | Multi-table | ICL | Schema-agnostic | No task training |
|---|---|---|---|---|
| HeteroGraphSAGE ([fey2024rdlposition](fey2024rdlposition.md)) | Yes | No | No | No |
| [hollmann2025tabpfnv2](hollmann2025tabpfnv2.md) | No (single table) | Yes | Yes | Yes |
| [wang2025griffin](wang2025griffin.md) | Yes | No (meta-learning) | Yes | Yes |
| **KumoRFM v1** | Yes | Yes | Yes | Yes (ICL) |
| [fey2025kumorfm2](fey2025kumorfm2.md) | Yes | Yes | Yes | Yes |

- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT is used directly as KumoRFM v1's Stage 2 (cross-table attention); KumoRFM wraps it in the ICL framework.
- [hollmann2025tabpfnv2](hollmann2025tabpfnv2.md): TabPFN v2 is the tabular ICL ancestor; KumoRFM extends its paradigm to multi-table relational data.
- [fey2025kumorfm2](fey2025kumorfm2.md): v2 replaces the 3-stage pipeline with unified hierarchical 4-axis attention and early label injection, improving both performance and scalability.

## Technical Details

**Three-stage architecture:**

*Stage 1 — Table-Invariant Row Encoder.* Semantic-type-specific encoders (numerical, categorical, multi-categorical, timestamp, text, embedding, hashed ID) encode each cell; a Transformer over the cell grid $\mathbf{C}_T \in \mathbb{R}^{N_T \times C_T \times F}$ produces row embeddings $\mathbf{H}_T^{(0)} \in \mathbb{R}^{N_T \times F}$. Table-size-agnostic — works for any number of rows/columns.

*Stage 2 — Relational Graph Transformer ([dwivedi2025relgt](dwivedi2025relgt.md)).* Self-attention across all nodes (rows) in the subgraph. Four positional encodings: node type, hop distance, relative time, subgraph GNN PE. $L$ layers produce entity embedding $h_{T,e}^{(L)}$.

*Stage 3 — ICL Transformer.* Context and test subgraph representations are encoded independently, stacked with labels $y_\text{train}$, and a Transformer predicts $y_\text{test}$. Two ICL modes: within-subgraph (autoregressive entity history + relational neighbors) and cross-subgraph (subgraph-wise attention over full context set).

**PQL (Predictive Query Language).** Declarative PREDICT/FOR/WHERE syntax — 1 line of code vs. ~878 for data scientists. Labels computed by evaluating the query at historical timestamps, generating context triples $(v_i, t_i, y_i)$ automatically.

**Explainability.** Two levels: (1) global cohort-level column importance via prediction variance across cohorts; (2) local entity-level gradient saliency per cell (adapted for multi-modal inputs). Both passed to an LLM for natural-language summaries.

## Experiments

- ICL-only: avg AUROC 76.71 on RelBench v1 classification, beating supervised HeteroGraphSAGE (75.83) — first RFM to surpass a supervised RDL baseline.
- Fine-tuned: avg AUROC 81.14 (+7% relative over best supervised baseline); regression normalized MAE 0.862 vs. 1.0 for supervised.
- ~1 second to first prediction vs. ~30 min (RDL training) vs. ~12.3 hours (data scientist pipeline).

## Entities & Concepts

- [relational-foundation-model](relational-foundation-model.md)
- [relational-deep-learning](relational-deep-learning.md)
- [relational-entity-graph](relational-entity-graph.md)
- [relbench](relbench.md)
- [graph-transformer](graph-transformer.md)
