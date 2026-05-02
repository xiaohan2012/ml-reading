---
title: "KumoRFM: A Foundation Model for In-Context Learning on Relational Data"
tags: [source, paper, relational-deep-learning, foundation-model, in-context-learning, explainability]
sources: [kumorfm-1]
updated: 2026-04-29
---

# KumoRFM: A Foundation Model for In-Context Learning on Relational Data

**Source:** https://kumo.ai/research/kumo_relational_foundation_model.pdf
**Title:** KumoRFM: A Foundation Model for In-Context Learning on Relational Data
**Type:** paper (whitepaper)
**Authors:** Matthias Fey, Vid Kocijan, Federico Lopez, Jan Eric Lenssen, Jure Leskovec (Kumo AI / Stanford)
**Venue:** Kumo AI technical whitepaper (no public arxiv)
**Year:** 2025

## Summary

KumoRFM (v1) is the first Relational Foundation Model (RFM) — a pre-trained model capable of making predictions on any relational database and any predictive task without task-specific training. It extends in-context learning (ICL) to the multi-table relational graph setting by combining three components: a table-invariant row embedding module, a Relational Graph Transformer for cross-table attention, and a final ICL module that conditions on context subgraphs and their labels at inference time.

The core insight is that historical labels for a predictive query can be generated on-the-fly from the database itself using the Predictive Query Language (PQL), a SQL-like declarative interface. Given a PQL query, KumoRFM dynamically constructs context examples `{(G^{≤t̂}[ê], y^{t̂}_ê)}` from historical database snapshots and uses them to predict the label for a new entity at a future timestamp — all in a single forward pass with frozen weights. ICL operates in two complementary ways: within the entity's subgraph (attending to its own historical labels and those of nearby entities) and across sampled subgraphs (subgraph-wise attention for broader contextual patterns).

On RelBench v1 (30 tasks, 7 databases, zero overlap with pretraining data), KumoRFM in-context mode outperforms supervised RDL (HeteroGraphSAGE) on classification tasks (76.71 vs 75.83 avg AUROC) and achieves strong regression/recommendation results — all within ~1 second and 1 line of code (vs. hours and hundreds of lines for supervised methods). Fine-tuning boosts classification to 81.14 avg AUROC, surpassing the best supervised baseline by 7% relative.

## Key Takeaways

- **First RFM system**: schema-agnostic ICL directly on relational databases; no task/dataset-specific training needed
- **3-stage architecture**: table-invariant encoder (per-column semantic type → row embeddings) → Relational Graph Transformer (node type + hop + time + subgraph PEs) → ICL Transformer (context+test subgraphs + labels)
- **Dual ICL mechanism**: within-subgraph (autoregressive entity history + relational neighbors) + cross-subgraph (subgraph-wise attention)
- **PQL interface**: declarative PREDICT/FOR/WHERE syntax; 1 LoC vs. ~878 for data scientists, ~56 for RDL
- **Speed**: ~1 second to first prediction vs. ~30 min (RDL) vs. ~12.3 hours (data scientist)
- **Built-in explainability**: saliency-per-cell (gradient-based, adapted for multi-modal inputs) + global cohort-level column importance + LLM-generated text summaries
- **Fine-tuning path**: replaces table-invariant encoders with dataset-specific ones + task-specific head; converts ICL to supervised production pipeline for billion-scale inference

## Architecture Details

**Stage 1 — Table-Invariant Row Encoder:**
- Each column encoded by semantic type: numerical, categorical, multi-categorical, timestamp, text, embedding, hashed ID
- Transformer over the 2D cell grid `C_T ∈ R^{N_T × C_T × F}` → row embeddings `H^(0)_T ∈ R^{N_T × F}`
- Table-size-agnostic: works for any number of rows/columns

**Stage 2 — Relational Graph Transformer (RelGT, Dwivedi et al. 2025):**
- Applies self-attention across all nodes in the subgraph (rows from all tables + dynamically attached context table T̂)
- 4 positional encodings added to row embeddings: node type, hop distance, relative time, subgraph structure
- L layers of table-wise interaction; reads out entity embedding `h^(L)_{T,e}`

**Stage 3 — ICL Module:**
- Encodes context and test subgraph representations independently
- Stacks into `H^(L)_train` + labels `y_train`, then applies Transformer to predict `y_test`
- Link prediction: ContextGNN-style pair-wise user/item representations (Yuan et al. 2025)

## Results on RelBench v1

| Task Type | KumoRFM (ICL) | KumoRFM (fine-tuned) | Best Supervised |
|---|---|---|---|
| Classification (AUROC) | 76.71 | **81.14** | 75.83 (RDL) |
| Regression (normalized MAE) | **0.984** (vs RDL 1.0) | **0.862** | 1.0 (RDL) |
| Recommendation (MAP@k) | **7.29** | **8.82** | 6.74 (NBFNet) |

## Explainability

KumoRFM provides two levels of explanation:
- **Global (cohort-level)**: columns across all tables organized into cohorts; prediction variance across cohorts quantifies column importance
- **Local (entity-level)**: saliency scores computed per cell (not per feature) using adapted gradient attribution — identifies which specific cells drove a prediction
- Both signals passed to an LLM to generate natural-language summaries

## Entities & Concepts

- [relational-foundation-model](relational-foundation-model.md)
- [relational-deep-learning](relational-deep-learning.md)
- [relational-entity-graph](relational-entity-graph.md)
- [relbench](relbench.md)
- [graph-transformer](graph-transformer.md)

## Relation to Other Wiki Pages

- **Predecessor to** [kumorfm-2](kumorfm-2.md): v2 adds hierarchical two-stage attention, task-conditioned processing, billion-scale SQL/graph engine support, and stronger results
- **Uses** [relational-graph-transformer](relational-graph-transformer.md) (RelGT by Dwivedi et al. 2025) as its table-wise interaction backbone
- **Contrasts with** [relational-transformer](relational-transformer.md) (RT): KumoRFM uses row-level tokens + FK graph attention + explicit ICL module vs. RT's cell-level Relational Attention masks; both are schema-agnostic but different architectures
- **Contrasts with** [griffin](griffin.md): KumoRFM uses ICL for zero-shot transfer; Griffin uses meta-learning; both use GNN across tables
- **Introduces** explainability as a first-class feature of an RFM — gradient saliency per cell adapted for multi-modal table inputs
