---
title: "CARTE: Pretraining and Transfer for Tabular Learning"
tags: [source, tabular, pretraining, graph, transfer-learning, foundation-model]
sources: [kim2024carte]
updated: 2026-05-07
---

# CARTE: Pretraining and Transfer for Tabular Learning

**Source:** https://arxiv.org/abs/2402.16785
**Title:** CARTE: Pretraining and Transfer for Tabular Learning
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Kim, Grinsztajn, Varoquaux
**Venue:** ICML 2024

## Summary

- **What:** Tabular foundation models fail to transfer across tables with different columns and schemas because they treat rows as flat feature vectors — there is no schema-agnostic representation.
- **How:** Represent each row as a star-shaped graph (center = row, leaves = cell values, edges = column names as FastText embeddings), pretrain on YAGO3 knowledge base triples using contrastive loss, then fine-tune.
- **So what:** First pretrained model that transfers across heterogeneous tables without schema alignment; outperforms 42 baselines including CatBoost, XGBoost, TabPFN, and LLM-based methods across 51 tabular datasets — particularly strong on high-cardinality string/categorical columns.

## Challenges & Novelty

Cross-table transfer is fundamentally blocked by schema heterogeneity: each table has different columns with different names and types. A flat feature vector representation assumes fixed column positions, making transfer impossible when columns differ. CARTE sidesteps this by using the column name itself as a relational edge — placing all tables in the same graphical computation domain.

- **Star graph as schema-agnostic representation:** every row becomes a small graph where column names are edge features. The same Transformer can process any row from any table — the column identity is carried by the edge, not the node position.
- **Edge-augmented attention disambiguates column context:** the element-wise product $\mathbf{X}_i \odot \mathbf{E}_{ij}$ in key/value computation means "George Bush" as value of edge *father-of* vs. *son-of* produces different attention keys — essential for entity-heavy tables.
- **YAGO3 pretraining provides real-world entity knowledge:** 18.1M knowledge graph triplets (real entities + relations) give CARTE world knowledge without explicit entity linking — any table with string/categorical values implicitly benefits.

## Relation to Prior Work

| Model | Cross-table transfer | String columns | Pretraining data | Architecture |
|---|---|---|---|---|
| CatBoost / XGBoost | No | Limited | None | Tree |
| [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md) | No (fixed columns) | No | Synthetic | Transformer |
| [somepalli2021saint](somepalli2021saint.md) | No | Limited | None | Transformer |
| [qu2025tabicl](qu2025tabicl.md) | No (fixed-width) | No | Real + synthetic | Transformer |
| **CARTE** | Yes (via star graph) | Yes (FastText) | YAGO3 (real KG) | Graph Transformer |

- [qu2025tabicl](qu2025tabicl.md): both from INRIA SODA; CARTE handles strings and cross-table transfer via graphs; TabICL handles large numerical tables in-context — complementary strengths.
- [somepalli2021saint](somepalli2021saint.md): SAINT uses row-and-column attention with contrastive pretraining; CARTE uses graph-per-row with KG pretraining. Different data regimes: SAINT for structured numeric tables, CARTE for string/entity-heavy tables.
- [wang2025griffin](wang2025griffin.md): Griffin represents relational database rows similarly (text encoder per cell, cross-attention per row); CARTE is the tabular analogue from the non-RDL community.

## Technical Details

**Graph representation.** For a row with $c$ columns:
- Center node: row representation (initialized as mean of leaf embeddings)
- $c$ leaf nodes: one per cell; features initialized as FastText embeddings of string/categorical values, or $\text{value} \times \mathbf{e}_\text{col}$ for numerical values
- Edges: column name embeddings $\mathbf{E}_{ij} \in \mathbb{R}^d$ (FastText of column name)

**Edge-augmented attention:**

$$\vec{K}_{ij} = (\mathbf{X}_i^{(l)} \odot \mathbf{E}_{ij}^{(l)}) \cdot W_K, \qquad \vec{V}_{ij} = (\mathbf{X}_i^{(l)} \odot \mathbf{E}_{ij}^{(l)}) \cdot W_V$$

Queries are from node features only ($\mathbf{X}_i^{(l)} W_Q$). Attention masks enforce star-graph structure — center attends to all leaves; leaves attend only to center.

**Pretraining.** YAGO3 triples $(head, relation, tail)$ → star graphlets matching the row format. InfoNCE contrastive loss: original graphlet vs. truncated version (30–70% of edges deleted) as positive pair; other batch items as negatives.

**Fine-tuning modes:**
- *Single table*: supervised fine-tuning on the target table
- *Pairwise transfer*: joint training on source + target; ensemble with single-table model
- *Multi-table*: all pairwise learners ensembled, weighted by validation performance

**Bagging.** Multiple train/validation splits for early stopping; averaged predictions reduce variance on small tables.

## Experiments

- Outperforms 42 baselines across 51 tabular datasets; especially strong advantage on tables with high-cardinality string/categorical columns.
- On tables where CatBoost excels (purely numerical, no string features), CARTE advantage is smaller but generally positive.
- Limitation: 85–315 seconds inference time vs. <1 second for CatBoost — impractical for latency-sensitive applications.
- Multi-table transfer consistently outperforms single-table fine-tuning when source and target share entity types.

## Entities & Concepts

- [tabular-learning](tabular-learning.md)
- [relational-deep-learning](relational-deep-learning.md)
