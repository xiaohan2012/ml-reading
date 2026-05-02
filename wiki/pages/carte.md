---
title: "CARTE: Pretraining and Transfer for Tabular Learning"
tags: [source, tabular, pretraining, graph, transfer-learning, foundation-model]
sources: [carte]
updated: 2026-04-29
---

# CARTE: Pretraining and Transfer for Tabular Learning

**Source:** https://arxiv.org/abs/2402.16785
**Title:** CARTE: Pretraining and Transfer for Tabular Learning
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Kim, Grinsztajn, Varoquaux
**Venue:** ICML 2024
**Year:** 2024

## Summary

Kim, Grinsztajn, and Varoquaux (INRIA) introduce CARTE (Context-Aware Representation of Table Entries) — the first pretrained model for tabular data that can generalize across tables with *different* columns and schemas, without requiring entity matching or schema alignment. The key insight: represent each row as a star-shaped graph and use edge features (column names) to contextualize cell values, enabling pretraining across heterogeneous tables.

**Graph representation.** Each table row $i$ becomes a small graph $G_i(X, E)$: a star topology where the center node represents the row and $k$ leaf nodes represent the $k$ cell values. Edges carry column name embeddings. Features are initialized from FastText embeddings — categorical/string cell values get a direct language model embedding; numerical cells get $\text{value} \times \text{column\_embedding}$ (element-wise product). The center node is initialized as the mean of leaf embeddings and serves as the readout.

**Architecture.** CARTE adapts a Transformer encoder to graph attention. The key innovation is incorporating *edge features* (column names) into key and value computation:

$$\vec{K}_{ij} = (\vec{X}_i^{(l)} \odot \vec{E}_{ij}^{(l)}) \cdot \mathbf{W}_K, \quad \vec{V}_{ij} = (\vec{X}_i^{(l)} \odot \vec{E}_{ij}^{(l)}) \cdot \mathbf{W}_V$$

This element-wise product between node and edge features (from knowledge graph embedding literature) lets attention weight neighbors by both their value and their semantic relation (column identity). Attention masks enforce the graph structure (only connected neighbors attend).

**Pretraining.** CARTE is pretrained on YAGO3 — a 18.1M-triplet knowledge base extracted from Wikidata. Triplets $(head, relation, tail)$ are converted to star-shaped graphlets matching the table row format. Self-supervised contrastive loss (InfoNCE): original graphlet vs. truncated version (30-70% of edges deleted) as positives; other batch items as negatives.

**Fine-tuning.** Reuses the pretrained attention layers + readout layer (not the full network — avoids over-smoothing). Bagging across random train/validation splits for early stopping. Supports:
- *Single table*: standard supervised fine-tuning
- *Pairwise transfer*: joint training on source + target table without schema matching (ensemble with single-table model)
- *Multi-table*: all pairwise learners ensembled with single-table model, weighted by validation performance

**Results.** CARTE outperforms 42 baselines (including CatBoost, XGBoost, TabPFN, LLM-based) across 51 tabular datasets. Particularly strong on tables with high-cardinality string/categorical columns — a regime where tree-based models historically struggle. Robust to missing values (handles them by simply removing leaf nodes). Limitation: significantly slower than tree-based models (85–315 sec vs <1 sec for CatBoost).

## Key Takeaways

- **Graph = schema-agnostic representation**: star-shaped graphlet with column names as edges puts rows from different tables in the same computational domain — the key enabling pretraining across heterogeneous tables without data integration.
- **Edge-augmented attention disambiguates context**: the element-wise product $X_i \odot E_{ij}$ encodes that entry meaning depends on its column — "George Bush" as "son of" vs. "father of" leads to different entity resolutions.
- **YAGO pretraining provides entity world knowledge**: 6.3M real-world entities with their relations; fine-tuning tables with string entries benefit from this prior even without explicit entity matching.
- **FastText + value×embedding handles numerical cells**: no special numerical tokenization needed; column name embedding serves as the "unit" and value scales it — a simple but effective approach.
- **Bagging compensates for small tables**: most tabular benchmarks have small training sets; averaging predictions across different train/validation splits for early stopping consistently helps.

## Entities & Concepts

- [relational-deep-learning](relational-deep-learning.md)

## Relation to Other Wiki Pages

- [relational-deep-learning](relational-deep-learning.md): CARTE's graph representation of rows is closely related to RDL's graph representation of relational databases; both use graph structure to enable cross-table learning.
- [tabicl](tabicl.md): both are tabular foundation models from INRIA's SODA team; CARTE uses graph + pretraining on real knowledge, TabICL uses Transformer + pretraining on synthetic data; CARTE handles strings better, TabICL handles large numerical tables better.
- [graph-neural-network](graph-neural-network.md): CARTE's architecture is GAT-style with edge features; the edge-in-attention mechanism mirrors HGT's type-specific parameterization for the schema-as-edge case.
