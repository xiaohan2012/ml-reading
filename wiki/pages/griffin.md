---
title: "Griffin: Towards a Graph-Centric Relational Database Foundation Model"
tags: [source, relational-deep-learning, foundation-model, gnn]
sources: [griffin]
updated: 2026-04-29
---

# Griffin: Towards a Graph-Centric Relational Database Foundation Model

**Source:** https://arxiv.org/abs/2505.05568
**Title:** Griffin: Towards a Graph-Centric Relational Database Foundation Model
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Wang, Wang, Gan, Wang, Yang, Wipf, Zhang
**Venue:** ICML 2025
**Year:** 2025

## Summary

Wang, Wang, Gan, Wang, Yang, Wipf, and Zhang (PKU / Amazon) introduce Griffin, positioned as the first foundation model for relational databases. The core thesis: a single pretrained model can generalize across diverse RDBs with different schemas, domains, and task types — analogous to LLMs for text.

**Architecture.** Griffin converts an RDB to a heterogeneous temporal graph (rows as nodes, PK-FK links as edges) and processes it with a three-component pipeline:
1. *Unified data encoder*: categorical and text features are encoded with a pretrained text encoder (Nomic); numerical features pass through a pretrained float encoder (MLP trained to encode/decode floats with cosine loss); RDB metadata (table names, column names, edge types) provides additional node/edge context; a task embedding (text of the target column name) distinguishes which cell to predict.
2. *MPNN with cross-attention*: a cross-attention module uses the current node representation + task embedding as query, column names as keys, and cell features as values — attending over cells within a row before propagating. Hierarchical aggregation then averages neighbors within each relation type before taking max across relation types. This reduces information loss from mean-aggregating all columns uniformly.
3. *Unified task decoder*: classification uses inner products with text embeddings of category labels (handles arbitrary label sets); regression uses the pretrained float decoder (denormalized per task).

**Training pipeline.** Three stages: (1) masked cell completion pretraining on single-table datasets (cosine loss between predicted row embedding and true cell embedding); (2) joint supervised fine-tuning (SFT) on single-table + RDB tasks; (3) task-specific fine-tuning on downstream tasks. Pretraining covers 150M+ nodes across heterogeneous and temporal graphs. Three variants: Griffin-unpretrained, Griffin-pretrained (single-table only), Griffin-RDB-SFT.

**Results on RelBench and 4DBInfer.** Even without pretraining, Griffin's architecture outperforms GNN baselines in many cases (cross-attention + hierarchical aggregation provide architectural gains). Pretraining on single-table data adds further gains. RDB SFT is most beneficial in low-data scenarios and when pretraining and downstream tasks are similar in domain/structure. Griffin does not consistently beat individually fine-tuned task-specific models at high data regimes.

## Key Takeaways

- **Unified input encoders are critical for schema-agnostic generalization**: text encoder for categoricals + float encoder for numericals maps all feature types to the same embedding space, enabling transfer across databases with completely different column semantics.
- **Cross-attention over row cells beats mean aggregation**: attending to which cells are relevant per task (keyed by column names) recovers information lost when columns are naively pooled — a general principle for row-level representation.
- **Hierarchical aggregation (intra-relation mean → inter-relation max)**: stabilizes aggregation when neighbor counts across relation types are highly variable, a practical issue in real-world RDL graphs.
- **Pretraining transfers but doesn't dominate at high data**: Griffin-RDB-SFT > Griffin-pretrained > Griffin-unpretrained in low-data regimes; gap narrows with more downstream data — consistent with general foundation model behavior.
- **Masked cell completion as self-supervised RDB pretraining**: predict masked column values from the same row (single-table) or from relational context (RDB); this unifies with the [Relational Transformer](relational-transformer.md)'s MTP objective and [RelBench v2](relbench-v2.md)'s autocomplete tasks.

## Entities & Concepts

- [relational-deep-learning](relational-deep-learning.md)
- [relational-entity-graph](relational-entity-graph.md)
- [relational-foundation-model](relational-foundation-model.md)
- [relbench](relbench.md)
- [autocomplete-tasks](autocomplete-tasks.md)

## Relation to Other Wiki Pages

- [relational-foundation-model](relational-foundation-model.md): Griffin is the first explicit attempt at a pretrained foundation model for RDBs; [Relational Transformer](relational-transformer.md) achieves better zero-shot AUROC (93%) at smaller scale (22M params), but Griffin is the first to attempt pretraining across diverse RDB corpora.
- [relational-transformer](relational-transformer.md): Both use masked/cell-level prediction as self-supervised objective; RT uses cell-level tokenization across tables, Griffin uses cross-attention within a row + GNN across tables.
- [relbench-v2](relbench-v2.md): Griffin is evaluated on RelBench (v1); its masked cell completion pretraining directly connects to RelBench v2's autocomplete tasks and ReDeLEx pretraining infrastructure.
- [relgnn](relgnn.md): RelGNN fixes the message-passing topology within the supervised RDL setting; Griffin focuses on pretraining generalization — complementary approaches.
