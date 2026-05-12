---
title: Relational Foundation Model
tags: [concept, foundation-model, relational-deep-learning]
sources: [fey2024rdlposition, ranjan2025relationaltr, gu2026relbenchv2, wang2025griffin, fey2025kumorfm, fey2025kumorfm2]
updated: 2026-04-29
---

# Relational Foundation Model

## Description

A relational foundation model is a model pretrained on diverse relational databases and capable of generalizing to new datasets and tasks with minimal or no fine-tuning. The vision is analogous to LLMs/foundation models in NLP: a single pretrained model serves as a strong initialization for any relational prediction task.

The core challenges distinguish this from NLP/CV foundation models:
- **Schema diversity**: different databases have different tables, columns, datatypes, and relational topology — the model must be schema-agnostic.
- **No shared vocabulary**: unlike text tokens, relational cells don't form a universal vocabulary; column semantics must be inferred from names/types.
- **Temporal constraints**: temporal leakage must be enforced differently for each database's schema and task definition.

**Key approaches in the wiki:**
- [KumoRFM-1](fey2025kumorfm.md): first RFM system; table-invariant encoder + RelGT + ICL module; PQL declarative interface; explainability (saliency-per-cell); 76.71 AUROC ICL / 81.14 fine-tuned on RelBench v1.
- [KumoRFM-2](fey2025kumorfm2.md): hierarchical 4-scale attention (row/col within tables → FK/cross-sample across tables); task-conditioned ICL; first few-shot RFM to surpass supervised SOTA (RelGNN) on RelBench v1; scales to 500B+ rows.
- [Relational Transformer (RT)](ranjan2025relationaltr.md): cell-level tokenization + Relational Attention + task table prompting; zero-shot 93% of supervised AUROC on binary classification with 22M params.
- [Griffin](wang2025griffin.md): first explicit pretrained foundation model for diverse RDBs; unified text encoder (Nomic) + float encoder + cross-attention over row cells + GNN across tables; 3-stage training (completion pretraining → joint SFT → task fine-tuning) on 150M+ nodes.
- [RelGT](dwivedi2025relgt.md): not a foundation model (schema-specific), but the strongest supervised architecture — row-level tokenization with rich 5-element PEs.

**Scale requirements**: [RelBench v2](gu2026relbenchv2.md) and ReDeLEx (70+ databases) exist specifically to provide pretraining data of sufficient scale and diversity for this goal.

## Appearances in Sources

- [fey2024rdlposition](fey2024rdlposition.md) — Sec. 4 explicitly calls for foundation models for relational data; identifies self-supervised label mining and inductive schema generalization as open problems
- [ranjan2025relationaltr](ranjan2025relationaltr.md) — RT is the first architecture to achieve strong zero-shot transfer across relational databases
- [gu2026relbenchv2](gu2026relbenchv2.md) — v2's expanded scale and ReDeLEx integration are explicitly motivated by foundation model development
- [wang2025griffin](wang2025griffin.md) — first model to pretrain across a corpus of diverse RDBs; introduces masked cell completion as the pretraining objective; most beneficial in low-data regimes
- [fey2025kumorfm](fey2025kumorfm.md) — first RFM; 3-stage architecture (table-invariant encoder + RelGT + ICL); PQL interface; explainability via saliency-per-cell; 76.71 AUROC in-context / 81.14 fine-tuned
- [fey2025kumorfm2](fey2025kumorfm2.md) — ICL-based RFM with hierarchical attention over row/col/FK/cross-sample dimensions; task-conditioned; first few-shot model to beat supervised methods on RelBench v1

## Related Concepts

- [relational-deep-learning](relational-deep-learning.md) — the learning paradigm; foundation models are its frontier
- [relational-attention](relational-attention.md) — RT's core mechanism enabling schema-agnostic modeling
- [autocomplete-tasks](autocomplete-tasks.md) — new task type especially useful for self-supervised pretraining
- [relbench](relbench.md) — benchmark used for pretraining and evaluation
