---
title: Autocomplete Tasks
tags: [concept, benchmark, relational-deep-learning]
sources: [gu2026relbenchv2, ranjan2025relationaltr]
updated: 2026-04-28
---

# Autocomplete Tasks

## Description

Autocomplete tasks are a class of predictive tasks introduced in [RelBench v2](gu2026relbenchv2.md) where the objective is to predict missing or masked attribute values in *existing* relational table columns, as opposed to **forecasting tasks** that predict future outcomes constructed via SQL queries over future rows.

Autocomplete is still temporal: a seed time is assigned to each prediction, and only rows created before that seed time are available. A target column's value is masked; the model must infer it from relational and temporal context in the surrounding database.

**Contrast with forecasting**:
| | Autocomplete | Forecasting |
|---|---|---|
| Target | Existing column value | Future SQL-derived label |
| Data location | Already in the database | Constructed from future rows |
| Example | Infer a product's shipping category | Predict if a user will churn next month |

**Leakage prevention**: columns highly correlated with the target must be dropped (e.g., `review_text` for a `review-rating` autocomplete task).

**Real-world motivation**: directly inspired by the SAP S/4HANA sales order interface, which autocompletes payment categories based on relational context.

**RT framing**: the [Relational Transformer](ranjan2025relationaltr.md) unifies autocomplete and forecasting as **masked token prediction (MTP)**: in both cases a cell value is masked and the model predicts it. This allows a single pretraining objective to cover both task types.

**RelBench v2**: 23 autocomplete tasks across 7 datasets; RDL (HeteroGraphSAGE) outperforms LightGBM on all, confirming relational context is informative for attribute inference.

## Appearances in Sources

- [gu2026relbenchv2](gu2026relbenchv2.md) — introduced as a new benchmark task type (Section 3, 4)
- [ranjan2025relationaltr](ranjan2025relationaltr.md) — framed as masked token prediction alongside forecasting tasks

## Related Concepts

- [relbench](relbench.md) — the benchmark that defines these tasks
- [relational-deep-learning](relational-deep-learning.md) — the paradigm evaluated on autocomplete tasks
- [training-table](training-table.md) — autocomplete tasks differ from forecasting in their training table construction (no future SQL labels; masked column value is the target)
- [relational-foundation-model](relational-foundation-model.md) — autocomplete tasks useful for self-supervised pretraining
