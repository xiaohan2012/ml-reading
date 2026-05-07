---
title: "Explainability in Relational Deep Learning"
tags: [query, analysis, relational-deep-learning, explainability]
sources: [fey2025kumorfm, fey2025kumorfm2]
updated: 2026-04-29
---

# Explainability in Relational Deep Learning

The only RDL papers in the wiki with explicit explainability methods are [fey2025kumorfm](fey2025kumorfm.md) and [fey2025kumorfm2](fey2025kumorfm2.md). No other ingested papers address explainability. See **Gaps** section below.

## KumoRFM Explainability (Two Levels)

### 1. Global / Data-Level (Analytical)

Organizes all column-level context data into *cohorts* — groups of entities sharing similar column values — then links cohort distributions to ground-truth labels.

**Column importance score** = variance of model predictions across its cohorts. Higher variance → the column's value meaningfully shifts predictions → it matters.

Works across all tables, not just the seed entity's table: adjacent tables are handled via weighted cohorts. Example: a user's purchase history columns can be scored even though purchases live in a child table.

### 2. Local / Entity-Level (Gradient-Based)

KumoRFM is fully differentiable end-to-end, enabling gradient attribution directly.

**Method: saliency** (Simonyan et al. 2013) — gradient of prediction w.r.t. each input feature. Chosen for strong tradeoff between interpretability and compute efficiency (vs. SHAP/LIME which require many model evaluations).

**Key novelty: scores at the cell level, not the feature level.** Each cell may span multiple features (e.g., a text column is embedded into many dimensions). KumoRFM aggregates gradients across those dimensions using type-specific routines per semantic type (numerical, categorical, text, embedding, etc.), producing a single importance score per `(table, row, column)` triple — directly actionable for users.

### LLM Summarization

Both global and local signals are passed to an LLM to generate natural-language summaries. The saliency scores are the structured signal; the LLM provides prose.

**Example — churn prediction** (`PREDICT COUNT(orders.*, 0, 30) > 0 FOR users.user_id=1`):
> "Users with only a few past orders have a very low likelihood of ordering soon… Recent orders (6–12 months ago) greatly increase the chance… Users who regularly receive fashion news or have active club membership show higher probabilities."
> Top factors: Order Count, Order Date Recency, Fashion News Frequency, Club Membership.

**Example — recommendation** (`PREDICT LIST_DISTINCT(orders.item_id, 0, 7) FOR users.user_id=2`):
> "Descriptions of items recently viewed (jackets, t-shirts, jumpers) are among the most important features, with importance scores above 95%… User is currently active."

## Trust Layer: Prediction Accuracy Evaluation

Alongside explanations, KumoRFM provides a quantitative evaluation mode to build confidence before acting on predictions:

- **Temporal tasks**: replays historical database states at earlier timestamps where ground truth is known; evaluates performance there and reports it to the user.
- **One-off (static) tasks**: masks known cell values and measures imputation accuracy.
- Reports both **performance metrics** (AUROC, AP, MAE, MAP@k) and **behavioral metrics** (diversity, popularity bias mitigation) — recognizing that accuracy-only optimization can collapse to recommending only popular items.

## KumoRFM-2

[fey2025kumorfm2](fey2025kumorfm2.md) states that the system "incorporates mechanisms for explainability, offering insights into the reasoning processes that lead to its predictions" but provides no new technical details beyond v1. The same two-level approach is assumed to carry over.

## Gaps

- **No graph-topology explanation**: saliency is over input features, not over graph structure (edges, hops, subgraph paths). Which neighbor nodes or FK links drove the prediction is not addressed.
- **No comparison to GNN-specific explainability methods**: GNNExplainer, PGExplainer, SubgraphX — perturbation/mask-based approaches designed specifically for GNN architectures — are not discussed or compared. The wiki has no pages on these methods.
- **Cell-level saliency aggregation** per semantic type is described conceptually but not formally specified in the whitepaper (no formulas given).
- **No other RDL paper in the wiki** (RelGNN, RelGT, Griffin, RT) addresses explainability at all.

## Related Pages

- [fey2025kumorfm](fey2025kumorfm.md) — source of all explainability technical detail
- [fey2025kumorfm2](fey2025kumorfm2.md) — mentions explainability as a system feature
- [relational-foundation-model](relational-foundation-model.md) — broader RFM context
- [rdl-approaches](rdl-approaches.md) — other RDL methods (none address explainability)
