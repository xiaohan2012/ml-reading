---
title: "Main Approaches to Relational Deep Learning"
tags: [query, analysis, relational-deep-learning, gnn, graph-transformer, foundation-model]
sources: [relational-deep-learning-position, relgnn, relational-graph-transformer, relational-transformer, griffin, kumorfm-2]
updated: 2026-04-29
---

# Main Approaches to Relational Deep Learning

## 1. Heterogeneous GNN (baseline and beyond)

**HeteroGraphSAGE** is the standard RDL baseline from [relational-deep-learning-position](relational-deep-learning-position.md): separate weight matrices per relation type + temporal neighbor sampling (filter to τ(w) ≤ seed_time). Simple and strong enough to beat data scientists on all 30 RelBench tasks.

**RelGNN** ([relgnn](relgnn.md), ICML 2025, Chen, Kanatsoulis, Leskovec) goes further by exploiting PK-FK topology specifically:
- Identifies **bridge nodes** (2 FKs) and **hub nodes** (3+ FKs) as the characteristic patterns of relational graphs
- Replaces 2-hop aggregation with a single **FUSE operation**: `h_fuse = W₁·h_mid + W₂·h_src`, followed by attention at the destination
- Result: SOTA on 27/30 RelBench v1 tasks, up to 25% gain over HeteroGraphSAGE

## 2. Graph Transformer (schema-specific, supervised)

**RelGT** ([relational-graph-transformer](relational-graph-transformer.md), 2025): first GT designed for RDL. Solves the PE problem for heterogeneous temporal graphs with **5-element tokenization** per node: multimodal features + node type + hop distance + relative time + subgraph GNN PE (with random resampled init — the most critical component). Combines local all-pair attention (K=300 tokens) with global attention (B=4096 EMA centroids). Best supervised performance on RelBench.

## 3. Pretrained Foundation Models (schema-agnostic)

**Relational Transformer** ([relational-transformer](relational-transformer.md), ICLR 2026, Ranjan et al.): tokenizes at the **cell level** — `(value, column, table)` triples — making it universal across schemas. Uses **Relational Attention** (4 mask types: column/feature/neighbor/global) instead of positional encodings. Pretrained via masked token prediction on 6 RelBench databases. Achieves **93% of supervised AUROC zero-shot**, beating Gemma3-27B (27B params) at 22M params.

**Griffin** ([griffin](griffin.md), ICML 2025, Wang et al.): keeps GNN message passing across tables but adds **cross-attention within rows** (query = node+task embedding, keys = column names, values = cell features). Uses unified text+float encoders for schema-agnostic input. Pretrained on 150M+ nodes. Strongest in low-data regimes; gap narrows with more task data.

## 4. KumoRFM (ICL-based, schema-agnostic)

**KumoRFM-2** ([kumorfm-2](kumorfm-2.md), 2025, Fey et al., Kumo AI/Stanford): combines ICL with end-to-end relational processing. Unlike RT (masked token prediction pretraining) or Griffin (meta-learning), KumoRFM uses **task-conditioned ICL**: labels are injected early so all 4 attention axes (row, column, FK, cross-sample) can filter task-relevant signals. Results: **first few-shot RFM to beat supervised SOTA** — avg AUROC 79.60 vs RelGNN 78.06 on RelBench v1, with only 10k context examples (sometimes 0.2% of available data). Scales to 500B+ rows via SQL pushdown or memory-mapped graph engine.

## Summary Comparison

| Approach | Key idea | Best for |
|---|---|---|
| HeteroGraphSAGE | Relation-typed weights + temporal sampling | Simple baseline |
| RelGNN | FUSE op for bridge/hub topology | Supervised SOTA |
| RelGT | 5-element tokenization + hybrid attention | Supervised, schema-known |
| Griffin | Cross-attention within rows + GNN + pretraining | Low-data, schema-agnostic |
| RT | Cell-level tokens + Relational Attention + pretraining | Zero-shot cross-database |
| KumoRFM-2 | Task-conditioned ICL + hierarchical 4-scale attention | Few-shot, surpasses supervised |

## Central Tension

**Schema-specific methods** (RelGNN, RelGT) win on supervised benchmarks by exploiting relational structure deeply. **Schema-agnostic pretrained models** (RT, Griffin, KumoRFM) sacrifice some supervised performance to generalize across databases without retraining. KumoRFM-2 now breaks this tradeoff: its task-conditioned ICL surpasses supervised SOTA (RelGNN) without any task-specific training, suggesting the gap is closable. The open question: can RT or Griffin reach the same level with scale and fine-tuning?

## Related Pages

- [relational-deep-learning](relational-deep-learning.md) — the RDL framework
- [relational-entity-graph](relational-entity-graph.md) — the graph representation all methods operate on
- [rt-vs-relgt](rt-vs-relgt.md) — detailed comparison of the two GT approaches
- [relational-foundation-model](relational-foundation-model.md) — the foundation model frontier
- [relbench](relbench.md) — benchmark used to evaluate all methods
