---
title: "Towards Mechanistic Interpretability of Graph Transformers via Attention Graphs"
tags: [source, graph-transformer, interpretability, gnn, attention]
sources: [attention-graphs-gt-interpretability]
updated: 2026-04-30
---

# Towards Mechanistic Interpretability of Graph Transformers via Attention Graphs

**Source:** https://arxiv.org/abs/2502.12352
**Date ingested:** 2026-04-30
**Type:** paper
**Authors:** Batu El, Deepro Choudhury, Pietro Liò, Chaitanya K. Joshi
**Venue:** preprint 2025

## Summary

This paper introduces **Attention Graphs**, a mechanistic interpretability framework for GNNs and Graph Transformers. The core insight is that message passing in GNNs and self-attention in Transformers are mathematically equivalent — both propagate information along weighted edges. Attention Graphs make this explicit by aggregating learned attention matrices into a single directed graph that represents actual information routing, independently of the input graph topology.

Aggregation works in two steps: (1) **across heads** — simple averaging; (2) **across layers** — matrix multiplication, so that the entry (i,j) in the final Attention Graph reflects all indirect paths (i→k→j across layers). The resulting graph is compared to the original input graph to measure whether the model learned to follow the graph structure or deviate from it.

Experiments on node classification tasks reveal that identical accuracy can be achieved via fundamentally different internal strategies. Dense-learned (DL) models recover <4% of input graph structure (F1), while models with graph structural bias (DLB, e.g. Graphormer) recover 28–86%. On heterophilous graphs, DLB models rely heavily on self-attention (diagonal attention pattern = feature identity preservation), while DL models develop "reference nodes" — hubs receiving near-universal attention that act as comparison anchors for classification.

## Key Takeaways

- **Attention Graph construction**: average across heads, multiply across layers → encodes full multi-hop information flow in one matrix
- **Four GT variants** classified by parametrization × sparsity: SC (GCN), SL (GAT), DLB (Graphormer), DL (pure Transformer)
- **Structure recovery**: DL models learn attention nearly orthogonal to input graph (<4% F1); DLB models retain moderate-to-high alignment (28–86%)
- **Attention distance decay**: DLB concentrates on 1-hop neighbors (sharp locality); DL distributes uniformly across all distances regardless of graph connectivity
- **Heterophilous strategies diverge**: DLB → diagonal self-attention (rely on own features); DL → vertical "reference node" patterns (classify by comparing to anchor nodes)
- **Same accuracy, different algorithms**: mechanistic analysis is necessary — end-to-end metrics hide divergent computational strategies

## Entities & Concepts

- [graph-transformer](graph-transformer.md) — Attention Graphs apply to GTs across the SC/SL/DLB/DL design space
- [graphormer](graphormer.md) — DLB archetype; spatial bias retains graph structure in attention
- [gat](gat.md) — SL archetype; sparse learned attention
- [gcn](gcn.md) — SC archetype; constant sparse attention

## Relation to Other Wiki Pages

- vs. [graphormer](graphormer.md): Graphormer's SPD bias (graph structural bias) causes DLB-type behavior — high structure recovery, locality decay. Attention Graphs give a new lens on *why* this bias matters mechanistically.
- vs. [san](san.md): SAN and Graphormer both enable full global attention; Attention Graphs distinguish their information routing strategies despite similar benchmark performance.
- vs. [rampavsek2022graphgps](rampavsek2022graphgps.md): GPS hybrid (MPNN ∥ GlobalAttn) would be interesting to analyze via Attention Graphs — local vs. global component contributions not yet studied.
- **Gap**: No wiki page yet on GNN-specific explainability methods (GNNExplainer, PGExplainer, SubgraphX). This paper is complementary — it analyzes emergent attention routing rather than perturbing inputs.
