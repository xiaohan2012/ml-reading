---
title: "Gophormer: Ego-Graph Transformer for Node Classification"
tags: [source, graph-transformer, scalability]
sources: [zhao2021gophormer]
updated: 2026-05-06
---

# Gophormer: Ego-Graph Transformer for Node Classification

**Source:** https://arxiv.org/abs/2110.13094
**Title:** Gophormer: Ego-Graph Transformer for Node Classification
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Jianan Zhao, Chaozhuo Li, Qianlong Wen, Yiqi Wang, Yuming Liu, Hao Sun, Xing Xie, Yanfang Ye
**Venue:** arXiv 2021

## Summary

- **What:** Full-graph Graph Transformers require O(N²) attention and cannot be mini-batched for large graphs, blocking scalable node classification.
- **How:** Gophormer samples an ego-graph (local neighborhood subgraph) per target node and applies full Transformer attention within that subgraph, enabling standard mini-batch training with progressive label propagation and consistency regularization.
- **So what:** Gophormer scales GT to large graphs and establishes the ego-graph subgraph-sampling paradigm that RelGT later adapts for relational entity graphs.

## Challenges & Novelty

Graph Transformers with full O(N²) attention cannot scale to large graphs used in real node classification tasks. The key insight is that full-graph attention is unnecessary — a node's label depends primarily on its local neighborhood, so restricting attention to a sampled ego-graph suffices while enabling standard GPU mini-batching.

- **O(N²) attention barrier:** applying full GT attention to millions of nodes is infeasible; prior scalable GTs (like linear Transformers) sacrifice expressiveness.
- **Stochastic sampling variance:** different ego-graph samples for the same node produce different representations; naive averaging degrades training.
- **Label underutilization:** GNNs exploit label propagation; pure GT designs ignore label context available at training time.

## Relation to Prior Work

| Model | Scalability strategy | Attention scope | Node interaction |
|---|---|---|---|
| Full GT ([kreuzer2021san](kreuzer2021san.md)) | None (O(N²)) | Full graph | All pairs |
| **Gophormer** | Ego-graph sampling | Local subgraph | Nodes in subgraph |
| [chen2022nagphormer](chen2022nagphormer.md) | Hop pre-aggregation | Per-node sequence | Between hops only |
| [dwivedi2025relgt](dwivedi2025relgt.md) | K-neighbor sampling | Local subgraph | Nodes in subgraph |

- [chen2022nagphormer](chen2022nagphormer.md): NAGphormer takes a different approach — pre-aggregates hop-level features into a sequence and attends between hops, not between nodes; no subgraph sampling needed.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT's K=300 subgraph sampling is the direct RDL extension of Gophormer's ego-graph paradigm, adding heterogeneous and temporal dimensions.

## Technical Details

**Ego-graph sampling.** For each target node $v$, sample a fixed-size neighborhood subgraph (ego-graph) within L hops. Apply multi-head self-attention to all nodes within this subgraph — O(K²) per node where K is ego-graph size.

**Progressive label propagation.** At each training step, use labels from already-trained nodes to enrich representations of unlabeled target nodes via label propagation over the graph. This provides global label context without requiring full-graph attention.

**Consistency regularization.** Multiple stochastic ego-graph samples of the same node should produce similar representations. A regularization loss penalizes discrepancy between different samples of the same node:
$$\mathcal{L}_{\text{cons}} = \sum_v \|h_v^{(1)} - h_v^{(2)}\|^2$$

where $h_v^{(1)}, h_v^{(2)}$ are representations from two independent ego-graph samples.

## Experiments

- Outperforms GNN baselines on several large-scale node classification benchmarks while enabling mini-batch training.
- Consistency regularization provides consistent gains by reducing variance from stochastic sampling.
- Ego-graph size (L, K) is the primary hyperparameter tradeoff: larger ego-graphs improve accuracy but increase compute per node.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
