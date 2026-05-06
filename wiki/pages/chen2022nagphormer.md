---
title: "NAGphormer: A Tokenized Graph Transformer for Node Classification in Large Graphs"
tags: [source, graph-transformer, scalability]
sources: [chen2022nagphormer]
updated: 2026-05-06
---

# NAGphormer: A Tokenized Graph Transformer for Node Classification in Large Graphs

**Source:** https://arxiv.org/abs/2206.04910
**Title:** NAGphormer: A Tokenized Graph Transformer for Node Classification in Large Graphs
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Jinsong Chen, Kaiyuan Gao, Gaichao Li, Kun He
**Venue:** ICLR 2023

## Summary

- **What:** Full-graph Graph Transformers are O(N²) and ego-graph approaches (Gophormer) require subgraph sampling at training time, both limiting scalability to large graphs.
- **How:** NAGphormer's Hop2Token module pre-aggregates neighborhood features at each hop into a fixed-length per-node token sequence, letting a Transformer attend between hop levels rather than between nodes — with no graph access at training time.
- **So what:** NAGphormer scales to graphs with millions of nodes, achieves mini-batch training over independent nodes, and outperforms decoupled GNNs (SIGN, SGC) by capturing cross-hop interactions.

## Challenges & Novelty

Scalable GTs must eliminate O(N²) attention, but most alternatives either sacrifice expressiveness (linear Transformers) or still require graph access during training (ego-graph sampling). NAGphormer decouples graph access entirely from training by pre-computing hop-level aggregations, enabling pure i.i.d. mini-batching.

- **O(N²) attention:** full GT is infeasible at scale; ego-graph sampling reduces this but still needs graph access per training step.
- **Decoupled GNNs ignore cross-hop interactions:** SIGN and SGC pre-aggregate features at each hop but concatenate them — treating hops as independent features rather than a sequence with inter-hop dependencies.
- **Subgraph sampling variance:** Gophormer's stochastic ego-graph samples introduce training variance; NAGphormer avoids this with deterministic pre-aggregation.

## Relation to Prior Work

| Model | Graph access at training | Attention scope | Hop interaction |
|---|---|---|---|
| SIGN / SGC | Pre-compute only | None | No (concat) |
| [zhao2021gophormer](zhao2021gophormer.md) | Yes (sampling) | Nodes in subgraph | Yes |
| **NAGphormer** | Pre-compute only | Between hops (per node) | Yes |
| [dwivedi2025relgt](dwivedi2025relgt.md) | Yes (sampling) | Nodes in subgraph | Yes (via attention) |

- [zhao2021gophormer](zhao2021gophormer.md): Gophormer attends over nodes within a sampled subgraph; NAGphormer attends over hop levels for a single node — fundamentally different scopes.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT is closer to Gophormer (subgraph node attention); NAGphormer's hop-token idea is a possible future direction for REGs but doesn't handle heterogeneity or temporality.

## Technical Details

**Hop2Token.** For each node $v$, pre-aggregate neighborhood features at each hop:
$$h_v^{(k)} = \text{mean}_{u \in \mathcal{N}^k(v)} x_u, \quad k = 0, 1, \ldots, K$$

This produces a sequence of $K+1$ tokens per node: $[h_v^{(0)}, h_v^{(1)}, \ldots, h_v^{(K)}]$. Pre-computation is a one-time cost; training requires no graph access.

**Transformer over hops.** A standard Transformer processes the $K+1$-token sequence for each node independently, capturing interactions between different hops (e.g., how 1-hop and 3-hop neighborhoods relate). This gives a multi-scale receptive field without node-level attention.

**Training.** Nodes are mini-batched independently — each node's token sequence is self-contained. This enables exact i.i.d. sampling with no graph dependencies at training time.

## Experiments

- Outperforms SIGN, SGC, and other decoupled GNNs on large-scale benchmarks by capturing cross-hop interactions that simple concatenation misses.
- Competitive with ego-graph GTs (Gophormer) while being strictly more scalable — no graph access during training.
- K (number of hops) is the key hyperparameter; K=3–5 typically sufficient for most benchmarks.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
