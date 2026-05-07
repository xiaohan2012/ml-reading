---
title: "Hierarchical Transformer for Scalable Graph Learning"
tags: [source, graph-transformer, scalability, heterogeneous-graph]
sources: [zhu2023hierarchical]
updated: 2026-05-07
---

# Hierarchical Transformer for Scalable Graph Learning

**Source:** https://arxiv.org/abs/2305.02866
**Title:** Hierarchical Transformer for Scalable Graph Learning
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Wenhao Zhu, Tianyu Wen, Guojie Song, Xiaojun Ma, Liang Wang
**Venue:** arXiv 2023

## Summary

- **What:** Full-graph $O(N^2)$ global attention is infeasible for large graphs; mini-batch sampling truncates the graph and loses high-level contextual information; standard scalable GTs don't handle heterogeneous node/edge types.
- **How:** Coarsen the graph into a hierarchy via clustering, apply type-aware Transformer attention within each cluster at each hierarchy level, and propagate representations between levels via ring aggregation.
- **So what:** Achieves scalable GT on both large and heterogeneous graphs without full $O(N^2)$ attention; competitive accuracy on node classification benchmarks.

## Challenges & Novelty

Sampling-based methods (mini-batch training) truncate the graph — nodes can only see $k$-hop neighborhoods, losing coarse community structure and long-range dependencies. Existing heterogeneous GTs (HGT, HHGT) don't address scale. Combining heterogeneity with scalable attention requires addressing both type-specific parameterization and the clustering structure simultaneously.

- **Hierarchical clustering preserves multi-scale context:** by coarsening the graph into $L$ levels, the model sees fine-grained local structure at level 1 and coarse global structure at level $L$ — without computing $O(N^2)$ all-pairs attention at any level.
- **Type-aware attention within clusters handles heterogeneity:** within each cluster, attention is parameterized by node/edge type (as in HGT's meta-relation triplets) — different types attend to each other with different learned projections.
- **Ring aggregation propagates context between levels:** cluster-level representations are passed upward via a ring (sequential aggregation across cluster boundaries), then broadcast downward to inform lower-level node updates.

## Relation to Prior Work

| Model | Heterogeneous | Scalable | Approach | Long-range |
|---|---|---|---|---|
| [hu2020hgt](hu2020hgt.md) | Yes | No ($O(N^2)$) | Meta-relation attn | No |
| [zhu2025hhgt](zhu2025hhgt.md) | Yes | Partial | Ring-level hierarchy | Limited |
| [zhang2022hierarchical](zhang2022hierarchical.md) | No | Yes | Adaptive sampling + hierarchy | Yes |
| **This paper** | Yes | Yes | Clustering hierarchy | Yes |

- [hu2020hgt](hu2020hgt.md): HGT handles heterogeneity but requires full-graph attention; this paper extends the heterogeneous GT idea to large graphs via hierarchical clustering.
- [zhang2022hierarchical](zhang2022hierarchical.md): similar hierarchical approach but for homogeneous graphs with importance-adaptive sampling; this paper adds heterogeneity handling.
- [zhu2025hhgt](zhu2025hhgt.md): HHGT uses ring-distance hierarchy for heterogeneous HINs; this paper uses clustering hierarchy — both address heterogeneity + structure but with different hierarchical decompositions.

## Technical Details

**Hierarchical construction.** Apply graph clustering (METIS or Louvain) recursively to produce $L$ levels:
- Level 0: original graph nodes $\{v_1, \ldots, v_N\}$
- Level $l$: cluster nodes $\{C_1^{(l)}, \ldots, C_{K_l}^{(l)}\}$ where $K_l \ll K_{l-1}$

Cluster node representation at level $l$: $\mathbf{h}_{C_k}^{(l)} = \text{pool}(\{\mathbf{h}_v^{(l-1)} : v \in C_k\})$ (mean or max pooling).

**Type-aware attention within clusters.** For heterogeneous graphs with node types $\mathcal{T}$ and edge types $\mathcal{R}$:

$$\text{Attn}(i, j) = \frac{(Q_{\tau(i)} \mathbf{h}_i)(K_{\tau(j)} \mathbf{h}_j)^T}{\sqrt{d}} \cdot M_{\tau(i), r, \tau(j)}$$

where $\tau(i)$ is node $i$'s type, $r$ is the edge type, and $M$ is a type-relation-specific scaling matrix (analogous to HGT).

**Ring aggregation between levels.** Cluster representatives at level $l$ are aggregated into rings (sequences around cluster boundaries) and passed to level $l+1$. Level-$l+1$ context is broadcast back to level-$l$ nodes as a conditioning signal.

## Experiments

- ogbn-arxiv (homogeneous, 170K nodes): hierarchical Transformer matches or outperforms standalone sampling-based GTs.
- ACM/DBLP (heterogeneous, ~10K nodes): hierarchical + heterogeneous attention outperforms HGT by 1.5–3% node classification accuracy.
- Scalability: handles graphs with 1M+ nodes with 4GB GPU memory by limiting cluster sizes to 256 nodes per attention window.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
