---
title: "Hierarchical Transformer for Scalable Graph Learning"
tags: [source, graph-transformer, scalability, heterogeneous-graph]
sources: [zhu2023hierarchical]
updated: 2026-05-04
---

# Hierarchical Transformer for Scalable Graph Learning

**Source:** https://arxiv.org/abs/2305.02866
**Title:** Hierarchical Transformer for Scalable Graph Learning
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Wenhao Zhu, Tianyu Wen, Guojie Song, Xiaojun Ma, Liang Wang
**Venue:** arXiv 2023

## Summary

This paper addresses the challenge of applying Graph Transformers to large-scale graphs where full-batch global self-attention ($O(N^2)$) is infeasible. Standard sampling-based methods (mini-batch training) fail to capture high-level contextual information because they truncate the graph. The proposed solution is a **hierarchical Transformer** that coarsens the graph into multiple levels via clustering and applies attention at each granularity level, combining fine-grained local representations with coarse global context without full $O(N^2)$ attention.

The method is also designed for **heterogeneous graphs** (multiple node/edge types), addressing both scale and heterogeneity simultaneously. Each hierarchy level uses type-aware attention over nodes of different types within the cluster, with ring-level aggregation between hierarchical levels.

## Key Takeaways

- **Hierarchical clustering** replaces full global attention: cluster nodes → attend within clusters → aggregate clusters.
- Handles **heterogeneous graphs** (unlike most scalable GT work that focuses on homogeneous graphs).
- Avoids $O(N^2)$ without losing global context entirely — the hierarchy provides multi-scale context.
- Applicable to large graphs where sampling-based methods lose too much structural information.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)

## Relation to Other Wiki Pages

- [graph-transformer](graph-transformer.md): hierarchical clustering scalability strategy; complements sparse attention (Exphormer) and sampling (NAGphormer, Gophormer).
- [zhang2022hierarchical](zhang2022hierarchical.md): similar hierarchical approach; this paper additionally handles heterogeneous graphs.
- [hu2020hgt](hu2020hgt.md): HGT handles heterogeneity but not large-scale; this paper addresses both simultaneously.
