---
title: "k-GNN: Weisfeiler and Leman Go Neural — Higher-order Graph Neural Networks"
tags: [source, gnn, expressiveness, wl-test, higher-order]
sources: [k-gnn]
updated: 2026-04-29
---

# k-GNN: Weisfeiler and Leman Go Neural

**Source:** https://arxiv.org/abs/1810.02244
**Title:** Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Morris, Ritzert, Fey, Hamilton, Lenssen, Rattan, Grohe
**Venue:** AAAI 2019
**Year:** 2019

## Summary

Morris, Ritzert, Fey, Hamilton et al. (AAAI 2019) provide a rigorous theoretical analysis of GNN expressiveness and propose **k-GNNs** to go beyond the 1-WL limit. The paper has two main contributions: (1) proving that standard GNNs (1-GNNs) are exactly as powerful as the 1-WL isomorphism test, and (2) constructing higher-order k-GNNs based on the k-WL algorithm that are strictly more powerful.

**1-WL / 1-GNN equivalence**: The 1-WL algorithm colors nodes by iteratively hashing (current_color, multiset_of_neighbor_colors). Any GNN with injective aggregation (e.g., GIN's sum+MLP) can simulate this; conversely, no GNN can do better. This result is parallel to GIN (Xu et al., ICLR 2019) but derived independently with a different proof approach.

**k-GNN**: Instead of coloring individual nodes, k-WL colors *k-tuples* of nodes. The j-th neighborhood of a k-tuple s = (s_1,...,s_k) is formed by replacing the j-th component with each node in V(G). The coloring update is: `c^t_{k,l}(s) = hash(c^{t-1}_{k,l}(s), (C^t_1(s), ..., C^t_k(s)))` where each C^t_j aggregates over j-th neighbors. k-GNNs implement this neurally: they perform message passing between k-tuples, not individual nodes.

**Hierarchical 1-k-GNNs**: the initial messages for k-GNN are initialized from the output of a 1-GNN (not random), enabling multi-scale representations combining fine-grained (node-level) and coarse-grained (tuple-level) structure.

## Key Takeaways

- Standard 1-GNNs ≡ 1-WL: cannot distinguish any pair of graphs that 1-WL cannot distinguish
- k-GNNs operate on k-tuples with j-th neighborhood message passing; strictly more powerful than 1-GNNs for k≥2
- Hierarchical 1-k-GNNs bootstrap from 1-GNN output — multi-granularity representations
- 54.45% average MAE reduction on QM9 vs 1-GNNs (12 regression tasks)
- Computational cost: O(n^k) node-tuples; practical variants use k=2 or k=3
- Key limitation: expressiveness gain comes at significant computational cost

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md) — k-GNN extends the WL-based expressiveness analysis from GIN to higher-order structures

## Relation to Other Wiki Pages

- [gin](gin.md): GIN and k-GNN both prove 1-GNN ≡ 1-WL, independently and contemporaneously. GIN's contribution focuses on the sum vs mean/max distinction; k-GNN's focuses on the higher-order generalization. Together they form the theoretical foundation of GNN expressiveness.
- [graphormer](graphormer.md): Graphormer exceeds 1-WL via its spatial attention (attending across all pairs) — an orthogonal approach to k-GNN's tuple-based message passing.
- [mpnn](mpnn.md): k-GNNs can be described in MPNN language where messages are passed between k-tuples rather than nodes.
- [graph-neural-network](graph-neural-network.md): The GNN expressiveness hierarchy: mean/max-GCN < sum-GIN = 1-WL < k-GNN for k≥2; this is the theoretical context for all architectural choices in the wiki.
