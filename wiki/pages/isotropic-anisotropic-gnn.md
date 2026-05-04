---
title: "Isotropic vs. Anisotropic GNNs"
tags: [query, analysis, concept, graph-neural-network, expressiveness]
sources: [gcn, gat, gin, alon2020bottleneck, graph-neural-network, hgt]
updated: 2026-05-04
---

# Isotropic vs. Anisotropic GNNs

The distinction is about **how a GNN weights its neighbors during aggregation** — whether the weight assigned to each neighbor depends on features, or is determined solely by graph structure.

## Isotropic GNNs

An isotropic GNN assigns the same aggregation weight to all neighbors, up to a fixed structural factor like degree. The weight is **independent of node features** — every neighbor is treated equally regardless of what it represents.

The canonical example is [gcn](gcn.md): its update rule is

$$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-\frac{1}{2}} \tilde{A}\tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)$$

Each neighbor's contribution is weighted only by $1/\sqrt{d_i \cdot d_j}$ (degrees), regardless of what those neighbors contain. [gcn](gcn.md) explicitly names this "isotropic aggregation" as both its simplicity and its limitation. [gin](gin.md) further shows that mean aggregation is not injective over multisets — GCN cannot distinguish graphs where neighbor feature distributions differ only in count — placing it strictly below WL expressive power.

**Examples:** GCN, GIN (sum aggregation is isotropic — all neighbors contribute equally via sum, no learned per-neighbor weighting).

## Anisotropic GNNs

An anisotropic GNN assigns **feature-dependent weights** to each neighbor. The importance of neighbor $j$ to node $i$ is a learned function of their features (and/or the edge between them), so the model can selectively suppress or amplify specific neighbors.

The canonical example is [gat](gat.md):

$$\alpha_{ij} = \text{softmax}_j\!\left(\text{LeakyReLU}\!\left(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]\right)\right)$$

The weight $\alpha_{ij}$ is computed from the concatenation of both nodes' features — different neighbors get different weights. [gat](gat.md) describes this as replacing "GCN's isotropic degree normalization with learned attention weights."

**Examples:** GAT, GGNN (gated GNN), HGT ([hgt](hgt.md) uses per-meta-relation attention keys/queries).

## Why the Distinction Matters

[alon2020bottleneck](alon2020bottleneck.md) identifies this as a key factor in **over-squashing** susceptibility: isotropic aggregators (GCN, GIN) cannot selectively gate out noisy long-range messages — they must compress all $\mathcal{O}(\exp(K))$ neighbors equally. Anisotropic aggregators (GAT, GGNN) can in principle down-weight irrelevant distant neighbors, making them less susceptible (though not immune). This is also reflected in [graph-neural-network](graph-neural-network.md)'s Limitations section.

Importantly, [gin](gin.md) notes that GAT's attention-weighted mean is *still* not injective — anisotropy improves over GCN but does not achieve WL-level expressive power, since mean aggregation (even weighted) cannot distinguish multisets that are scaled copies of each other.

## Summary Table

| | Isotropic | Anisotropic |
|---|---|---|
| Neighbor weights | Fixed (degree-based) | Learned (feature-based) |
| Examples | GCN, GIN | GAT, GGNN, HGT |
| Over-squashing | More susceptible | Less susceptible |
| Expressive power | GCN < WL; GIN = WL | GAT < WL (still) |
| Complexity | O(\|E\|) | O(\|E\|) — same order |

## Related Concepts

- [graph-neural-network](graph-neural-network.md) — the general MPNN framework in which this distinction applies
- [over-squashing](over-squashing.md) — isotropic models more susceptible; anisotropic models partially mitigate
- [graph-transformer](graph-transformer.md) — full global attention is the extreme anisotropic case: all-pairs feature-dependent weights
