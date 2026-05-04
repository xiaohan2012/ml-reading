---
title: Over-Squashing
tags: [concept, graph-neural-network, expressiveness]
sources: [alon2020bottleneck, shirzad2023exphormer, wu2023sgformer, san, zhang2022hierarchical, mao2023hinormer]
updated: 2026-05-04
---

# Over-Squashing

## Description

Over-squashing is a fundamental limitation of message-passing GNNs on tasks that require long-range interaction. To capture signals at graph-distance $r$, a GNN needs $K \geq r$ layers. But the receptive field of each node grows exponentially with $K$ — $\mathcal{O}(\exp(K))$ nodes must be aggregated into a single fixed-size vector. This bottleneck forces the GNN to "squash" exponentially many messages into a constant-dimensional representation, causing long-range signals to be lost.

**Formal intuition:** the Jacobian $\partial \mathbf{h}_v^{(K)} / \partial \mathbf{x}_u$ — sensitivity of node $v$'s representation to distant node $u$ — shrinks exponentially with graph distance, making it practically impossible to propagate information from far-away nodes.

**Contrast with over-smoothing:** over-smoothing occurs in *deep* GNNs on *short-range* tasks — representations become indistinguishable across nodes as layers accumulate. Over-squashing occurs regardless of depth on *long-range* tasks — it is a structural bottleneck at each aggregation step, not a depth problem.

**Which GNNs are most susceptible:** isotropic aggregators (GCN, GIN — uniform neighbor averaging) are more susceptible than anisotropic ones (GAT, GGNN — learned gates/weights), because they cannot selectively suppress noisy long-range messages.

**Proposed fixes:**
- *FA layer* ([alon2020bottleneck](alon2020bottleneck.md)): add a single fully-adjacent layer enabling all-pairs communication; breaks the bottleneck; ↓42% error on QM9.
- *Global attention* (Graph Transformers): bypass aggregation entirely; all nodes interact directly. Used in [san](san.md), [graphormer](graphormer.md), [graphgps](graphgps.md), [shirzad2023exphormer](shirzad2023exphormer.md), [wu2023sgformer](wu2023sgformer.md), [mao2023hinormer](mao2023hinormer.md).
- *Graph rewiring*: add long-range edges (e.g., expander graph) to shorten paths; used in [shirzad2023exphormer](shirzad2023exphormer.md).
- *Hierarchical aggregation*: coarsen the graph to reduce effective diameter; used in [zhang2022hierarchical](zhang2022hierarchical.md).

## Appearances in Sources

- [alon2020bottleneck](alon2020bottleneck.md) — **origin paper**: formalized over-squashing; FA layer fix; ICLR 2021
- [san](san.md) — full-graph Laplacian attention bypasses over-squashing by design
- [shirzad2023exphormer](shirzad2023exphormer.md) — expander graph edges shorten paths to reduce over-squashing
- [wu2023sgformer](wu2023sgformer.md) — single global attention layer as the minimal fix for over-squashing at scale
- [zhang2022hierarchical](zhang2022hierarchical.md) — hierarchical coarsening reduces effective graph diameter
- [mao2023hinormer](mao2023hinormer.md) — cites over-squashing as motivation for adding global Transformer to local HIN encoder

## Related Concepts

- [over-smoothing](over-smoothing.md) — the complementary depth-related limitation (distinct problem, different regime)
- [graph-neural-network](graph-neural-network.md) — the architecture family where over-squashing occurs
- [graph-transformer](graph-transformer.md) — global attention as the standard architectural fix
