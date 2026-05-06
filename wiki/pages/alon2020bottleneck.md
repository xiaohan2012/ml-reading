---
title: "On the Bottleneck of Graph Neural Networks and its Practical Implications"
tags: [source, graph-neural-network, expressiveness, over-squashing]
sources: [alon2020bottleneck]
updated: 2026-05-04
---

# On the Bottleneck of Graph Neural Networks and its Practical Implications

**Source:** https://arxiv.org/abs/2006.05205
**Title:** On the Bottleneck of Graph Neural Networks and its Practical Implications
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Uri Alon, Eran Yahav
**Venue:** ICLR 2021

## Summary

This paper introduces and formalizes **over-squashing** — a fundamental limitation of message-passing GNNs on long-range tasks, distinct from the previously identified *over-smoothing* problem. The core argument: to capture interactions at radius $r$, a GNN needs $K \geq r$ layers, but the receptive field of each node grows exponentially with $K$ (as $\mathcal{O}(\exp(K))$). This means an exponentially-growing amount of information must be compressed into a fixed-size node vector at each aggregation step — the *bottleneck* — causing the GNN to squash out long-range signals and learn only short-range patterns from data.

![[Pasted image 20260506144004.png]]

Over-squashing is contrasted with over-smoothing: over-smoothing is a problem on short-range tasks (representations become indistinguishable with many layers), while over-squashing is the bottleneck on long-range tasks. Both phenomena explain why deeper GNNs don't improve with more layers, but for different reasons.

The paper demonstrates that GCN and GIN (uniform aggregation of neighbors) are more susceptible to over-squashing than GAT and GGNN (which can gate or weight incoming messages). A simple fix — adding a single **fully adjacent (FA) layer** that enables direct all-pairs communication — breaks the bottleneck and improves SOTA on QM9 (−42% error), ENZYMES (−12%), and VarMisuse without any additional hyperparameter tuning.

## Key Takeaways

- **Over-squashing** = information from exponentially-growing neighborhoods compressed into fixed-size vectors; long-range signals fail to propagate.
- **Distinct from over-smoothing**: over-smoothing → short-range tasks + too many layers; over-squashing → long-range tasks + structural bottleneck at aggregation.
- GCNs/GINs (isotropic aggregation) are more susceptible than GAT/GGNN (anisotropic), because they treat all neighbors equally and can't selectively gate out noise.
- **FA layer** (fully adjacent = all nodes attend to each other) breaks the bottleneck; this is the earliest motivation for global attention in Graph Transformers.
- Theoretical lower bound: hidden size required grows with the problem radius; fixing architecture without sufficient width can't resolve over-squashing.

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md)
- [graph-transformer](graph-transformer.md)
- [over-squashing](over-squashing.md)
- [over-smoothing](over-smoothing.md)

## Relation to Other Wiki Pages

- [graph-neural-network](graph-neural-network.md): over-squashing is a fundamental expressiveness limitation of all MPNNs; motivates moving beyond local aggregation.
- [graph-transformer](graph-transformer.md): global attention in GTs (Graphormer, SAN, GraphGPS) is directly motivated by the over-squashing bottleneck identified here — all-to-all attention eliminates the path-length constraint.
- [ying2021graphormer](ying2021graphormer.md): Graphormer's full global attention is one solution to over-squashing.
- [kreuzer2021san](kreuzer2021san.md): SAN's fully-connected attention also eliminates over-squashing.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GPS explicitly cites over-squashing as motivation for including global attention alongside local MPNN.
