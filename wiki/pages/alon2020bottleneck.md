---
title: "On the Bottleneck of Graph Neural Networks and its Practical Implications"
tags: [source, graph-neural-network, expressiveness, over-squashing]
sources: [alon2020bottleneck]
updated: 2026-05-07
---

# On the Bottleneck of Graph Neural Networks and its Practical Implications

**Source:** https://arxiv.org/abs/2006.05205
**Title:** On the Bottleneck of Graph Neural Networks and its Practical Implications
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Uri Alon, Eran Yahav
**Venue:** ICLR 2021

## Summary

- **What:** GNNs fail on long-range tasks not only because of over-smoothing (too many layers → indistinguishable representations) but due to a distinct phenomenon: exponentially-growing neighborhoods must be compressed into fixed-size vectors at each aggregation step.
- **How:** Formalize **over-squashing** as information loss from exponential neighborhood growth; show that adding a single **fully-adjacent (FA) layer** (all-pairs attention) at the end of a standard GNN breaks the bottleneck.
- **So what:** The FA layer yields SOTA improvements on QM9 (−42% error), ENZYMES (−12%), and VarMisuse with no additional tuning; provides the earliest practical motivation for global attention in Graph Transformers.

## Challenges & Novelty

Over-smoothing was the dominant explanation for why deeper GNNs don't help. But over-smoothing is a problem of short-range tasks with too many layers — it doesn't explain why GNNs fail on long-range tasks even with appropriately many layers. This paper introduces over-squashing as the complementary failure mode.

- **Over-squashing is distinct from over-smoothing:** over-smoothing: representations converge, losing per-node distinctiveness (too many layers, short-range task); over-squashing: exponentially many nodes' information is compressed into a fixed-size vector, losing long-range signals (task radius $r$, graph with high-degree nodes).
- **Anisotropic aggregation is more resistant:** GCN/GIN treat all neighbors equally — they cannot selectively gate out noise from less informative neighbors and must compress everything. GAT/GGNN can weight neighbors and are empirically less susceptible.
- **FA layer as a direct fix:** the FA layer enables direct all-pairs communication, completely eliminating the path-length constraint. Because it appears only once (at the end), it doesn't cause over-smoothing — the local GNN layers maintain rich intermediate representations.

## Relation to Prior Work

| Phenomenon | Cause | Task type | Layer regime | Fix |
|---|---|---|---|---|
| Over-smoothing | Too many aggregation steps | Short-range | Too many layers | Residual connections, DropEdge |
| **Over-squashing** | Exponential neighborhood growth | Long-range | Too few (or appropriate) layers | FA layer / global attention |

- [xu2019gin](xu2019gin.md): GIN shows GNNs are bounded by 1-WL; over-squashing is a complementary limitation — even a maximally expressive 1-WL GNN fails on long-range tasks due to the bottleneck.
- [ying2021graphormer](ying2021graphormer.md): Graphormer's full global attention is one solution to over-squashing — all-pairs attention eliminates the path-length constraint entirely.
- [kreuzer2021san](kreuzer2021san.md): SAN's fully-connected attention also eliminates over-squashing; motivated by the same bottleneck argument.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GPS explicitly cites over-squashing as motivation for including global attention alongside local MPNN.
- [shirzad2023exphormer](shirzad2023exphormer.md): Exphormer's virtual global nodes are a sparse solution to over-squashing, providing global shortcuts at $O(N)$ cost.

## Technical Details

**Over-squashing formalization.** For a node $v_0$ that needs information from node $v_r$ at distance $r$, a GNN with $K < r$ layers cannot propagate this information at all. With $K = r$ layers, the Jacobian $\partial h_{v_0}^K / \partial x_{v_r}^0$ decreases exponentially with $r$ because each aggregation step multiplies by the normalized adjacency $\hat{A}$ — for high-degree graphs, this factor is small, and $r$ multiplications produce exponentially small gradients.

Formally, for a message-passing GNN: $\left|\frac{\partial h_{v_0}^{(K)}}{\partial x_{v_r}^{(0)}}\right| \leq c_r \cdot \prod_{l=0}^{r-1} \left\|\hat{A}_{v_l}\right\|$, where $\hat{A}_{v_l}$ involves the degree normalization — this product decreases exponentially in the minimum degree along the path.

**Fully-adjacent (FA) layer.** A single standard Transformer self-attention layer added after $K$ MPNN layers:

$$h_v^{(K+1)} = \text{MultiheadAttn}\!\left(h_v^{(K)},\, \{h_u^{(K)} : u \in V\}\right)$$

No masking — every node attends to every other. This bypasses all path-length constraints in one step.

**Why not all FA layers?** Multiple FA layers cause over-smoothing and lose the structural inductive bias. One FA layer at the end captures long-range dependencies while preserving local GNN representations.

## Experiments

- QM9 molecular property prediction: GCN + FA layer reduces mean absolute error by 42% on average across 13 quantum targets vs. GCN alone.
- ENZYMES protein graph classification: +12% accuracy with FA layer over baseline GNN.
- VarMisuse code understanding (long-range pointer task): FA layer essential; GNN alone fails on sequences requiring references across hundreds of tokens.
- GAT + FA $>$ GCN + FA: GAT's anisotropic aggregation already partially mitigates over-squashing, so the FA layer provides a smaller marginal improvement.

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md)
- [graph-transformer](graph-transformer.md)
