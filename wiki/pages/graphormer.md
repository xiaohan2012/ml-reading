---
title: "Graphormer: Do Transformers Really Perform Badly for Graph Representation?"
tags: [source, graph-transformer, positional-encoding, gnn, expressiveness]
sources: [graphormer]
updated: 2026-04-29
---

# Graphormer: Do Transformers Really Perform Badly for Graph Representation?

**Source:** https://arxiv.org/abs/2106.05234
**Title:** Do Transformers Really Perform Badly for Graph Representation?
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Ying, Cai, Luo, Zheng, Ke, He, Shen, Liu
**Venue:** NeurIPS 2021
**Year:** 2021

## Summary

Graphormer (Ying et al., NeurIPS 2021) answers whether Transformers can match or surpass GNNs on graph tasks with an emphatic yes — by encoding structural information explicitly into three complementary encodings rather than relying on message passing. The paper demonstrates that the standard Transformer fails on graphs primarily because it lacks structural inductive biases, not because attention is fundamentally unsuited to graphs.

The three structural encodings are: (1) **Centrality Encoding** — learnable in/out-degree embeddings added to node features at input; (2) **Spatial Encoding** — a learnable scalar bias `b_{phi(v_i,v_j)}` added to each attention logit, indexed by shortest-path distance `phi(v_i,v_j)` between node pairs; (3) **Edge Encoding** — a bias `c_ij = (1/N) sum_n x_{e_n} · w_n^E` averaged over edge features along the shortest path. Combined, each attention logit becomes: `A_ij = (h_i W_Q)(h_j W_K)^T / sqrt(d) + b_{phi(v_i,v_j)} + c_ij`. A special `[VNode]` is added (connected to all nodes) for graph-level readout, analogous to BERT's `[CLS]` token.

Graphormer provably subsumes GCN, GIN, and GraphSAGE as special cases and can distinguish graphs that 1-WL fails on. It achieves SOTA on OGB-LSC PCQM4M-LSC (quantum chemistry, 3.8M graphs), improving on GIN-VN by 11.5% relative MAE.

## Key Takeaways

- Three structural encodings (centrality, spatial, edge) inject graph structure as soft biases into standard Transformer attention — no architectural surgery required
- Full equation: `A_ij = (h_i W_Q)(h_j W_K)^T / sqrt(d) + b_{phi(v_i,v_j)} + c_ij`
- `b_{phi}` is a shared learnable scalar indexed by SPD; if learned to decrease with distance it recovers locality of GNNs
- Graphormer covers GCN/GIN/GraphSAGE as special cases; exceeds 1-WL expressiveness
- OGB-LSC PCQM4M-LSC validate MAE: 0.1234 (Graphormer 47M) vs 0.1395 (GIN-VN, previous SOTA), 11.5% relative improvement
- No over-smoothing observed — performance keeps improving with depth/width

## Entities & Concepts

- [graph-transformer](graph-transformer.md) — Graphormer is a major GT baseline alongside GraphGPS
- [positional-encoding](positional-encoding.md) — Spatial encoding via SPD; centrality encoding via degree
- [graph-neural-network](graph-neural-network.md) — Graphormer subsumes GCN/GIN/GraphSAGE as special cases

## Relation to Other Wiki Pages

- vs. [rampavsek2022graphgps](rampavsek2022graphgps.md): GPS uses MPNN ∥ GlobalAttn with separate PE injection; Graphormer bakes structure directly into attention biases. Both achieve global receptive field but by different mechanisms.
- vs. [dwivedi2021graph](dwivedi2021graph.md): Dwivedi & Bresson's GT uses LapPE; Graphormer uses SPD-based spatial encoding — more structurally informative and no sign ambiguity.
- vs. [san](san.md): SAN uses full Laplacian spectrum as LPE; Graphormer uses shortest-path distance. SAN focuses on theoretical expressiveness; Graphormer focuses on practical SOTA performance.
- vs. [gin](gin.md): GIN's sum aggregation achieves 1-WL; Graphormer exceeds 1-WL via spatial attention across all node pairs.
