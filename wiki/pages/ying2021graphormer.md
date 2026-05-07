---
title: "Graphormer: Do Transformers Really Perform Badly for Graph Representation?"
tags: [source, graph-transformer, positional-encoding, gnn, expressiveness]
sources: [ying2021graphormer]
updated: 2026-05-06
---

# Graphormer: Do Transformers Really Perform Badly for Graph Representation?

**Source:** https://arxiv.org/abs/2106.05234
**Title:** Do Transformers Really Perform Badly for Graph Representation?
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu
**Venue:** NeurIPS 2021

## Summary

- **What:** Standard Transformers fail on graphs not because attention is unsuitable, but because they lack structural inductive biases — graph topology is invisible to the model.
- **How:** Graphormer injects three complementary structural encodings (centrality, spatial, edge) as soft biases into standard Transformer attention, without any architectural surgery.
- **So what:** Graphormer subsumes GCN/GIN/GraphSAGE as special cases, exceeds 1-WL expressiveness, and achieves SOTA on OGB-LSC PCQM4M (11.5% relative MAE improvement over GIN-VN).

## Challenges & Novelty

Prior Graph Transformers either required custom architectures or struggled to scale to large molecular datasets. The core problem was not attention itself but the absence of graph-structural signals that GNNs receive implicitly through message passing. Graphormer shows that encoding topology as explicit attention biases is sufficient to unlock Transformer expressiveness on graphs.

- **No topology in attention:** Standard Transformers treat all nodes as equivalent tokens — no sense of which nodes are neighbors, far apart, or highly connected.
- **GNNs already encode structure implicitly:** Degree signals (centrality), hop distance (spatial), and edge attributes along paths (edge encoding) are what GNNs actually exploit; Graphormer injects these explicitly.
- **Scalability gap:** Prior spectral methods (SAN) used full Laplacian eigenvectors — expensive and graph-size-dependent; Graphormer's encodings are all O(N²) or cheaper.

## Relation to Prior Work

| Model | PE type | Full attention | Subsumes GNNs |
|---|---|---|---|
| GCN / GIN / GraphSAGE | None | No | — |
| [kreuzer2021san](kreuzer2021san.md) | Full Laplacian spectrum | Yes | No |
| [rampavsek2022graphgps](rampavsek2022graphgps.md) | RWSE / LapPE | Hybrid | Partially |
| **Graphormer** | Centrality + Spatial + Edge | Yes | Yes |

- [kreuzer2021san](kreuzer2021san.md): SAN uses full Laplacian spectrum for PE; Graphormer uses SPD-based spatial encoding — cheaper and more directly structural, less theoretically motivated.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GPS uses MPNN ∥ GlobalAttn with separate PE injection; Graphormer bakes structure directly into attention biases — both achieve global receptive field by different mechanisms.
- [dwivedi2021gt](dwivedi2021gt.md): earlier GT using k lowest LapPE eigenvectors; Graphormer replaces with SPD-based spatial encoding, more informative and without sign ambiguity.

## Technical Details

Three structural encodings are added to standard Transformer attention:

**1. Centrality Encoding** — learnable in/out-degree embeddings added to node features at input:
$$h_v^{(0)} = x_v + z_{\deg^-(v)}^- + z_{\deg^+(v)}^+$$

**2. Spatial Encoding** — learnable scalar bias added per attention logit, indexed by shortest-path distance $\phi(v_i, v_j)$:
$$A_{ij} = \frac{(h_i W_Q)(h_j W_K)^\top}{\sqrt{d}} + b_{\phi(v_i,v_j)}$$

**3. Edge Encoding** — bias from edge features along the shortest path between $i$ and $j$:
$$c_{ij} = \frac{1}{N}\sum_{n=1}^N x_{e_n} \cdot w_n^E$$

Combined attention logit:
$$A_{ij} = \frac{(h_i W_Q)(h_j W_K)^\top}{\sqrt{d}} + b_{\phi(v_i,v_j)} + c_{ij}$$

A special `[VNode]` (virtual node connected to all nodes) is added for graph-level readout, analogous to BERT's `[CLS]` token.

If $b_\phi$ is learned to decrease with distance, the model recovers locality of GNNs. Graphormer provably subsumes GCN, GIN, and GraphSAGE as special cases and can distinguish graphs that 1-WL fails on.

## Experiments

- Achieves SOTA on OGB-LSC PCQM4M-LSC: validate MAE 0.1234 (Graphormer 47M params) vs 0.1395 (GIN-VN), an 11.5% relative improvement.
- No over-smoothing observed — performance keeps improving with depth and width, unlike standard GNNs.
- Spatial encoding ($b_\phi$) alone accounts for the largest single gain in ablations.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [positional-encoding](positional-encoding.md)
- [graph-neural-network](graph-neural-network.md)
