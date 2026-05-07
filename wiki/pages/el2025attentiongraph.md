---
title: "Towards Mechanistic Interpretability of Graph Transformers via Attention Graphs"
tags: [source, graph-transformer, interpretability, gnn, attention]
sources: [el2025attentiongraph]
updated: 2026-05-07
---

# Towards Mechanistic Interpretability of Graph Transformers via Attention Graphs

**Source:** https://arxiv.org/abs/2502.12352
**Title:** Towards Mechanistic Interpretability of Graph Transformers via Attention Graphs
**Date ingested:** 2026-04-30
**Type:** paper
**Authors:** Batu El, Deepro Choudhury, Pietro Liò, Chaitanya K. Joshi
**Venue:** preprint 2025

## Summary

- **What:** Graph Transformer accuracy metrics don't reveal *how* the model routes information internally — two models with identical test accuracy can implement fundamentally different computational strategies, and it is unknown whether GTs learn to follow graph structure or ignore it.
- **How:** Construct an **Attention Graph** by aggregating learned attention matrices across heads (average) and across layers (matrix multiply) into a single directed graph that captures all indirect information routing; compare it to the input graph topology.
- **So what:** Reveals that pure Transformer models (DL class) recover less than 4% of input graph structure yet match accuracy; structurally-biased models (DLB class, e.g., Graphormer) recover 28–86%; same accuracy can be achieved via fundamentally different internal algorithms.

## Challenges & Novelty

Standard interpretability tools (GNNExplainer, gradient saliency) perturb inputs to identify important features — they don't analyze the information routing structure of the model itself. For GTs, which blend local MPNN computation with global attention, it is unclear whether the model learns to follow edges or discovers its own routing.

- **Attention Graph construction captures multi-layer information flow:** averaging across heads then matrix-multiplying across layers produces entry $(i, j)$ proportional to the total attention influence of node $j$ on node $i$ across all indirect paths — a compact summary of actual information routing.
- **Four GT variants classified by design:** SC (static, sparse = GCN-style), SL (sparse learned = GAT-style), DLB (dense learned with structural bias = Graphormer), DL (dense learned, no bias = pure Transformer). Each class exhibits distinct Attention Graph patterns.
- **Same accuracy, different algorithms:** the central finding is that mechanistic analysis is necessary — end-to-end metrics hide divergent computational strategies. DL models develop "reference nodes" (attention hubs that aggregate global context) while DLB models use self-attention diagonals (preserve own features).

## Relation to Prior Work

| GT class | Representative | Structure recovery | Strategy |
|---|---|---|---|
| SC (static sparse) | GCN ([kipf2017gcn](kipf2017gcn.md)) | High (by design) | Follow graph edges |
| SL (sparse learned) | GAT ([velickovic2018gat](velickovic2018gat.md)) | Moderate | Learned edge weights |
| DLB (dense + bias) | Graphormer ([ying2021graphormer](ying2021graphormer.md)) | 28–86% | Locality + bias |
| DL (dense) | Pure Transformer | $<$ 4% | Reference nodes |

- [ying2021graphormer](ying2021graphormer.md): Graphormer's SPD bias (DLB class) causes high structure recovery (locality decay) and intermediate behavior between local GNN and pure Transformer — Attention Graphs explain *why* the bias matters mechanistically.
- [kreuzer2021san](kreuzer2021san.md): SAN and Graphormer both enable full global attention; Attention Graphs distinguish their internal routing strategies despite similar benchmark performance.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GPS's hybrid MPNN + global attention would be a natural next subject for Attention Graph analysis — local vs. global component contributions not yet studied.
- [velickovic2018gat](velickovic2018gat.md): GAT (SL class) shows intermediate structure recovery; learned attention selectively follows edges but doesn't match the input graph perfectly.

## Technical Details

**Attention Graph construction.**

Step 1 — Head aggregation: for each layer $l$, average attention matrices across $H$ heads:
$$A^{(l)} = \frac{1}{H} \sum_{h=1}^H A_h^{(l)} \in \mathbb{R}^{N \times N}$$

Step 2 — Layer composition: multiply across $L$ layers to capture all indirect paths:
$$A^\text{total} = A^{(L)} \cdot A^{(L-1)} \cdots A^{(1)} \in \mathbb{R}^{N \times N}$$

Entry $[A^\text{total}]_{ij}$ reflects all indirect attention paths $i \to k \to \cdots \to j$ — the total influence of node $j$ on node $i$ across all layers.

**Structure recovery metric.** Compare $A^\text{total}$ to the adjacency matrix $\mathcal{A}$ of the input graph via:
$$\text{F1} = \text{F1}\!\left(\text{top-}|\mathcal{E}|(A^\text{total}),\; \mathcal{A}\right)$$

where $\text{top-}|\mathcal{E}|$ takes the $|\mathcal{E}|$ largest entries of $A^\text{total}$.

**Key Attention Graph patterns:**

1. *Locality decay* (DLB): $[A^\text{total}]_{ij}$ decreases with graph distance $\text{dist}(i, j)$ — model preferentially routes through short paths.
2. *Diagonal self-attention* (DLB on heterophilous graphs): $A^\text{total} \approx I$ — each node mostly attends to itself, preserving own features for classification.
3. *Reference nodes* (DL): a small set of nodes receive near-universal attention — they aggregate global context and serve as comparison anchors for all other nodes' predictions.
4. *Uniform attention* (DL base): $A^\text{total} \approx \frac{1}{N}\mathbf{1}\mathbf{1}^T$ — pure Transformer without structural bias distributes attention uniformly.

## Experiments

- Node classification benchmarks (Cora, CiteSeer, heterophilous graphs): all four GT classes achieve similar test accuracy (within 2%); Attention Graphs reveal fundamentally different internal routing.
- DL models: structure recovery F1 $< 4\%$; develop reference node patterns (2–5 nodes receive $> 50\%$ of total attention).
- DLB models (Graphormer): structure recovery F1 = 28–86% depending on dataset; higher on homophilous graphs.
- Heterophilous graphs: DLB models shift to diagonal self-attention (rely on own features); DL models shift to reference nodes (classify by comparison).

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
