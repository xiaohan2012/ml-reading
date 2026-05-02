---
title: "Temporal Graph Networks for Deep Learning on Dynamic Graphs (TGN)"
tags: [source, temporal-graph, dynamic-graph, memory, ctdg]
sources: [tgn]
updated: 2026-04-29
---

# Temporal Graph Networks for Deep Learning on Dynamic Graphs (TGN)

**Source:** https://arxiv.org/abs/2006.10637
**Title:** Temporal Graph Networks for Deep Learning on Dynamic Graphs
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Rossi, Chamberlain, Frasca, Eynard, Monti, Bronstein
**Venue:** ICML 2020 Workshop on Graph Representation Learning
**Year:** 2020

## Summary

Rossi, Chamberlain, Frasca, Eynard, Monti, and Bronstein (Twitter) introduce TGN — a generic, modular framework for deep learning on continuous-time dynamic graphs (CTDGs) represented as event streams. The key contribution is combining **per-node memory** (for long-term history) with **graph-based temporal aggregation** (for up-to-date local context), resolving the staleness problem that afflicts all prior methods.

**Framework modules.** TGN decomposes the encoder into five composable modules:

1. **Memory** $\mathbf{s}_i(t)$: a compact vector summarizing node $i$'s full interaction history. Initialized to zero; updated each time node $i$ is involved in an event.

2. **Message Function**: on edge $(i,j,t)$ computes messages for both source and destination from their memories, time elapsed, and edge features:
   $$\mathbf{m}_i(t) = \text{msg}_s(\mathbf{s}_i(t^-), \mathbf{s}_j(t^-), \Delta t, \mathbf{e}_{ij}(t))$$

3. **Message Aggregator**: collapses multiple messages in the same batch for one node. Options: *most-recent* (keep last) or *mean*.

4. **Memory Updater**: $\mathbf{s}_i(t) = \text{mem}(\bar{\mathbf{m}}_i(t), \mathbf{s}_i(t^-))$ — typically a GRU.

5. **Embedding Module**: generates the temporal embedding $\mathbf{z}_i(t)$ using the memory plus a graph-based aggregation over recent temporal neighbors:
   - *Identity*: $\mathbf{z}_i(t) = \mathbf{s}_i(t)$ — fastest, most stale
   - *Time projection* (Jodie-style): $(1 + \Delta t \mathbf{w}) \circ \mathbf{s}_i(t)$
   - *Temporal Graph Attention* (attn): multi-head attention over L-hop temporal neighborhood, each neighbor encoded as $[\mathbf{h}_j(t) \| \mathbf{e}_{ij} \| \phi(t - t_j)]$
   - *Temporal Graph Sum* (sum): faster linear alternative to attn

**Best configuration: TGN-attn** = GRU memory + 1-layer temporal graph attention + most-recent message aggregator. This is 30× faster per epoch than TGAT while outperforming it on all datasets.

**Why memory beats 2-hop attention.** With memory, each node's 1-hop neighbors carry compressed history from their past interactions — so 1 layer effectively reaches deeper into history than 2 layers without memory.

**Training trick (Raw Message Store).** To avoid information leakage, memory is updated from *previous-batch* raw messages before predicting the *current batch*. This means gradient flows through the memory update path without the model seeing the answer during prediction.

**Unification.** TGN is a strict generalization: TGAT = TGN with no memory + attn embedding; Jodie = TGN with GRU memory + time projection embedding; DyRep = TGN with GRU memory + graph attention in message function.

**Results.** SOTA on future edge prediction (transductive + inductive) and dynamic node classification on Wikipedia, Reddit, and Twitter. TGN-attn: Wikipedia transductive AP 98.46% (vs. TGAT 95.34%), inductive 97.81% (vs. TGAT 93.99%).

## Key Takeaways

- **Memory resolves staleness**: without memory, a node embedding goes stale between interactions; memory compresses all past history into a fixed-size vector that persists across batches.
- **1 layer + memory > 2 layers without**: adding the memory module to TGN makes the 2nd graph attention layer redundant — the memory acts as a pre-computed higher-hop summary.
- **Most-recent neighbor sampling beats uniform**: temporal graphs favor recency; sampling the $k$ most recent edges consistently outperforms uniform sampling (Figure in Appendix).
- **Raw Message Store is the training innovation**: memory modules need gradients but can't update with the current batch (leakage) — delay by one batch, store raw messages, apply before predicting.
- **Batch size matters**: larger batch = more stale memory for later items in batch; batch size 200 is the best tradeoff (not too large, not too slow).

## Entities & Concepts

- [temporal-graph](temporal-graph.md)

## Relation to Other Wiki Pages

- [temporal-graph](temporal-graph.md): TGN is the canonical CTDG learning framework; the memory + graph attention combination defines the state-of-the-art approach.
- [tgat](tgat.md): TGAT = TGN without memory; TGN paper shows memory alone adds ~4% AP, motivating the full TGN framework.
- [tgm-temporal-graph-modelling](tgm-temporal-graph-modelling.md): TGM library implements TGN as one of its models; the library's hook system mirrors TGN's modular design.
- [gat](gat.md): TGN's temporal graph attention embedding extends GAT attention to include time encodings φ(t−t_j) alongside neighbor features.
- [graphsage](graphsage.md): TGN's neighborhood sampling follows GraphSAGE's fixed-size sampling; but samples most-recent rather than uniform.
