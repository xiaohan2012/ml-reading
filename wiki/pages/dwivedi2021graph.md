---
title: "A Generalization of Transformer Networks to Graphs"
tags: [source, graph-transformer, positional-encoding, gnn]
sources: [dwivedi2021graph]
updated: 2026-04-29
---

# A Generalization of Transformer Networks to Graphs

**Source:** https://arxiv.org/abs/2012.09699
**Title:** A Generalization of Transformer Networks to Graphs
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Dwivedi, Bresson
**Venue:** AAAI 2021 Workshop (Deep Learning on Graphs)
**Year:** 2021

## Summary

Dwivedi and Bresson (NTU) propose the Graph Transformer (GT), a generalization of the original Transformer architecture to arbitrary graphs. The paper argues that two design axes are critical: **sparsity** and **positional encodings** — both stem from a fundamental mismatch between Transformers designed for sequences and the irregular, unordered structure of graphs.

**Why sparsity matters:** in NLP, full connectivity is justified because (a) word interactions are contextual and hard to pre-specify — any word can plausibly depend on any other — and (b) sentences are short (tens to hundreds of words), making $O(N^2)$ attention computationally feasible. In graph datasets, neither holds: the graph's edge structure *is* explicit domain knowledge (bonds, citations, friendships) that encodes meaningful sparse connectivity, and node counts can reach millions or billions. Ignoring this structure wastes a rich inductive bias and makes full attention computationally impossible at scale. The paper's sparse-vs-full ablation directly confirms this: sparse (neighborhood) attention outperforms full-graph attention on all three benchmark datasets.

**Why positional encodings matter:** most GNNs learn representations that are *structurally invariant* — they cannot distinguish a node's position within a graph from its structural role. This means two nodes with similar local neighborhoods but different global positions get identical representations. Simple attention-based models like GAT, which condition attention only on local connectivity without any positional signal, fail to achieve competitive performance on graph datasets for this reason. Without PE, a Transformer applied to a graph has no way to assign unique identities to nodes. LapPE — the $k$ smallest non-trivial eigenvectors of the normalized graph Laplacian — provides exactly this: nearby nodes receive similar eigenvector values (distance-aware), and the eigenvectors generalize the sinusoidal PE from NLP to arbitrary graph topology. Nodes with identical features but different graph positions can now be distinguished.

**Four architectural modifications from vanilla Transformer:**
1. *Neighborhood-sparse attention*: attention computed only over $\mathcal{N}(i)$, not all-to-all — exploiting graph sparsity as an inductive bias.
2. *Laplacian eigenvector PE (LapPE)*: pre-computed $k$ smallest non-trivial eigenvectors of the normalized graph Laplacian added to input node features; sign ambiguity handled by random sign-flipping during training.
3. *BatchNorm instead of LayerNorm*: empirically faster training and better generalization on graph datasets.
4. *Edge feature extension*: edge attributes multiply the raw attention score before softmax: $\hat{w}_{ij} = \left(\frac{Q_i \cdot K_j}{\sqrt{d}}\right) \cdot E \cdot e_{ij}$ — tying domain edge information directly to pairwise attention; edge representations maintained via a separate residual pipeline.

**Experiments.** Evaluated on ZINC (graph regression), PATTERN, CLUSTER (node classification) from Benchmarking-GNNs. GT with LapPE + BatchNorm outperforms GCN and GAT; the edge-feature variant closes the gap to GatedGCN on ZINC. The paper also demonstrates that sparse (neighborhood) attention dramatically outperforms full-graph attention — validating sparsity as the right inductive bias.

## Key Takeaways

- **Sparse neighborhood attention ≫ full-graph attention on graphs**: full connectivity loses the structural inductive bias and causes the model to overfit to PE/SE rather than learning local patterns.
- **LapPE is the canonical graph PE**: eigenvectors of the normalized graph Laplacian are the natural generalization of sinusoidal PE; they capture global positions (close nodes → similar PE) and are more informative than Graph-BERT's WL-PE.
- **BatchNorm > LayerNorm**: consistent empirical finding; best GT results used BatchNorm across all three datasets.
- **Edge features should multiply attention scores**: injecting `e_{ij}` into the raw attention score (before softmax) is more effective than using it only in the value stream; especially important for molecular graphs with bond-type features.
- **Historical significance**: this paper (along with [dwivedi2020benchmarking](dwivedi2020benchmarking.md)) established LapPE as the standard graph PE, which GraphGPS later categorized as "global PE" and compared to SignNet, RWSE, etc.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [positional-encoding](positional-encoding.md)

## Relation to Other Wiki Pages

- [graph-transformer](graph-transformer.md): this paper formalizes the basic GT layer used as the "Transformer+LapPE" baseline in GraphGPS and SAN.
- [graphgps](graphgps.md): GPS builds directly on this architecture; GraphGPS categorizes LapPE as "global PE" in its taxonomy, uses RWSE as a better practical alternative for molecular tasks, and shows that LapPE+SignNet outperforms LapPE alone.
- [positional-encoding](positional-encoding.md): LapPE introduced in Dwivedi & Bresson is the canonical global PE, later improved by SignNet and replaced by RWSE for structural tasks.
- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md): HGT references GT as its homogeneous counterpart — extends the meta-relation triplet idea to handle heterogeneous node/edge types that GT doesn't address.
