---
title: "TokenGT: Pure Transformers are Powerful Graph Learners"
tags: [source, graph-transformer, positional-encoding, expressiveness]
sources: [kim2022pure]
updated: 2026-05-04
---

# TokenGT: Pure Transformers are Powerful Graph Learners

**Source:** https://arxiv.org/abs/2207.02505
**Title:** Pure Transformers are Powerful Graph Learners
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Jinwoo Kim, Dat Nguyen, Seonwoo Min, Sungjun Cho, Moontae Lee, Honglak Lee, Seunghoon Hong
**Venue:** NeurIPS 2022

## Summary

TokenGT argues against graph-specific architectural modifications to Transformers and instead applies a **standard Transformer directly to graphs** by treating every node and every edge as an independent token. The key insight is that a plain Transformer cannot understand graph structure from node/edge features alone — it needs structural information in the token embeddings. TokenGT provides this via two token-wise embeddings:

1. **Node identifiers (orthonormal)**: each node is assigned an orthonormal vector $\mathbf{P}_v \in \mathbb{R}^{d_p}$. A node token gets $[\mathbf{X}_v, \mathbf{P}_v, \mathbf{P}_v]$; an edge token $(u,v)$ gets $[\mathbf{X}_{(u,v)}, \mathbf{P}_u, \mathbf{P}_v]$. The Transformer can infer connectivity via dot-products between node identifiers (e.g., $[\mathbf{P}_u, \mathbf{P}_v][\mathbf{P}_k, \mathbf{P}_k]^\top = 1$ iff $k \in \{u,v\}$). Two practical choices: *orthogonal random features* (ORF, structure-agnostic) or *Laplacian eigenvectors* (structure-aware, better empirically).
2. **Type identifiers (trainable)**: two learned vectors $\mathbf{E}^\mathcal{V}$ and $\mathbf{E}^\mathcal{E}$ distinguish node tokens from edge tokens.

**Theoretical result**: with orthonormal node identifiers, a Transformer over these tokens can approximate any permutation-equivariant linear operator on a graph — making it at least as expressive as a 2nd-order Invariant Graph Network (2-IGN), hence strictly more expressive than all message-passing GNNs. Extension to hypergraphs: order-$k$ generalized embeddings achieve $k$-IGN / $k$-WL expressiveness.

**Empirical**: on PCQM4Mv2 (3.7M molecular graphs), TokenGT outperforms all GNN baselines and is competitive with graph-specific Transformer variants (Graphormer, GRPE). Naturally supports linear attention (kernel attention), unlike graph-specific models that hardcode sparse structure.

## Key Takeaways

- **Pure Transformer + orthonormal node IDs = ≥ 2-IGN expressiveness** — no message-passing, no graph-specific attention masking required.
- Nodes and edges are both first-class tokens; edge tokens carry source/target node identifiers, so the Transformer sees the full incidence structure.
- ORF-based node IDs are structure-agnostic yet still outperform GNNs on large-scale tasks; LapPE-based IDs are better (structure-aware).
- Freely compatible with efficient Transformer variants (linear attention) — a key advantage over graph-specific architectures.
- Theoretical expressiveness: 2-WL > 1-WL (all MPNNs), achievable with a simple token embedding choice.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [positional-encoding](positional-encoding.md)
- [graph-neural-network](graph-neural-network.md)

## Relation to Other Wiki Pages

- [graph-transformer](graph-transformer.md): TokenGT is the "pure Transformer" endpoint — minimal graph-specific bias, maximum compatibility with standard Transformer engineering.
- [positional-encoding](positional-encoding.md): node identifiers are a form of PE; LapPE-based IDs connect to [dwivedi2020benchmarking](dwivedi2020benchmarking.md); ORF-based IDs are a novel alternative.
- [ying2021graphormer](ying2021graphormer.md): Graphormer also uses full global attention but encodes structure via attention biases (SPD, centrality); TokenGT encodes it via token embeddings.
- [graph-neural-network](graph-neural-network.md): TokenGT proves it surpasses all MPNNs in expressiveness via 2-IGN equivalence.
