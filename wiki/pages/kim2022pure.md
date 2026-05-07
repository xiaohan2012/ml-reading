---
title: "TokenGT: Pure Transformers are Powerful Graph Learners"
tags: [source, graph-transformer, positional-encoding, expressiveness]
sources: [kim2022pure]
updated: 2026-05-07
---

# TokenGT: Pure Transformers are Powerful Graph Learners

**Source:** https://arxiv.org/abs/2207.02505
**Title:** Pure Transformers are Powerful Graph Learners
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Jinwoo Kim, Dat Nguyen, Seonwoo Min, Sungjun Cho, Moontae Lee, Honglak Lee, Seunghoon Hong
**Venue:** NeurIPS 2022

## Summary

- **What:** All prior Graph Transformers add graph-specific inductive biases (sparse masks, structural encodings, graph-specific positional encodings) to standard Transformers — but it is unclear whether these modifications are necessary or if a pure Transformer with the right token design suffices.
- **How:** Treat every node and every edge as an independent token; assign each node a unique orthonormal identifier (either orthogonal random features or Laplacian eigenvectors) concatenated into both node and edge tokens so the Transformer can infer graph connectivity.
- **So what:** Provably at least as expressive as 2-IGN (strictly more than all MPNNs); competitive with graph-specific Transformers on PCQM4Mv2 (3.7M molecules) while being compatible with efficient Transformer variants.

## Challenges & Novelty

Existing Graph Transformers modify attention masking, add graph PEs as features, or inject graph structure into attention biases — all of which prevent direct use of efficient Transformer engineering (linear attention, flash attention, etc.). TokenGT shows that graph structure can be encoded entirely in the token design, leaving the Transformer architecture untouched.

- **Both nodes and edges are tokens:** edge tokens carry the identifiers of both endpoints — the Transformer can infer which nodes an edge connects via dot-products between node identifiers. This is sufficient for the model to learn any permutation-equivariant function on the graph.
- **Orthonormal node identifiers give provable expressiveness:** with orthonormal $\mathbf{P}_v \in \mathbb{R}^{d_p}$, a Transformer over the tokenized graph can approximate any permutation-equivariant linear operator — equivalent to 2-IGN, which is strictly more expressive than 1-WL (all MPNNs).
- **ORF-based IDs are structure-agnostic yet sufficient:** orthogonal random features (random orthonormal vectors per node) do not encode graph structure — yet the full tokenization (node + edge tokens) still allows the Transformer to learn to distinguish nodes by their incident edge pattern, achieving >1-WL power.

## Relation to Prior Work

| Model | Token type | PE type | Expressiveness | Efficient attn. |
|---|---|---|---|---|
| [dwivedi2021gt](dwivedi2021gt.md) | Node only | LapPE (added to features) | 1-WL | No (sparse mask) |
| [ying2021graphormer](ying2021graphormer.md) | Node only | SPD bias | $>$ 1-WL | No (dense) |
| [rampavsek2022graphgps](rampavsek2022graphgps.md) | Node only | RWSE/SignNet | $>$ 1-WL | Partial |
| **TokenGT** | Node + Edge | Orthonormal IDs | $\geq$ 2-IGN $>$ 1-WL | Yes |

- [ying2021graphormer](ying2021graphormer.md): Graphormer encodes structure via attention biases (SPD, centrality); TokenGT encodes it via token embeddings — same expressiveness target, different mechanism.
- [dwivedi2021gt](dwivedi2021gt.md): GT adds LapPE to node features in a sparse attention Transformer; TokenGT uses node IDs in a pure global-attention Transformer.
- [lim2022signnet](lim2022signnet.md): LapPE-based node IDs (one choice in TokenGT) require SignNet-style invariant processing; ORF-based IDs avoid this issue entirely.

## Technical Details

**Token construction.** For a graph $G = (\mathcal{V}, \mathcal{E})$:

- *Node token* for $v$: $[\mathbf{X}_v\|\mathbf{E}^\mathcal{V},\; \mathbf{P}_v\|\mathbf{P}_v]$ — node features + type ID, then two copies of node identifier (source and target, both $= \mathbf{P}_v$ for node tokens)
- *Edge token* for $(u, v)$: $[\mathbf{X}_{(u,v)}\|\mathbf{E}^\mathcal{E},\; \mathbf{P}_u\|\mathbf{P}_v]$ — edge features + type ID, then source identifier $\mathbf{P}_u$ and target identifier $\mathbf{P}_v$

**Connectivity inference.** A standard Transformer attention head can compute the inner product $[\mathbf{P}_u\|\mathbf{P}_v] \cdot [\mathbf{P}_k\|\mathbf{P}_k]^T = \mathbf{P}_u \cdot \mathbf{P}_k + \mathbf{P}_v \cdot \mathbf{P}_k$. With orthonormal IDs, this equals:
- 2 if $k \in \{u, v\}$ and $u = v$ (self-loop)
- 1 if $k \in \{u, v\}$ (edge is incident to $k$)
- 0 otherwise

So the Transformer can recover the incidence structure from inner products alone — no attention masking needed.

**Node identifier choices:**
1. *Orthogonal random features (ORF):* sample a random orthogonal matrix; assign each node a column. Structure-agnostic, but the full tokenization still encodes graph structure via edge tokens.
2. *Laplacian eigenvectors:* LapPE as node identifier. Structure-aware, empirically better, but requires eigendecomposition and sign invariance (use SignNet or random flipping).

**Expressiveness result.** With orthonormal node IDs, a sufficiently large Transformer over node+edge tokens can approximate any permutation-equivariant linear map on graphs — this is equivalent to 2-IGN expressive power. 2-WL $>$ 1-WL, so TokenGT strictly exceeds all MPNNs.

## Experiments

- PCQM4Mv2 (3.7M molecular graphs, MAE prediction): TokenGT outperforms all GNN baselines (GCN, GIN, PNA, GatedGCN); competitive with Graphormer-Slim and GRPE.
- ORF-based IDs outperform GNNs; LapPE-based IDs further outperform ORF on most tasks.
- Naturally supports linear attention (Performer-style) without any structural modifications — demonstrating the engineering advantage of keeping the Transformer architecture pure.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [positional-encoding](positional-encoding.md)
- [graph-neural-network](graph-neural-network.md)
