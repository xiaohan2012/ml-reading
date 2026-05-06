---
title: "SAN: Rethinking Graph Transformers with Spectral Attention"
tags: [source, graph-transformer, positional-encoding, expressiveness]
sources: [kreuzer2021san]
updated: 2026-05-06
---

# SAN: Rethinking Graph Transformers with Spectral Attention

**Source:** https://arxiv.org/abs/2106.03893
**Title:** Rethinking Graph Transformers with Spectral Attention
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Devin Kreuzer, Dominique Beaini, Will Hamilton, Vincent Létourneau, Prudencio Tossou
**Venue:** NeurIPS 2021

## Summary

- **What:** Prior graph Transformers use only the k lowest Laplacian eigenvectors as PE, discarding higher-frequency structural information and ignoring eigenvalue ordering.
- **How:** SAN uses the *full Laplacian spectrum* as a learned positional encoding (LPE), processing each (eigenvalue, eigenvector component) pair with a small Transformer before full-graph attention.
- **So what:** SAN is provably more expressive than 1-WL GNNs and the first fully-connected GT to achieve competitive performance on standard benchmarks (ZINC, MNIST, CIFAR10, PATTERN).

## Challenges & Novelty

Partial LapPE (k lowest eigenvectors) loses high-frequency structural information and treats all eigenvectors uniformly regardless of their eigenvalue. Graph Transformers need structural encodings that are both maximally expressive and eigenvalue-aware. SAN treats the Laplacian spectrum as a function over the frequency axis, processing it with a Transformer that uses eigenvalues as positional signals.

- **Partial spectrum discards information:** using only k lowest eigenvectors misses substructures captured by higher frequencies (e.g., bipartiteness, triangle counts).
- **Sign ambiguity:** eigenvectors are defined up to sign; prior work ignores this; SAN processes pairs $(+v, -v)$ symmetrically to be sign-invariant.
- **Degenerate eigenvalues:** standard LapPE doesn't handle repeated eigenvalues (multigraphs, symmetric graphs); SAN models the full multiplicity structure.

## Relation to Prior Work

| Model | Spectrum used | Full attention | Sign-invariant |
|---|---|---|---|
| [dwivedi2021graph](dwivedi2021graph.md) | k lowest eigenvectors | No (MPNN) | No |
| **SAN** | Full spectrum | Yes | Yes |
| [ying2021graphormer](ying2021graphormer.md) | SPD-based spatial | Yes | N/A |
| [rampavsek2022graphgps](rampavsek2022graphgps.md) | RWSE / LapPE | Hybrid | Via SignNet |

- [dwivedi2021graph](dwivedi2021graph.md): the direct predecessor — uses k-lowest LapPE; SAN shows this loses structural information by not considering the full spectrum.
- [ying2021graphormer](ying2021graphormer.md): uses SPD-based spatial encoding instead of spectral; cheaper but less theoretically motivated than full spectrum.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GPS explicitly builds on SAN-style full-spectrum LapPE in its PE taxonomy, and later work (SignNet) addresses the sign-invariance issue SAN acknowledged.

## Technical Details

**LPE module.** Eigenvectors are viewed as vectors positioned on the eigenvalue axis. A small Transformer processes each (eigenvalue $\lambda_k$, eigenvector component $v_k^{(i)}$) pair for each node $i$:

$$\text{LPE}(i) = \text{Transformer}\left(\{(\lambda_k, v_k^{(i)})\}_{k=1}^{N}\right)$$

The eigenvalue acts as a positional signal (frequency); the component acts as the feature. The output is a per-node PE embedding added to node features before the main fully-connected Transformer.

**Full graph attention.** All N nodes attend to all N nodes — O(N²) — eliminating over-squashing from local message passing. This gives the model full-range information flow.

**Sign invariance.** SAN processes each eigenvector symmetrically with respect to sign flips, ensuring the PE is invariant to arbitrary sign choices in eigenvector computation.

**Expressiveness.** SAN provably exceeds 1-WL: the full Laplacian spectrum determines the graph up to isomorphism for almost all graphs; the LPE module can learn to distinguish non-isomorphic graphs that sparse MPNNs cannot.

## Experiments

- First fully-connected GT to match/outperform GNN baselines on ZINC, MNIST, CIFAR10, PATTERN benchmarks.
- Full spectrum consistently outperforms partial-k LapPE, especially on tasks requiring high-frequency structural features.
- O(N²) attention limits SAN to small-to-medium graphs; acknowledged as an open scalability problem (addressed by GPS).

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [positional-encoding](positional-encoding.md)
- [graph-neural-network](graph-neural-network.md)
