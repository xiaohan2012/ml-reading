---
title: "SAN: Rethinking Graph Transformers with Spectral Attention"
tags: [source, graph-transformer, positional-encoding, expressiveness]
sources: [san]
updated: 2026-04-29
---

# SAN: Rethinking Graph Transformers with Spectral Attention

**Source:** https://arxiv.org/abs/2106.03893
**Title:** Rethinking Graph Transformers with Spectral Attention
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Kreuzer, Beaini, Hamilton, Létourneau, Tossou
**Venue:** NeurIPS 2021
**Year:** 2021

## Summary

The Spectral Attention Network (SAN) (Kreuzer, Beaini, Hamilton et al., NeurIPS 2021) provides a theoretically principled approach to graph Transformers by using the *full Laplacian spectrum* as a learned positional encoding (LPE). The central insight is that eigenvectors of the graph Laplacian are the natural graph-domain analogue of sine/cosine functions in Euclidean space — so using all of them (not just the k lowest-frequency ones) gives a maximally expressive positional encoding.

Eigenvectors are viewed not as columns of a matrix but as vectors positioned on the axis of eigenvalues (frequencies). The LPE module processes each eigenvector using a small Transformer that takes (eigenvalue, eigenvector_component) pairs as input, with the eigenvalue as a positional signal and the component as a feature. This produces a per-node positional embedding that is then added to node features before the main fully-connected Transformer. Crucially, the LPE is aware of eigenvalue multiplicities (degenerate eigenvalues), considers the full spectrum, and uses variable numbers of eigenvectors.

SAN uses **full graph attention** (all nodes attend to all nodes), which eliminates over-squashing and enables the model to capture long-range dependencies. By leveraging the full spectrum, SAN is provably more expressive than 1-WL GNNs and the first fully-connected architecture to achieve competitive performance on standard graph benchmarks.

## Key Takeaways

- **LPE (Learned Positional Encoding)**: small Transformer on (eigenvalue, eigenvector component) pairs → per-node PE; handles sign ambiguity via symmetric processing
- Full Laplacian spectrum used (not just k lowest eigenvectors); eigenvalue multiplicities handled
- Full graph attention (O(n²)) eliminates over-squashing present in GNNs
- Provably exceeds 1-WL expressiveness; can distinguish graphs that sparse MPNN fails on
- Property table: SAN is the only model with: local structure + global connectivity + structural PE + full spectrum + variable eigenvector count
- Eigenvalue invariant (norm-invariant) but NOT eigenvector-sign invariant (acknowledged limitation)
- First fully-connected graph model to match/outperform GNN baselines on ZINC, MNIST, CIFAR10, PATTERN

## Entities & Concepts

- [graph-transformer](graph-transformer.md) — SAN is a key GT with spectral PE
- [positional-encoding](positional-encoding.md) — SAN's LPE extends LapPE (introduced in [dwivedi2020benchmarking](dwivedi2020benchmarking.md)) to full spectrum with a learned processing module
- [graph-neural-network](graph-neural-network.md) — SAN exceeds 1-WL expressiveness

## Relation to Other Wiki Pages

- Directly informs [graphgps](graphgps.md): GPS's PE taxonomy explicitly cites SAN-style full-spectrum LapPE as the most expressive global PE; sign invariance issue (noted by SAN) is addressed by later work (Lim et al., ICLR 2022)
- vs. [graphormer](graphormer.md): Graphormer uses SPD-based spatial encoding; SAN uses spectral encoding. Both are fully-connected but encode graph structure differently.
- vs. [dwivedi2021graph](dwivedi2021graph.md): Dwivedi's GT uses k lowest eigenvectors as LapPE — SAN shows this loses structural information by not considering the full spectrum and eigenvalue ordering.
- [positional-encoding](positional-encoding.md): SAN is the key reference for the argument that full Laplacian spectrum > partial LapPE; GPS cites it when motivating RWSE over simple LapPE.
