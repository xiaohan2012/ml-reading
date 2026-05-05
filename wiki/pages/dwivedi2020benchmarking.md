---
title: "Benchmarking Graph Neural Networks"
tags: [source, graph-neural-network, positional-encoding, benchmark]
sources: [dwivedi2020benchmarking]
updated: 2026-05-04
---

# Benchmarking Graph Neural Networks

**Source:** https://arxiv.org/abs/2003.00982
**Title:** Benchmarking Graph Neural Networks
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Vijay Prakash Dwivedi, Chaitanya K. Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, Xavier Bresson
**Venue:** JMLR 2022 (arXiv 2020)

## Summary

This paper establishes an open-source benchmarking framework for GNNs built around a curated collection of 12 medium-scale datasets spanning real-world domains (chemistry: ZINC, AQSOL; computer vision: MNIST, CIFAR10; social/academic: WikiCS, OGBL-COLLAB) and mathematical graphs designed to test specific structural properties (PATTERN, CLUSTER, CSL, CYCLES, TSP, GraphTheoryProp). The central design principle is a fixed parameter budget (100k or 500k parameters) applied uniformly to all models, enabling fair architectural comparison and separating modeling gains from raw capacity gains. The PyTorch/DGL infrastructure is modular, making it easy to swap datasets, layers, or training protocols.

The paper's most consequential contribution is the introduction of **Laplacian eigenvectors as graph positional encodings (LapPE)**: nodes lack canonical positions, so Laplacian eigenvectors of the graph Laplacian matrix serve as a structural signal analogous to sinusoidal PEs in sequence Transformers. LapPE substantially improved performance on structural detection tasks (CSL, CYCLES, PATTERN) where message-passing GNNs fail entirely. This proposal catalyzed a wave of PE research — SAN, SignNet, GraphGPS, Graphormer, and RelGT all trace their PE lineage here, whether building on LapPE directly or proposing alternatives to overcome its limitations (sign ambiguity, O(N³) precomputation, heterogeneous inapplicability).

## Key Takeaways

- **12 medium-scale datasets** covering graph/node/edge tasks across chemistry, vision, and mathematical graph theory — enabled fair academic-scale GNN comparison.
- **Fixed parameter budget** (100k/500k): separates architectural gains from capacity gains; standard for GT papers since 2020.
- **Introduced LapPE**: Laplacian eigenvectors as node positional encodings — the origin of the PE research direction now central to all Graph Transformers.
- **Origin of a field**: SAN, GraphGPS, SignNet, Graphormer, LSPE all cite this as the PE starting point they build on or improve.
- **Historical note**: the benchmark pre-dates Graph Transformers; the PE insight emerged from noticing that message-passing GNNs fail on structural tasks without positional information.

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-neural-network](graph-neural-network.md)
- [graph-transformer](graph-transformer.md)

## Relation to Other Wiki Pages

- [positional-encoding](positional-encoding.md): LapPE was introduced here — this is the origin page for the PE concept in graphs.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GraphGPS benchmarks against this framework's datasets and extends PE to the RWSE/SignNet era.
- [san](san.md): SAN (Kreuzer 2021) builds on LapPE, using the full Laplacian spectrum to address sign ambiguity.
- [graphormer](graphormer.md): Graphormer proposes SPD-based spatial encoding as an alternative to LapPE.
- [dwivedi2021graph](dwivedi2021graph.md): Dwivedi & Bresson (AAAI 2021) formalizes GTs using LapPE from this benchmark.
- [relational-graph-transformer](relational-graph-transformer.md): RelGT explicitly benchmarks Laplacian PE on REGs (finds it expensive and underperforms subgraph GNN PE).
