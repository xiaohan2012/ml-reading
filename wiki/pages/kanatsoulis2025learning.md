---
title: "PEARL: Learning Efficient Positional Encodings with Graph Neural Networks"
tags: [source, positional-encoding, graph-transformer, graph-neural-network]
sources: [kanatsoulis2025learning]
updated: 2026-05-04
---

# PEARL: Learning Efficient Positional Encodings with Graph Neural Networks

**Source:** https://arxiv.org/abs/2502.01122
**Title:** Learning Efficient Positional Encodings with Graph Neural Networks
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Charilaos I. Kanatsoulis, Evelyn Choi, Stephanie Jegelka, Jure Leskovec, Alejandro Ribeiro
**Venue:** ICLR 2025

## Summary

PEARL identifies four properties that graph PEs should satisfy: **(1) stability** (small graph perturbations → small PE changes), **(2) expressive power** (can distinguish non-isomorphic graphs), **(3) scalability** (linear complexity in graph size), and **(4) genericness** (applicable to any graph without special structure). Existing eigenvector-based methods fail on at least one: full LapPE has quadratic cost and instability; SignNet/SPE are expressive but still cubic; RWSE is cheap but less expressive.

PEARL's core insight: **message-passing GNNs are nonlinear mappings of eigenvectors**. Rather than explicitly computing eigenvectors and processing them with invariant networks, PEARL uses a GNN with specially chosen initializations to implicitly learn PE that achieves all four properties. Nodes are initialized with either **random features** (enabling sign equivariance via averaging) or **standard basis vectors** (one-hot per node, enabling structural separation). Statistical pooling across multiple random initializations maintains permutation equivariance while extracting structural information.

PEARL achieves linear complexity in graph size — one or two orders of magnitude cheaper than full eigenvector-based PEs — while matching their expressive performance and being provably stable. The method is directly cited in RelGT (as `kanatsoulis2025learning`) as the theoretical foundation for RelGT's subgraph GNN PE with random resampled initialization.

## Key Takeaways

- **4 criteria for good PE**: stability + expressiveness + scalability + genericness; PEARL satisfies all four; most prior methods fail ≥1.
- **GNNs as eigenvector mappings**: avoids explicit eigendecomposition; instead, GNN + random initialization implicitly computes eigenvector-equivalent representations.
- **Linear complexity**: 1–2 orders of magnitude cheaper than full LapPE / SignNet / SPE at comparable or better performance.
- **Random initialization + pooling**: multiple random seeds → average over permutations → equivariant in expectation; same principle as RelGT's subgraph GNN PE.
- Directly cited as theoretical foundation for RelGT's [subgraph-gnn-pe](subgraph-gnn-pe.md).

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-transformer](graph-transformer.md)
- [subgraph-gnn-pe](subgraph-gnn-pe.md)

## Relation to Other Wiki Pages

- [positional-encoding](positional-encoding.md): PEARL provides a scalable, stable, expressive PE — the closest to satisfying all desiderata simultaneously.
- [subgraph-gnn-pe](subgraph-gnn-pe.md): RelGT's subgraph GNN PE with random resampled initialization is a direct application of PEARL's principle on local subgraphs.
- [lim2022sign](lim2022sign.md): SignNet achieves expressiveness but at high cost; PEARL achieves comparable expressiveness with linear complexity.
- [huang2024stability](huang2024stability.md): SPE achieves stability but at eigenvector-computation cost; PEARL achieves both stability and linear complexity via GNN.
- [relational-graph-transformer](relational-graph-transformer.md): RelGT cites PEARL as the learnable PE approach that motivates their random-resampled subgraph GNN PE design.
