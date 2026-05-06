---
title: "PEARL: Learning Efficient Positional Encodings with Graph Neural Networks"
tags: [source, positional-encoding, graph-transformer, graph-neural-network]
sources: [kanatsoulis2025pearl]
updated: 2026-05-06
---

# PEARL: Learning Efficient Positional Encodings with Graph Neural Networks

**Source:** https://arxiv.org/abs/2502.01122
**Title:** Learning Efficient Positional Encodings with Graph Neural Networks
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Charilaos I. Kanatsoulis, Evelyn Choi, Stephanie Jegelka, Jure Leskovec, Alejandro Ribeiro
**Venue:** ICLR 2025

## Summary

- **What:** Existing graph PEs fail at least one of four key criteria — stability, expressiveness, scalability, and genericness — with eigenvector-based methods being expensive and random-walk methods being insufficiently expressive.
- **How:** PEARL shows that standard message-passing GNNs implicitly compute eigenvector mappings, and exploits this to produce PEs via GNN forward passes on randomly or basis-initialized node features — no eigendecomposition required.
- **So what:** PEARL satisfies all four criteria simultaneously, scales linearly in graph size, and directly motivates RelGT's stochastic subgraph GNN PE design.

## Challenges & Novelty

Graph positional encodings must balance four competing properties: stability (robust to graph perturbations), expressiveness (distinguishes non-isomorphic graphs), scalability (linear in N), and genericness (works on any graph). No prior PE satisfies all four: LapPE is expressive but O(N³); RWSE is scalable but less expressive; SignNet/SPE are expressive but still eigenvector-dependent.

- **Eigendecomposition cost:** LapPE and SPE require O(N³) eigendecomposition — infeasible at scale.
- **Structural symmetry collapse:** without identifiers, GNNs collapse structurally symmetric nodes to the same representation; random features break this symmetry cheaply.
- **Permutation equivariance from random init:** a single random initialization is not permutation equivariant; statistical pooling over multiple runs restores equivariance.

## Relation to Prior Work

| PE | Stable | Expressive | O(N) | Generic |
|---|---|---|---|---|
| LapPE | No | Yes | No | Yes |
| RWSE | Yes | Partial | Yes | Yes |
| SignNet / [lim2022signnet](lim2022signnet.md) | Yes | Yes | No | Yes |
| SPE / [huang2024stability](huang2024stability.md) | Yes | Yes | No | Yes |
| **PEARL (R-PEARL)** | Yes | Yes | Yes | Yes |

- [lim2022signnet](lim2022signnet.md): SignNet achieves expressiveness via sign-invariant eigenvector processing but still requires eigenvector computation — O(N²) or O(N³); PEARL achieves comparable expressiveness with linear complexity.
- [huang2024stability](huang2024stability.md): SPE achieves stability and expressiveness but requires eigenvectors; PEARL achieves both at O(N) via GNN.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT's subgraph GNN PE is a direct application of PEARL's principle on local subgraphs — stochastic resampling each training step instead of statistical pooling.

## Technical Details

**Core insight — GNNs as eigenvector mappings.** For any standard MPNN with MLP combination, k layers of message passing computes $\mathbf{S}^k\mathbf{x} = \mathbf{V}\Lambda^k\mathbf{V}^\top\mathbf{x}$, where $\mathbf{V}$ is the eigenvector matrix and $\mathbf{S}$ is the graph shift operator. An MLP on top can learn any nonlinear function of eigenvector components — making explicit eigendecomposition unnecessary.

**R-PEARL (random features, large graphs).** Each node gets a random identifier $\mathbf{q}^{(m)} \sim \mathcal{N}(0, I)$, independently drawn $M$ times. Random features break structural symmetries so the GNN can distinguish nodes. The number of runs $M$ needed is independent of graph size N, giving O(N) total complexity.

**B-PEARL (basis vectors, small graphs).** Node $m$ gets one-hot $\mathbf{e}_m$, so $M = N$ runs are needed — O(N²) complexity. Restricted to small graphs. Approximates SPE at lower cost.

**Statistical pooling for permutation equivariance.** A single random run breaks permutation equivariance. Fix: run $M$ independent initializations, take the empirical mean across runs:
$$\text{PE}(v) = \frac{1}{M}\sum_{m=1}^M \Phi(\mathcal{G}, \mathbf{q}^{(m)})[v]$$

The distribution of $\Phi(\mathcal{G}, \mathbf{q})$ is permutation equivariant, so any statistic (mean, variance) is also equivariant.

**Usage.** Drop-in PE: concatenate $[X, \text{PEARL}(\mathcal{G})]$ and feed to any downstream GNN/GT unchanged.

## Experiments

- R-PEARL matches SignNet/SPE expressiveness on graph distinguishing benchmarks at a fraction of the compute cost.
- B-PEARL closely approximates SPE on molecular benchmarks (ZINC, ogbg-molhiv) with significantly lower runtime.
- Linear scaling verified empirically on large graphs where LapPE is infeasible.

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-transformer](graph-transformer.md)
- [subgraph-gnn-pe](subgraph-gnn-pe.md)
