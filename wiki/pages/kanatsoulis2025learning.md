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

**Core insight — GNNs as eigenvector mappings:** for any standard message-passing GNN (GCN, GIN, GraphSAGE) with MLP combination, the update for node $v$ at layer $l$ can be written $\mathbf{x}_v^{(l)} = \text{MLP}(\mathbf{V}[v,:])$, where $\mathbf{V}$ is the eigenvector matrix. The intuition: $k$ layers of message passing computes $\mathbf{S}^k\mathbf{x} = \mathbf{V}\Lambda^k\mathbf{V}^\top\mathbf{x}$ — repeated multiplication by the graph shift operator is power iteration in the eigenvector basis. An MLP on top can learn any nonlinear function of those components. PEARL makes this explicit via initialization.

**Two initialization strategies:**

- **R-PEARL (random features, large graphs):** each node gets a random identifier $\mathbf{q}^{(m)} \sim \mathcal{N}(0,I)$, independently drawn $M$ times. Random features break structural symmetries so the GNN can distinguish nodes — without them, nodes with identical neighborhoods collapse. The number of samples $M$ needed is *independent of graph size $N$*, giving $O(N)$ complexity.

- **B-PEARL (standard basis vectors, small graphs):** node $m$ gets one-hot $\mathbf{e}_m$, so $M = N$ runs are needed — $O(N^2)$ complexity. Restricted to small graphs (molecular benchmarks with tens–hundreds of nodes). Approximates SPE ([huang2024stability](huang2024stability.md)) at much lower cost.

**Statistical pooling for permutation equivariance:** a single random run isn't permutation equivariant — relabeling nodes changes which node gets which seed. Fix: run $M$ independent initializations, get $M$ output matrices, then take the empirical mean across runs: $\text{PE}(v) = \frac{1}{M}\sum_{m=1}^M \Phi(\mathcal{G}, \mathbf{q}^{(m)})[v]$. The distribution of $\Phi(\mathcal{G}, \mathbf{q})$ is permutation equivariant, so any statistic derived from it (mean, variance, median) is also equivariant. PEARL uses mean for simplicity and convergence guarantees.

**Usage:** drop-in PE provider — same interface as LapPE/SignNet. Concatenate output with node features $[X, \text{PEARL}(\mathcal{G})]$ and feed to any downstream GNN/GT unchanged.

## Key Takeaways

- **4 criteria for good PE**: stability + expressiveness + scalability + genericness; PEARL satisfies all four; most prior methods fail ≥1.
- **GNNs as eigenvector mappings**: $k$ message-passing layers = power iteration in eigenvector basis; explicit eigendecomposition unnecessary.
- **Random init motivation**: breaks structural symmetries so GNN can distinguish nodes; $M$ samples needed is $O(1)$ in graph size → linear complexity.
- **Standard basis init**: one-hot per node; $O(N^2)$; for small graphs only; approximates SPE deterministically.
- **Statistical pooling**: empirical mean over $M$ runs restores permutation equivariance broken by random identifiers.
- **Drop-in PE**: $[X, \text{PEARL}(\mathcal{G})]$ — downstream model unchanged.
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
