---
title: "SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations"
tags: [source, graph-transformer, scalability]
sources: [wu2023sgformer]
updated: 2026-05-07
---

# SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations

**Source:** https://arxiv.org/abs/2306.10759
**Title:** SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Qitian Wu, Wentao Zhao, Chenxiao Yang, Hengrui Zhang, Fan Nie, Haitian Jiang, Yatao Bian, Junchi Yan
**Venue:** NeurIPS 2023

## Summary

- **What:** Standard Graph Transformers stack multiple full-attention layers — which is both computationally prohibitive at scale and theoretically unnecessary when local GNN layers already encode the graph's structural inductive bias.
- **How:** Use a single $O(N)$-complexity global attention layer (kernel attention approximation) combined with one GNN layer for local structure — "one global, one local" architecture.
- **So what:** SOTA on large-scale benchmarks including ogbn-papers100M (111M nodes); up to 141× faster inference than multi-layer Transformer baselines.

## Challenges & Novelty

Deep Transformer stacks are motivated by sequence modeling where no inductive bias exists — the model must learn all structure from attention alone. For graphs, the local adjacency already encodes the primary structural inductive bias. Stacking many global attention layers on top of this structure is over-parameterized and wastes compute.

- **Graph structure reduces effective complexity:** because GNN layers handle local structure, a single global attention layer only needs to capture long-range dependencies (complementary, not redundant). This enables minimal architecture — one global layer is sufficient.
- **Linear kernel attention enables $O(N)$ global attention:** the kernel trick approximates $\text{softmax}(QK^T)V \approx \phi(Q)(\phi(K)^T V)$ — computing $\phi(K)^T V$ first is $O(N \cdot d^2)$ rather than $O(N^2 \cdot d)$. This makes all-pairs attention viable at 111M nodes.
- **Empirical evidence that one layer suffices:** ablation comparing 1 vs. $L$ global attention layers shows marginal improvement beyond $L=1$ on all tested benchmarks — depth is less important than breadth (global reach) for graph tasks.

## Relation to Prior Work

| Model | # Global attn layers | Complexity | Max scale tested |
|---|---|---|---|
| Graphormer ([ying2021graphormer](ying2021graphormer.md)) | $L$ (full) | $O(LN^2)$ | ~130K nodes |
| GPS ([rampavsek2022graphgps](rampavsek2022graphgps.md)) | $L$ (full / Performer) | $O(LN^2)$ or $O(LN)$ | ~600K nodes |
| Exphormer ([shirzad2023exphormer](shirzad2023exphormer.md)) | $L$ (sparse) | $O(LN)$ | ~2.4M nodes |
| **SGFormer** | 1 (linear kernel) | $O(N)$ | 111M nodes |

- [rampavsek2022graphgps](rampavsek2022graphgps.md): GPS uses one MPNN + one global attention per layer, stacked $L$ times; SGFormer uses one GNN + one global attention total — SGFormer is the extreme simplification of the GPS paradigm.
- [alon2020bottleneck](alon2020bottleneck.md): SGFormer's single global attention layer is motivated by the same over-squashing argument as the FA layer — one global hop resolves all long-range bottlenecks.
- [shirzad2023exphormer](shirzad2023exphormer.md): Exphormer uses structured sparse attention over $L$ layers; SGFormer uses unstructured linear approximation in 1 layer — two different philosophies for achieving $O(N)$ global attention.

## Technical Details

**Architecture.** Two components in sequence:

1. **GNN layer** (local): any standard MPNN (e.g., GCN, SAGE) applied once. Handles local neighborhood aggregation.

$$\mathbf{h}_v^{(1)} = \text{GNN}\!\left(\mathbf{x}_v, \{\mathbf{x}_u : u \in \mathcal{N}(v)\}\right)$$

2. **Global attention layer** (linear kernel): standard all-pairs attention approximated via kernel trick.

$$\text{Attn}(Q, K, V) \approx \phi(Q) \cdot \left(\phi(K)^T V\right)$$

where $\phi(\mathbf{x}) = \text{ELU}(\mathbf{x}) + 1$ (Performer-style feature map). Computing $\phi(K)^T V \in \mathbb{R}^{d \times d}$ first costs $O(Nd^2)$; then $\phi(Q)(\phi(K)^T V)$ costs $O(Nd^2)$ — total $O(Nd^2)$ per node, $O(N^2)$ avoided entirely.

**Output:** $\mathbf{h}_v^{(2)} = \text{GlobalAttn}(\mathbf{h}_v^{(1)}, \{\mathbf{h}_u^{(1)} : u \in V\})$

**Training.** Mini-batch over nodes; the global attention layer uses random sub-sampling during training ($M \ll N$ nodes per batch) — approximates full global attention cheaply. At inference, use the full graph.

**Why one GNN + one global layer:** theoretical analysis shows that one GNN layer achieves the 1-WL expressiveness bound on local structure; one global attention layer achieves arbitrary-range dependency capture. Additional layers provide diminishing returns under this decomposition.

## Experiments

- ogbn-papers100M (111M nodes, 1.6B edges): SGFormer achieves competitive accuracy in 10% of the GPU hours required by prior Transformer baselines.
- ogbn-products (2.4M nodes): 141× faster inference than SIGN-Transformer (multi-layer GT); 1.2% accuracy improvement.
- Ablation: 1 global layer vs. $L \in \{2, 3, 4\}$ — marginal accuracy improvement beyond $L=1$; exponential compute cost increase.
- Kernel attention approximation error: measured as $\|QK^T V - \phi(Q)(\phi(K)^T V)\|_F / \|QK^T V\|_F < 3\%$ on all benchmarks — acceptable for graph tasks.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
