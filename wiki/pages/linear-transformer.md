---
title: Linear Transformer
tags: [concept, attention, scalability]
sources: [rampavsek2022graphgps, wu2023sgformer, dwivedi2023graph]
updated: 2026-05-06
---

# Linear Transformer

## What It Is

A linear transformer replaces the O(N²) softmax attention with a kernel decomposition that reduces complexity to O(N). The core idea: find a feature map φ such that the attention score can be factored as an inner product:

$$\exp(q_i^\top k_j) \approx \phi(q_i)^\top \phi(k_j)$$

Then the attention output for query i becomes:

$$o_i = \frac{\phi(q_i)^\top \bigl(\sum_j \phi(k_j)\, v_j^\top\bigr)}{\phi(q_i)^\top \bigl(\sum_j \phi(k_j)\bigr)}$$

The bracketed sums $S = \sum_j \phi(k_j) v_j^\top$ and $z = \sum_j \phi(k_j)$ are shared across all queries — computed once in O(Nd) — reducing total complexity from O(N²d) to **O(Nd) = O(N)**.

## Variants

**Performer** (Choromanski et al., ICLR 2021): uses random Fourier features to approximate the Gaussian (softmax) kernel unbiasedly:

$$\phi(x) = \frac{1}{\sqrt{m}} \bigl[\exp(\omega_1^\top x), \ldots, \exp(\omega_m^\top x)\bigr], \quad \omega_i \sim \mathcal{N}(0, I)$$

With enough samples m, $\mathbb{E}[\phi(q)^\top \phi(k)] = \exp(q^\top k / \sqrt{d})$ exactly. Used in [rampavsek2022graphgps](rampavsek2022graphgps.md) as the global attention stream in GPS layers; reported within ~1% of full Transformer on ZINC.

**BigBird** (Zaheer et al., NeurIPS 2020): uses structured sparse attention (local window + random + global tokens) rather than a kernel approximation — O(N) via sparsity, not factorization. Also used in [rampavsek2022graphgps](rampavsek2022graphgps.md) as an alternative to Performer.

**SGFormer kernel attention** ([wu2023sgformer](wu2023sgformer.md)): uses a simplified kernel (ELU-based or direct normalization) without the softmax approximation — a different functional form rather than an approximation of softmax. Achieves 141× faster inference on medium-sized graphs.

## In the Graph Transformer Context

[rampavsek2022graphgps](rampavsek2022graphgps.md) is the main source that makes linear attention practically useful for graph learning: edge features only go to the MPNN stream (not global attention), which means the global stream can freely use linear attention without losing edge information. This design choice is what enables O(N+E) overall GPS complexity.

[dwivedi2023graph](dwivedi2023graph.md) includes "linear attention" as one of four scalability strategies in its taxonomy (alongside sparse attention, hierarchical coarsening, and neighborhood sampling).

[wu2023sgformer](wu2023sgformer.md) pushes this further: a single global kernel attention layer + one GNN layer is sufficient for SOTA on graphs up to 111M nodes.

## Gaps

- Performer paper (Choromanski et al., ICLR 2021) not yet ingested
- BigBird paper (Zaheer et al., NeurIPS 2020) not yet ingested
- SGFormer page does not specify which exact kernel formula is used
