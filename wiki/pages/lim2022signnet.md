---
title: "SignNet / BasisNet: Sign and Basis Invariant Networks for Spectral Graph Representation Learning"
tags: [source, positional-encoding, graph-transformer, expressiveness]
sources: [lim2022signnet]
updated: 2026-05-07
---

# SignNet / BasisNet: Sign and Basis Invariant Networks for Spectral Graph Representation Learning

**Source:** https://arxiv.org/abs/2202.13013
**Title:** Sign and Basis Invariant Networks for Spectral Graph Representation Learning
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Derek Lim, Joshua Robinson, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, Stefanie Jegelka
**Venue:** ICLR 2022

## Summary

- **What:** LapPE has two fundamental symmetry problems: (1) sign flips — eigenvectors can be negated arbitrarily by the numerical solver, creating $2^k$ valid sign combinations for $k$ eigenvectors; (2) basis symmetries — degenerate eigenvalues have infinitely many valid orthonormal bases for their eigenspace.
- **How:** SignNet handles sign flips via DeepSets-style symmetric aggregation $\rho(\phi(v) + \phi(-v))$ per eigenvector; BasisNet handles the more general basis symmetry by processing entire eigenspaces with an $O(d)$-invariant architecture.
- **So what:** Both networks are provably universal under their respective symmetries and strictly more expressive than all prior spectral methods; SignNet is adopted by GraphGPS as the PE of choice in its best-performing configurations.

## Challenges & Novelty

The naive fix for sign ambiguity — random sign-flipping as data augmentation — fails in practice: on some base models (PNA, GINE), LapPE with random flipping doesn't improve over no PE at all. The fundamental issue is that a network trained on $+v$ and $-v$ interchangeably cannot learn functions that meaningfully use eigenvector directions. Basis symmetries (for degenerate eigenvalues) were not addressed at all by prior work.

- **DeepSets symmetry gives provable sign invariance:** $\rho(\phi(v) + \phi(-v))$ is symmetric in $+v$ and $-v$ by construction — any function of the form $g(v) = g(-v)$ can be expressed this way. The $+$ and $-$ contributions are both always present, so the network can use the eigenvector's information without being sensitive to sign choice.
- **BasisNet handles full basis freedom:** for a degenerate eigenspace of dimension $d$, any $Q \in O(d)$ rotation of the eigenvector basis is equally valid. BasisNet processes the entire eigenspace simultaneously with an architecture that is equivariant to $O(d)$ rotations — using a set function over the eigenvectors in the degenerate subspace.
- **Universal approximation under symmetry constraints:** SignNet can represent any continuous sign-invariant function of eigenvectors; BasisNet can represent any continuous function invariant to basis rotations within degenerate eigenspaces. These are the tightest possible expressiveness guarantees under the symmetry constraints.

## Relation to Prior Work

| PE Method | Sign ambiguity | Basis ambiguity | Universality | Expressiveness |
|---|---|---|---|---|
| LapPE ([dwivedi2020benchmarking](dwivedi2020benchmarking.md)) | Random flip (breaks) | Not addressed | No | Limited |
| SAN ([kreuzer2021san](kreuzer2021san.md)) | Partially addressed | Not addressed | No | Spectral |
| **SignNet** | Solved (provably) | Not addressed | Yes (sign-inv.) | $>$ all spectral |
| **BasisNet** | Solved | Solved (provably) | Yes (basis-inv.) | $>$ SignNet |
| SPE ([huang2024stability](huang2024stability.md)) | Solved | Solved + stable | Yes | $\geq$ SignNet |

- [dwivedi2020benchmarking](dwivedi2020benchmarking.md): SignNet directly addresses the sign ambiguity problem in LapPE introduced there; same benchmark datasets used for validation.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GraphGPS explicitly adopts SignNet as the PE in its best configurations; the GPS ablation validates SignNet > vanilla LapPE on molecular benchmarks.
- [huang2024stability](huang2024stability.md): SPE extends SignNet/BasisNet by additionally guaranteeing stability to graph perturbations (near-degenerate eigenvalues cause discontinuous jumps in SignNet).
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT benchmarks SignNet-style PE on REGs and finds it expensive (requires eigendecomposition of the full REG Laplacian) and underperforms subgraph GNN PE.

## Technical Details

**The sign problem.** If $\phi$ is a Laplacian eigenvector with eigenvalue $\lambda$, then $-\phi$ is also a valid eigenvector. Any function $f$ of eigenvectors that is meaningful must satisfy $f(\phi_1, \ldots, \phi_k) = f(\epsilon_1 \phi_1, \ldots, \epsilon_k \phi_k)$ for any $\epsilon_i \in \{+1, -1\}$ — this is the group $(\mathbb{Z}/2)^k$ symmetry.

**SignNet architecture** (per eigenvector $v_i$):

$$\text{SignNet}(v_1, \ldots, v_k) = \rho\!\left(\left[\phi(v_i) + \phi(-v_i)\right]_{i=1}^k\right)$$

where $\phi: \mathbb{R}^N \to \mathbb{R}^d$ is an MLP (applied to each eigenvector independently) and $\rho: \mathbb{R}^{k \times d} \to \mathbb{R}^{d'}$ is another MLP applied to the concatenated outputs. Since $\phi(v_i) + \phi(-v_i) = \phi(-v_i) + \phi(v_i)$, the output is invariant to sign flip of any eigenvector.

**BasisNet architecture.** For a degenerate eigenspace with eigenvectors $U \in \mathbb{R}^{N \times d}$, apply a function $\rho$ that is equivariant to $O(d)$ rotations of the columns of $U$. Implemented via a DeepSets-style function over the columns: $\rho_\text{basis}(U) = g\!\left(\sum_{i=1}^d \phi(U_{\cdot,i})\right)$.

**Usage pattern.** SignNet is a drop-in PE encoder:
1. Compute $k$ smallest non-trivial Laplacian eigenvectors $\{v_i\}_{i=1}^k$
2. Apply SignNet: $\text{PE}(v) = \text{SignNet}(v_1, \ldots, v_k)[v]$ — extract the $v$-th row
3. Concatenate with node features: $\tilde{x}_v = [x_v \| \text{PE}(v)]$
4. Feed into any base model (GatedGCN, Transformer, PNA, GIN) unchanged

## Experiments

- ZINC (graph regression): SignNet + GatedGCN outperforms vanilla LapPE + GatedGCN by 18%; SignNet + random-flip LapPE only 3% — confirms that the principled invariant architecture, not just the sign regularization, drives the gain.
- PATTERN (node classification, structural): SignNet + PNA achieves 86.7% vs. LapPE + PNA 72.3% (random-flip baseline) — the largest gain; structural discrimination requires sign-invariant PE.
- All four base models (GatedGCN, Transformer, PNA, GIN) benefit from SignNet; GIN with random-flip LapPE shows no improvement at all, confirming the failure mode.
- GraphGPS ablation (Rampášek et al.): SignNet = best single PE in GPS experiments on molecular benchmarks.

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-transformer](graph-transformer.md)
