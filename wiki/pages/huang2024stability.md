---
title: "SPE: On the Stability of Expressive Positional Encodings for Graphs"
tags: [source, positional-encoding, graph-transformer, expressiveness]
sources: [huang2024stability]
updated: 2026-05-07
---

# SPE: On the Stability of Expressive Positional Encodings for Graphs

**Source:** https://arxiv.org/abs/2310.02579
**Title:** On the Stability of Expressive Positional Encodings for Graphs
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Yinan Huang, William Lu, Joshua Robinson, Yu Yang, Muhan Zhang, Stefanie Jegelka, Pan Li
**Venue:** ICLR 2024

## Summary

- **What:** SignNet and BasisNet resolve non-uniqueness of Laplacian eigenvectors (sign/basis symmetries) but still produce discontinuous PEs near degenerate eigenvalues — a small graph perturbation can cause an abrupt eigenspace rotation, making generalization to out-of-distribution graphs poor.
- **How:** Use eigenvalues to "softly partition" eigenspaces, interpolating eigenvector contributions smoothly near degeneracies rather than hard-partitioning by eigenvalue clusters.
- **So what:** SPE is the first provably stable AND expressive PE architecture; achieves improved OOD generalization on molecular property prediction; subsumes SignNet/BasisNet as special cases at non-degenerate spectra.

## Challenges & Novelty

Non-uniqueness of Laplacian eigenvectors has two distinct failure modes: (1) **sign/basis ambiguity** (which SignNet/BasisNet address) and (2) **instability** — eigenspaces themselves are discontinuous as functions of the graph. When two eigenvalues $\lambda_i$ and $\lambda_j$ are close (near-degenerate), any small perturbation to the graph can arbitrarily rotate the corresponding eigenvectors within the joint eigenspace. SignNet, even when applied basis-invariantly, produces a hard partition of eigenvectors by their discrete eigenvalue cluster — this hard partitioning is discontinuous at near-degeneracies.

- **Hard partitioning = discontinuity:** any method that assigns eigenvectors to eigenspace groups based on exact eigenvalue equality will jump discontinuously when two eigenvalues cross — a problem that occurs generically in perturbed graphs.
- **Soft partitioning via eigenvalue weighting:** SPE weights each pair of eigenvectors by a kernel that depends on their eigenvalue similarity: $K(\lambda_i, \lambda_j)$ is large when $\lambda_i \approx \lambda_j$ and decays as they separate. When eigenvalues are near-degenerate, their eigenvectors are mixed (interpolated); when they are well-separated, they are processed independently as in SignNet.
- **Provable Lipschitz continuity:** the soft weighting makes the PE a Lipschitz-continuous function of the graph Laplacian — small graph perturbations produce small PE changes. This is the formal stability guarantee.

## Relation to Prior Work

| PE | Non-uniqueness | Instability | Expressiveness | OOD |
|---|---|---|---|---|
| LapPE ([dwivedi2020benchmarking](dwivedi2020benchmarking.md)) | Not addressed | Yes | Limited | Poor |
| SignNet ([lim2022signnet](lim2022signnet.md)) | Sign (provably) | Yes (near-degenerate) | Universal (sign-inv.) | Moderate |
| BasisNet ([lim2022signnet](lim2022signnet.md)) | Sign + Basis | Yes (near-degenerate) | Universal (basis-inv.) | Moderate |
| **SPE** | Sign + Basis | Solved (provably) | $\geq$ SignNet | Best |
| PEARL ([kanatsoulis2025pearl](kanatsoulis2025pearl.md)) | Via GNN | Via GNN smoothness | GNN-level | Good |

- [lim2022signnet](lim2022signnet.md): SPE extends SignNet/BasisNet; when eigenvalues are non-degenerate (well-separated), SPE reduces to SignNet — SPE strictly subsumes it. SignNet addressed symmetry but not stability; SPE addresses both.
- [kanatsoulis2025pearl](kanatsoulis2025pearl.md): PEARL takes an orthogonal approach — learn PE directly with GNNs (avoiding eigendecomposition altogether); stability is inherited from GNN continuity rather than eigenvalue weighting.
- [dwivedi2020benchmarking](dwivedi2020benchmarking.md): the original LapPE had both problems (non-uniqueness + instability); SPE closes the theoretical gap fully.

## Technical Details

**Root cause of instability.** The eigendecomposition $L = \sum_i \lambda_i \phi_i \phi_i^T$ is unique only when all eigenvalues are distinct. For an eigenvalue $\lambda_i$ of multiplicity $d > 1$, the corresponding eigenvector basis is any orthonormal basis for the eigenspace — an uncountably infinite family. Any method that processes eigenvectors one at a time (like SignNet applied independently) implicitly hard-partitions the spectrum: eigenvectors with the same discrete label $\lambda_i$ are processed together, but near $\lambda_i \approx \lambda_j$, the partition is discontinuous.

**SPE soft partitioning.** SPE replaces hard eigenspace grouping with a continuous weighting:

$$\text{SPE}(v)_i = \sum_{j=1}^k K(\lambda_i, \lambda_j) \cdot \phi_j(v) \cdot g(\lambda_j)$$

where $K(\lambda_i, \lambda_j) = \exp(-(\lambda_i - \lambda_j)^2 / \sigma^2)$ is a Gaussian kernel (bandwidth $\sigma$ controls the transition), and $g(\lambda_j) \in \mathbb{R}^d$ is a learned function of the eigenvalue. When $\lambda_i \approx \lambda_j$, the kernel weight is large — both eigenvectors contribute equally (smooth interpolation). When $\lambda_i \gg \lambda_j$, the weight is near zero — independent processing (reducing to SignNet).

**Lipschitz continuity proof.** The mapping from graph Laplacian to SPE output is $\sigma^{-1}$-Lipschitz in the spectral norm of $\Delta L$ — a perturbation of magnitude $\epsilon$ to the graph produces a PE change of at most $\epsilon/\sigma$.

**Expressiveness.** SPE is at least as expressive as SignNet/BasisNet for non-degenerate spectra; strictly more expressive near degeneracies (can distinguish graphs that differ only in near-degenerate eigenspace structure).

## Experiments

- ZINC (graph regression): SPE + GatedGCN outperforms SignNet + GatedGCN by 5% MAE — stability gain from smooth interpolation near degenerate eigenvalues.
- OOD evaluation (train on small molecules, test on larger ones): SPE generalizes better than SignNet across all base models; gap largest when test graphs have structural features not seen in training (high-symmetry graphs with many degenerate eigenvalues).
- Sensitivity analysis: SPE PE changes smoothly as graph edges are gradually perturbed; SignNet PE changes discontinuously at eigenvalue crossings.
- PEARL comparison: SPE and PEARL are complementary — PEARL avoids eigendecomposition entirely (better for dynamic/large graphs); SPE retains full spectral expressiveness (better for precision on small molecular benchmarks).

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-transformer](graph-transformer.md)
