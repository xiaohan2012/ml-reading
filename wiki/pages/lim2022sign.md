---
title: "SignNet / BasisNet: Sign and Basis Invariant Networks for Spectral Graph Representation Learning"
tags: [source, positional-encoding, graph-transformer, expressiveness]
sources: [lim2022sign]
updated: 2026-05-04
---

# SignNet / BasisNet: Sign and Basis Invariant Networks for Spectral Graph Representation Learning

**Source:** https://arxiv.org/abs/2202.13013
**Title:** Sign and Basis Invariant Networks for Spectral Graph Representation Learning
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Derek Lim, Joshua Robinson, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, Stefanie Jegelka
**Venue:** ICLR 2022

## Summary

LapPE has two fundamental symmetry problems: **(1) sign flips** — if $v$ is a Laplacian eigenvector with eigenvalue $\lambda$, so is $-v$; numerical solvers pick a sign arbitrarily, so the same graph can produce $v$ or $-v$ depending on the run. With $k$ eigenvectors there are $2^k$ valid sign combinations. Previous work applied random sign-flipping as data augmentation during training, but this fails in practice — on some base models (PNA, GINE), LapPE with random flipping doesn't improve over no PE at all. **(2) Basis symmetries** — when an eigenvalue has multiplicity $d > 1$, there are infinitely many valid orthonormal bases for that eigenspace (related by any $Q \in O(d)$); prior work didn't address this at all.

SignNet and BasisNet are neural architectures that are provably invariant to both symmetries. **SignNet** handles sign flips by applying a DeepSets-style function that is symmetric in $v$ and $-v$: $\rho\!\left(\phi(v) + \phi(-v)\right)$ for each eigenvector $v$. **BasisNet** handles the more general basis symmetry by processing the entire eigenspace simultaneously with an architecture invariant to $O(d)$ rotations of the eigenvector basis.

Both networks are **universal** under their respective symmetry constraints: they can approximate any continuous function of eigenvectors that is invariant to those symmetries. They are provably more expressive than all prior spectral methods on graphs, subsuming all spectral graph convolutions, certain spectral invariants, and existing graph PEs as special cases. Experiments show significant improvements on molecular graph regression, graph representation learning, and neural fields on meshes.

**Usage pattern:** SignNet is a drop-in PE encoder — eigenvectors are precomputed once (fixed), passed through SignNet ($\phi$ + $\rho$), and the output concatenated with node features: $[X,\ \rho([\phi(v_i)+\phi(-v_i)]_{i=1}^k)]$. Same interface as raw LapPE but with a learned invariant transformation in front. Tested as a plug-in with four base models (GatedGCN, Transformer, PNA, GIN) — none require modification. GraphGPS cites SignNet as the best-performing PE in its ablations.

## Key Takeaways

- **Identifies and fixes both LapPE symmetry problems**: sign flips (SignNet) and basis symmetries (BasisNet) via provably invariant architectures.
- SignNet: $\rho(\phi(v) + \phi(-v))$ — symmetric aggregation over $+v$ and $-v$ gives sign invariance.
- **Universal approximation**: can express any continuous sign-/basis-invariant function of eigenvectors.
- **Strictly more expressive** than all prior spectral GNN methods, including LapPE, SAN, and spectral convolutions.
- Adopted by GraphGPS as the PE of choice for best performance; also the PE that RelGT considers and benchmarks against subgraph GNN PE on REGs.

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-transformer](graph-transformer.md)

## Relation to Other Wiki Pages

- [positional-encoding](positional-encoding.md): SignNet is the recommended fix for LapPE's sign ambiguity; noted in the wiki as the best single encoding in GPS ablations.
- [dwivedi2020benchmarking](dwivedi2020benchmarking.md): directly addresses the sign ambiguity problem identified in Benchmarking-GNNs' LapPE proposal.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GraphGPS uses SignNet as the PE in its best configurations; the GPS ablation explicitly validates SignNet > vanilla LapPE.
- [san](san.md): SAN acknowledged sign ambiguity as a limitation; SignNet solves it rigorously.
- [huang2024stability](huang2024stability.md): SPE (stability paper) builds on SignNet's sign invariance and additionally addresses stability to graph perturbations.
