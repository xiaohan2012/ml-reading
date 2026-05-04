---
title: "SPE: On the Stability of Expressive Positional Encodings for Graphs"
tags: [source, positional-encoding, graph-transformer, expressiveness]
sources: [huangstability]
updated: 2026-05-04
---

# SPE: On the Stability of Expressive Positional Encodings for Graphs

**Source:** https://arxiv.org/abs/2310.02579
**Title:** On the Stability of Expressive Positional Encodings for Graphs
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Yinan Huang, William Lu, Joshua Robinson, Yu Yang, Muhan Zhang, Stefanie Jegelka, Pan Li
**Venue:** ICLR 2024

## Summary

This paper addresses two fundamental challenges with Laplacian eigenvector-based PEs that prior work only partially solved. **Non-uniqueness** — multiple valid eigendecompositions exist for the same Laplacian — was handled by SignNet/BasisNet via invariant architectures. But **instability** — small perturbations to the graph Laplacian can result in completely different eigenspaces — was largely ignored, leading to poor generalization on unseen graph structures.

The root cause of instability is a "hard partition" of eigenspaces: when eigenvalues are close (near-degenerate), small perturbations can abruptly rotate the eigenvector basis, causing a discontinuous jump in the positional encoding. Existing methods that process eigenvectors independently (including SignNet when applied eigenvector-by-eigenvector) are discontinuous at eigenvalue degeneracies.

The solution is **SPE (Stable and Expressive Positional Encoding)**: an architecture that uses **eigenvalues to "softly partition" eigenspaces** — when eigenvalues are close, their eigenvectors are processed jointly and interpolated, so the output changes smoothly with graph perturbations. SPE satisfies all symmetry requirements (sign and basis invariance) while being the first provably *stable* architecture. Theoretically: SPE is at least as expressive as SignNet/BasisNet for non-degenerate spectra, and strictly better near degeneracies. Empirically: improved generalization on molecular property prediction and out-of-distribution settings.

## Key Takeaways

- **Two challenges**: non-uniqueness (addressed by SignNet) + instability (newly addressed by SPE).
- **Instability cause**: "hard partitioning" of eigenspaces — discontinuous at near-degenerate eigenvalues.
- **SPE fix**: eigenvalues used to softly weight eigenvector contributions; smooth interpolation near degeneracies.
- **First provably stable AND expressive PE** architecture; subsumes SignNet/BasisNet as special cases.
- Improved OOD generalization — stability matters when test graphs differ structurally from training graphs.

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-transformer](graph-transformer.md)

## Relation to Other Wiki Pages

- [positional-encoding](positional-encoding.md): SPE is the most theoretically complete eigenvector-based PE; resolves both non-uniqueness and instability.
- [limsign](limsign.md): SPE extends SignNet/BasisNet by additionally guaranteeing stability; SignNet solved symmetry but not stability.
- [dwivedi2020benchmarking](dwivedi2020benchmarking.md): the original LapPE had both problems; this paper closes the theoretical gap.
- [kanatsoulis2025learning](kanatsoulis2025learning.md): PEARL (Kanatsoulis et al.) also targets stability and efficiency; provides a GNN-based alternative to eigenvector processing.
