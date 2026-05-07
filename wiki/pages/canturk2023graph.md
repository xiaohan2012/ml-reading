---
title: "GPSE: Graph Positional and Structural Encoder"
tags: [source, positional-encoding, graph-transformer]
sources: [canturk2023graph]
updated: 2026-05-07
---

# GPSE: Graph Positional and Structural Encoder

**Source:** https://arxiv.org/abs/2307.07107
**Title:** Graph Positional and Structural Encoder
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Semih Cantürk, Renming Liu, Olivier Lapointe-Gagné, Vincent Létourneau, Guy Wolf, Dominique Beaini, Ladislav Rampášek
**Venue:** arXiv 2023

## Summary

- **What:** Choosing and tuning positional/structural encodings (PSEs) for graphs is expensive — different PSEs capture complementary structural properties, and no single PSE is best across all tasks and datasets.
- **How:** Pretrain a universal graph encoder self-supervised on a large collection of graphs to simultaneously reconstruct multiple PSE signals (RWSE, LapPE, SignNet, etc.), producing a single fixed-dimensional PSE representation transferable to any downstream task.
- **So what:** GPSE replaces expensive per-graph eigendecomposition and PSE selection with a single pretrained encoder; plug-in replacement for individual PSEs in any GNN or GT without retraining.

## Challenges & Novelty

Individual PSEs each require their own precomputation: LapPE needs $O(N^3)$ eigendecomposition; RWSE needs $O(N^2 k)$ random walk computation; SignNet needs eigendecomposition + a learned invariant network. For new datasets, practitioners must try multiple PSEs and retrain. No prior work unified multiple PSE signals into a single transferable representation.

- **Multi-PSE pretraining via self-supervision:** GPSE is trained to reconstruct the outputs of multiple PSE types simultaneously — the encoder learns a compressed representation that captures what all PSEs encode, without committing to any single structural inductive bias.
- **Avoids eigendecomposition at inference:** once pretrained, GPSE produces PSE features in a single forward pass — no eigendecomposition needed for new graphs. The encoder has internalized the structural signals from training.
- **Transfer across datasets and tasks:** pretrained on large diverse graph collections (molecular + social), GPSE PSE features transfer to downstream tasks without re-pretraining — similar in spirit to language model embeddings.

## Relation to Prior Work

| PE Approach | Multi-signal | Transferable | Eigendecomp at inference | Team |
|---|---|---|---|---|
| LapPE ([dwivedi2020benchmarking](dwivedi2020benchmarking.md)) | No (single) | No | Yes | Dwivedi et al. |
| RWSE ([rampavsek2022graphgps](rampavsek2022graphgps.md)) | No (single) | No | No | Rampášek et al. |
| SignNet ([lim2022signnet](lim2022signnet.md)) | No (single) | No | Yes | Lim et al. |
| **GPSE** | Yes (multi) | Yes | No | Rampášek et al. |

- [rampavsek2022graphgps](rampavsek2022graphgps.md): from the same team as GraphGPS; GPSE extends GPS's PE taxonomy to a pretrained multi-signal encoder — a natural follow-up that operationalizes the "PE matters most" finding.
- [lim2022signnet](lim2022signnet.md): SignNet is one of the PSE signals encoded by GPSE; GPSE provides a route to using SignNet-quality representations without retraining SignNet per dataset.
- [dwivedi2022lspe](dwivedi2022lspe.md): LSPE showed PE can be learned per-model (adaptive); GPSE pretrains PE independently and transfers it — a different point on the same spectrum.

## Technical Details

**Pretraining objective.** Given a graph $G$, compute multiple PSEs: $\text{PSE}_1(G), \ldots, \text{PSE}_m(G)$ (e.g., RWSE, 5-step; LapPE, 8 eigenvectors; SignNet-LapPE). Train the encoder $f_\theta$ to reconstruct all PSEs from graph structure:

$$\mathcal{L} = \sum_{t=1}^m \|f_\theta(G)_t - \text{PSE}_t(G)\|^2$$

The encoder is a GNN (e.g., GIN or MPNN) that takes node features + adjacency as input and outputs $m$ PSE-dimensional vectors per node.

**Inference.** For a new graph $G'$: run $f_\theta(G')$ (single GNN forward pass) → PSE features per node. No eigendecomposition. Concatenate with node features → feed into any downstream GNN or GT.

**Pretraining data.** Large-scale diverse graphs: PCQM4Mv2 (3.7M molecular graphs), OGB-ArXiv (citation network), plus social graph collections — covering molecular, social, and citation domains.

## Experiments

- GPSE features used as drop-in PE replacement in GraphGPS: match or outperform individually tuned PSEs on ZINC, PCQM4Mv2, and peptides datasets.
- 10–100× faster than eigendecomposition-based PSEs (LapPE, SignNet) on large graphs (>10K nodes).
- Transfer from molecular pretraining: GPSE pretrained on PCQM4Mv2 transfers to social graph tasks (OGB-ArXiv) with minimal performance drop.
- Multi-signal GPSE consistently outperforms single-signal alternatives when any one PSE is suboptimal for the target task.

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-transformer](graph-transformer.md)
