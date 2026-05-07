---
title: "Benchmarking Graph Neural Networks"
tags: [source, graph-neural-network, positional-encoding, benchmark]
sources: [dwivedi2020benchmarking]
updated: 2026-05-07
---

# Benchmarking Graph Neural Networks

**Source:** https://arxiv.org/abs/2003.00982
**Title:** Benchmarking Graph Neural Networks
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Vijay Prakash Dwivedi, Chaitanya K. Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, Xavier Bresson
**Venue:** JMLR 2022 (arXiv 2020)

## Summary

- **What:** GNN evaluation used small, non-representative benchmarks (Cora, Citeseer, Pubmed) with inconsistent parameter budgets, making architectural comparisons unreliable.
- **How:** Establish 12 medium-scale datasets across chemistry, computer vision, and mathematical graph theory with a fixed 100k/500k parameter budget; introduce Laplacian eigenvectors as graph positional encodings (LapPE).
- **So what:** The benchmark and LapPE became the standard for all subsequent Graph Transformer papers (SAN, GraphGPS, Graphormer, LSPE, SignNet, RelGT); LapPE catalyzed an entire PE research direction.

## Challenges & Novelty

Prior GNN evaluation used citation networks (Cora, Citeseer) with at most a few thousand nodes — too small for meaningful architecture comparisons and too homophilous to test structural discrimination. Moreover, papers used widely varying parameter counts (10k–10M), making any comparison meaningless.

- **Fixed parameter budget enables fair comparison:** all models constrained to 100k or 500k parameters; a GCN and a Transformer are compared at the same capacity rather than different scales.
- **LapPE introduces positional identity to GNNs:** message-passing GNNs cannot distinguish node positions in structurally symmetric graphs (e.g., two nodes in a regular graph with identical neighborhoods). LapPE — the $k$ smallest non-trivial eigenvectors of the normalized graph Laplacian — provides each node with a unique positional signature that captures global graph structure. This catalyzed the entire PE research direction: SAN, SignNet, RWSE, SPE, PEARL, and RelGT's subgraph GNN PE all trace their lineage here.
- **Structural benchmarks expose GNN limitations:** PATTERN and CLUSTER require structural discrimination that pure aggregation-based GNNs fail on without PE — GCN without LapPE performs at random on CSL (cycle detection). These datasets don't appear in TU datasets or OGB and were specifically designed to test structural reasoning.

## Relation to Prior Work

| Benchmark | Size | Structural tasks | PE provided | Fixed budget |
|---|---|---|---|---|
| Cora/Citeseer | ~3K nodes | No | No | No |
| TU Datasets | Varies | No | No | No |
| **Benchmarking-GNNs** | 10K–500K nodes | Yes (PATTERN, CSL) | Yes (LapPE) | Yes (100k/500k) |
| OGB | Up to 3.8M nodes | No | No | No |
| [huang2023tgb](huang2023tgb.md) | 4M–67M edges | No | No | No |

- [dwivedi2021gt](dwivedi2021gt.md): the first GT to use this benchmark and LapPE; same lead author formalizes the GT architecture using this framework.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GraphGPS uses this benchmark for all experiments; extends LapPE to RWSE/SignNet.
- [kreuzer2021san](kreuzer2021san.md): SAN (NeurIPS 2021) builds on LapPE and evaluates on this benchmark.
- [lim2022signnet](lim2022signnet.md): SignNet addresses LapPE's sign ambiguity — a fundamental problem introduced with LapPE here.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT benchmarks LapPE on REGs (heterogeneous temporal graphs) and finds it expensive and inferior to subgraph GNN PE — showing the limits of molecular-graph PE on relational graphs.

## Technical Details

**Datasets (12 total).**

Molecular/chemical: ZINC (graph regression, 12K graphs, predict molecular properties); AQSOL (solubility prediction); OGBL-COLLAB (link prediction).

Computer vision (superpixels): MNIST (75 nodes/graph), CIFAR10 (150 nodes/graph) — edges connect spatially proximate superpixels.

Mathematical/structural: PATTERN (node classification, 10K graphs; detect specific graph patterns), CLUSTER (node clustering, community structure), CSL (Circular Skip Link: graph isomorphism test — GCN fails completely), CYCLES (detect cycles of specific length), GraphTheoryProp.

**Laplacian positional encoding.** For a graph $G$ with normalized Laplacian $L_\text{sym} = I - D^{-1/2}AD^{-1/2}$:

$$L_\text{sym} = U \Lambda U^T, \quad 0 = \lambda_0 \leq \lambda_1 \leq \cdots \leq \lambda_{N-1}$$

LapPE of node $v$: $\text{PE}(v) = [u_1(v), u_2(v), \ldots, u_k(v)] \in \mathbb{R}^k$ — the $v$-th entries of the $k$ smallest non-trivial eigenvectors. Concatenated with node features or added to input embeddings.

**Practical considerations:** eigendecomposition is $O(N^3)$; for large graphs this is prohibitive. The sign ambiguity ($u_i$ vs. $-u_i$) is handled by random sign-flipping during training — later rigorously resolved by SignNet.

**Parameter budget enforcement:** all weight matrices, embedding dimensions, and number of layers are constrained to match the 100k or 500k budget. Results are reported at both budgets for fair comparison across architecture families.

## Experiments

- GCN without LapPE: 0% accuracy on CSL (random baseline) — cannot distinguish any of the non-isomorphic circulant graphs. GCN + LapPE: 100% on CSL — LapPE is essential.
- PATTERN: GCN fails at ~50% (random); GatedGCN + LapPE: 85.6%. GT (Dwivedi & Bresson) with LapPE: 86.7%.
- ZINC: GatedGCN outperforms GCN/GAT/GIN significantly at same parameter budget; edge features are critical.
- BatchNorm consistently outperforms LayerNorm across all datasets — a finding later validated by Dwivedi & Bresson (AAAI 2021).

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-neural-network](graph-neural-network.md)
- [graph-transformer](graph-transformer.md)
