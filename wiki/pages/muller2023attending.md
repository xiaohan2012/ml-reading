---
title: "Attending to Graph Transformers (Survey)"
tags: [source, graph-transformer, survey, positional-encoding]
sources: [muller2023attending]
updated: 2026-05-07
---

# Attending to Graph Transformers (Survey)

**Source:** https://arxiv.org/abs/2302.04181
**Title:** Attending to Graph Transformers
**Date ingested:** 2026-05-04
**Type:** paper (survey)
**Authors:** Luis Müller, Mikhail Galkin, Christopher Morris, Ladislav Rampášek
**Venue:** arXiv 2023

## Summary

- **What:** The Graph Transformer literature grew rapidly with diverse architectures, making it hard to identify which design choices matter and compare methods systematically.
- **How:** Develop a unified taxonomy classifying GT designs by attention mechanism (local/global/hybrid), positional and structural encodings, tokenization granularity, and expressiveness guarantees; empirically probe how well different GTs recover structural graph properties.
- **So what:** First systematic survey of the GT design space; key finding: PE choice has outsized impact on performance and expressiveness, and global attention is not universally better than local attention.

## Challenges & Novelty

By 2023, at least 20 distinct Graph Transformer architectures had been published, each proposing different combinations of sparse/dense attention, structural encodings, and aggregation strategies — with no common framework for comparison. It was unclear whether performance differences were due to attention type, PE choice, training setup, or benchmark selection.

- **Taxonomy reveals redundancy and gaps:** many architectures are equivalent under the taxonomy — differing only in PE or normalization choice — while certain combinations (e.g., hybrid attention + equivariant PE) had not been explored.
- **PE choice dominates attention type:** empirical probing shows that for most structural graph properties, switching from LapPE to RWSE or SignNet changes performance more than switching from local to global attention.
- **Global attention is not always better:** on homophilous graphs (similar neighbors) and small graphs where over-squashing is not a bottleneck, local (sparse) attention outperforms full-graph attention — contradicting a common assumption.

## Relation to Prior Work

| Paper | Contribution | Covered by survey |
|---|---|---|
| [dwivedi2021gt](dwivedi2021gt.md) | Sparse GT + LapPE | Yes (baseline GT) |
| [ying2021graphormer](ying2021graphormer.md) | Global GT + SPD bias | Yes |
| [rampavsek2022graphgps](rampavsek2022graphgps.md) | Hybrid (MPNN + global) | Yes |
| [kim2022pure](kim2022pure.md) | Token GT | Yes |
| [shirzad2023exphormer](shirzad2023exphormer.md) | Sparse + virtual nodes | Yes |

- [rampavsek2022graphgps](rampavsek2022graphgps.md): GPS is cited as the canonical hybrid local+global architecture; the survey contextualizes it as the dominant design paradigm at time of writing.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT cites this survey to motivate that existing GTs fail on REGs — none of the surveyed architectures simultaneously handles heterogeneity, temporality, and scale.
- [alon2020bottleneck](alon2020bottleneck.md): the survey's analysis of when global attention helps connects to over-squashing — global attention is beneficial precisely when path lengths are long.

## Technical Details

**Taxonomy dimensions:**

1. *Attention scope:* **Local** (attend only to graph neighbors), **Global** (attend to all nodes), **Hybrid** (parallel local + global, as in GPS).

2. *Positional/Structural Encodings:*
   - Absolute: LapPE, RWSE, SignNet (added to node features)
   - Relative: SPD bias (Graphormer), kernel modulation (GraphiT)
   - Structural (graph-wide): WL colors, cycle counts, subgraph features
   - None: pure feature-based attention (GAT)

3. *Tokenization:* node-only tokens (most GTs) vs. node+edge tokens (TokenGT).

4. *Expressiveness:* $\leq$ 1-WL (most MPNNs), = 1-WL (GIN), $>$ 1-WL (GTs with structural PE or global attention).

**Empirical probing.** Train models to predict structural graph properties (degree distribution, clustering coefficient, shortest path distance, cycle counts) from node representations. Results: models with structural PE recover degree and clustering well; models with global attention recover SPD well; no single model recovers all properties.

**Key findings (5):**
1. PE choice matters more than attention type for most benchmarks.
2. Global attention alone, without PE, often underperforms sparse + good PE.
3. Hybrid (local + global) is the most robust paradigm.
4. Expressiveness guarantees (1-WL vs. $>$ 1-WL) don't always correlate with benchmark performance.
5. Graph heterophily changes the calculus: local attention + PE is preferable for heterophilous graphs.

## Experiments

- Survey probing results show no GT consistently dominates across all structural properties — motivates multi-signal PE (GPSE) and hybrid architectures.
- Benchmark re-evaluation with consistent training: differences between methods often smaller than reported; training protocol matters as much as architecture.
- Global attention methods (Graphormer, SAN) have highest SPD recovery but worst scaling to large graphs.
- Sparse attention methods (GT, Exphormer) have best scalability but lowest SPD recovery.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [positional-encoding](positional-encoding.md)
