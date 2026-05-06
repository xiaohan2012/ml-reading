---
title: "GPSE: Graph Positional and Structural Encoder"
tags: [source, positional-encoding, graph-transformer]
sources: [canturk2023graph]
updated: 2026-05-04
---

# GPSE: Graph Positional and Structural Encoder

**Source:** https://arxiv.org/abs/2307.07107
**Title:** Graph Positional and Structural Encoder
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Semih Cantürk, Renming Liu, Olivier Lapointe-Gagné, Vincent Létourneau, Guy Wolf, Dominique Beaini, Ladislav Rampášek
**Venue:** arXiv 2023

## Summary

GPSE addresses the challenge of selecting and combining positional and structural encodings (PSEs) for graphs. Rather than designing a single PSE, GPSE is a **pre-trained, universal graph encoder** that produces multi-scale PSE representations by combining multiple encoding strategies (RWSE, LapPE, SignNet, etc.) into a unified learned representation. The key insight: different PSEs capture complementary structural properties, and a pre-trained encoder that combines them can be used as a plug-in PSE for any downstream graph model.

GPSE is trained self-supervised on a large collection of molecular and social graphs, learning to reconstruct multiple PSE signals simultaneously. The resulting encoder produces a fixed-dimensional PSE vector for any new graph without requiring expensive eigendecomposition — the encoder has internalized structural signals from multiple PSE types. This enables pre-computed, fast, and rich PSEs that transfer across datasets and tasks.

## Key Takeaways

- **Universal pre-trained PSE encoder**: combines RWSE, LapPE, SignNet, etc. into a single learned representation via self-supervised pre-training.
- Avoids expensive per-graph eigendecomposition at inference time — the encoder replaces it.
- Plug-in PSE: drop into any graph model (GNN or GT) as a richer alternative to individual PSEs.
- Transfers across datasets/tasks — pre-train once, apply everywhere.
- From the GraphGPS team (Rampášek, Beaini et al.); extends GPS's PE/SE taxonomy to a learned multi-PSE regime.

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-transformer](graph-transformer.md)

## Relation to Other Wiki Pages

- [positional-encoding](positional-encoding.md): GPSE is the most comprehensive PE approach — a pre-trained multi-signal encoder rather than a single hand-crafted PE.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): same team; GPSE extends GPS's PE taxonomy to a pre-trained universal encoder.
- [lim2022signnet](lim2022signnet.md): SignNet is one of the PSE types encoded by GPSE.
- [dwivedi2022graph](dwivedi2022graph.md): LSPE showed PE can be learned per-model; GPSE pre-trains PE independently and transfers.
