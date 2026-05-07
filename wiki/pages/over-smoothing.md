---
title: Over-Smoothing
tags: [concept, graph-neural-network, expressiveness]
sources: [alon2020bottleneck, ying2021graphormer, rampavsek2022graphgps, kim2024carte, mao2023hinormer]
updated: 2026-05-04
---

# Over-Smoothing

## Description

Over-smoothing is a depth-related limitation of message-passing GNNs: as the number of layers $K$ increases, node representations converge to the same stationary distribution (the dominant eigenvector of the propagation matrix), becoming indistinguishable regardless of input features. The result is that deep GNNs lose the ability to distinguish nodes — a model with many layers can be *worse* than a shallow one on node-level tasks.

**Root cause:** each message-passing layer performs a low-pass filter on node features. Stacking many layers repeatedly applies this filter, eventually making all representations converge to a global average weighted by node degree. This is the graph analog of signal smoothing in spectral analysis.

**Practical symptom:** accuracy peaks at $K=2$–$3$ layers and degrades for deeper networks, even when deeper networks would theoretically have larger receptive fields.

**Contrast with over-squashing:** over-squashing is a *structural* bottleneck that limits long-range propagation regardless of depth; over-smoothing is a *depth* problem that destroys local discrimination as layers accumulate. Both explain why deeper GNNs often fail, but in opposite regimes:

| | Task type | Cause | Effect |
|---|---|---|---|
| Over-smoothing | Short-range / local | Too many layers | Representations collapse to same vector |
| Over-squashing | Long-range | Exponential receptive field | Long-range signals vanish |

**Proposed fixes:**
- Residual connections (skip connections): preserve local features across layers
- Normalization (PairNorm, DiffGroupNorm): re-scale representations to prevent collapse
- Limiting depth: use $K=2$–$3$ layers and rely on PE/SE for long-range signals
- Global attention ([graph-transformer](graph-transformer.md)): bypass repeated local aggregation

## Appearances in Sources

- [alon2020bottleneck](alon2020bottleneck.md) — contrasts over-smoothing with over-squashing; clarifies these are distinct failure modes
- [ying2021graphormer](ying2021graphormer.md) — cites over-smoothing as a motivation for avoiding deep GNN stacks; uses shallow architecture + global attention instead
- [rampavsek2022graphgps](rampavsek2022graphgps.md) — MPNN component kept shallow (typically 1 layer per GPS block) precisely to avoid over-smoothing
- [kim2024carte](kim2024carte.md) — mentions over-smoothing as a known GNN limitation when motivating graph-per-row architecture
- [mao2023hinormer](mao2023hinormer.md) — cites over-smoothing in heterogeneous GNNs as motivation for decoupling local and global processing

## Related Concepts

- [over-squashing](over-squashing.md) — the complementary long-range limitation (distinct problem, different regime)
- [graph-neural-network](graph-neural-network.md) — the architecture family where over-smoothing occurs
- [graph-transformer](graph-transformer.md) — global attention as one architectural escape from over-smoothing
