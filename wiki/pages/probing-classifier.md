---
title: Probing Classifier
tags: [interpretability, mechanistic-interpretability, probing]
sources: [balef2026onelayer]
updated: 2026-05-28
---

# Probing Classifier

A small supervised classifier trained on a frozen model's hidden state to measure how much of a target label is **linearly decodable** at that layer. Originated with Alain & Bengio (2016); now standard across LLM and TFM mechanistic interpretability.

## Method

- **Probe:** logistic regression on hidden states $h_\ell$ (alternatives: KNN, LDA, fine-tuned decoder).
- **Per-layer AUC:** train $h_\ell \to y$, report validation AUC — curve of where the label becomes linearly readable.
- **Cross-layer matrix:** cell $(i, j)$ = probe trained on layer $i$, tested on layer $j$. Off-diagonal cells expose whether representations are **cumulative** (each layer adds directions, keeps old ones — forward transfer ✅, backward ❌) or **replaced** (each layer's directions distinct — only diagonal ✅).

## What it can and cannot say

- ✅ Reveals what's *in* the representation (information-theoretic readout).
- ✅ Cross-layer transfer distinguishes cumulative vs replaced features.
- ❌ Does not say whether the **model itself** would emit the prediction at that layer — that's the job of the [tabular logit lens](tabular-logit-lens.md).

## Caveats

- **Token selection.** In ICL/TFM settings, support tokens leak labels — train probes on query tokens only.
- **Probe capacity.** Stronger probes can extract more; interpret AUC relative to a fixed probe family.
- **Adjacent-layer block redundancy** can dominate the heatmap and mask the longer-range cumulative-vs-replaced signal — read off-diagonal asymmetry at long range, not short range.

## Appearances in Sources

- [balef2026onelayer](balef2026onelayer.md) — cross-layer probe matrix shows forward-only transfer across six TFMs, evidencing cumulative-feature accumulation in the residual stream.
- [ferrando2024primer](ferrando2024primer.md) — surveys probing in the broader LLM mech-interp context; control tasks, MDL probes, correlation-vs-causation caveat.
- [bilos2026mechanistic](bilos2026mechanistic.md) — layer-wise linear probes locate where each TFM becomes class-readable (TabPFN's sharp L8→L9 jump; TabICL readable at the column-embedder output); frozen-vs-retrained probes distinguish a coordinate-frame change from information loss.

## Related Concepts

- [tabular-logit-lens](tabular-logit-lens.md) — functional counterpart: reads out via a trained decoder in the model's own output space.
- [representation-similarity](representation-similarity.md) — adjacent-layer redundancy seen in CKA / cosine heatmaps appears as block structure in the probe matrix.
