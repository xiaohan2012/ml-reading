---
title: Tabular Logit Lens
tags: [interpretability, mechanistic-interpretability, tabular, probing]
sources: [balef2026onelayer]
updated: 2026-05-25
---

# Tabular Logit Lens

Adaptation of the LLM "logit lens" (nostalgebraist 2020) / "tuned lens" (Belrose 2023) to tabular foundation models. Introduced by [Balef et al. 2026](balef2026onelayer.md).

## Method

For each transformer layer `l` of a frozen TFM:

1. Continue-pretrain a **fresh decoder** on synthetic datasets generated from the [TabICL](qu2025tabicl.md) prior, taking as input the hidden state at layer `l`.
2. At inference, forward the support+query through layers `1..l`, then apply decoder `l` to read out a prediction.

The per-layer decoder is closer to a tuned lens than to a logit lens — the original decoder by itself ("logit lens") gives brittle, sometimes non-monotonic readouts. Per-layer decoders are smooth and reliable.

## Why per-layer decoders, not the original

- TFM decoders project from the residual stream into class probabilities — they assume the final-layer representation.
- Early-layer hidden states already contain the predictive features but are not aligned with the original decoder's expected basis.
- Continued pretraining on a generic prior (TabICL's) is cheap and works across many TFMs.

## Use cases in Balef et al.

- **Early-exit study:** how shallow can inference be? — high AUC achievable from very early layers in all 6 TFMs studied.
- **Self-repair:** apply the lens to *every* layer after a skip intervention to see whether downstream layers compensate for the missing computation.
- **Inference-stage identification:** the gap between the original decoder and per-layer decoder defines the [prediction-ensembling stage](tfm-inference-stages.md).

## Limitations

- Decoder quality depends on the prior used for continued pretraining; here, TabICL priors. Models with more expressive priors (e.g. TabPFN(2.5), LimiX) may be under-served.
- Each decoder needs additional training cost (small relative to the TFM itself, but nonzero).

## Appearances in Sources

- [balef2026onelayer](balef2026onelayer.md) — introduces the method; uses it for early-exit, self-repair, and stage-identification analyses across six TFMs.

## Related Concepts

- [tfm-inference-stages](tfm-inference-stages.md) — the per-layer-decoder vs original-decoder gap defines the *prediction-ensembling* stage.
- [looped-transformer-tfm](looped-transformer-tfm.md) — a practical alternative to per-layer decoders for any-time predictions.
