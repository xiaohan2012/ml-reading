---
title: TFM Inference Stages
tags: [tabular, interpretability, mechanistic-interpretability]
sources: [balef2026onelayer]
updated: 2026-05-25
---

# TFM Inference Stages

Four depth-dependent stages of inference observed in tabular foundation models, adapted by [Balef et al. 2026](balef2026onelayer.md) from the LLM stage taxonomy of Lad et al. 2025 (early detokenization → feature refinement → ensembling → output sharpening).

The stages can overlap because of layer redundancy and self-repair.

![Schematic of the four TFM inference stages. Individual per-layer decoders (orange) reach high AUC during *feature engineering and labeling*; the final decoder (blue) catches up during *prediction ensembling*; accuracy completes during *prediction calibration*.](assets/tfm-inference-stages-overview.png)

## The four stages

| # | Stage | What happens | Where most visible |
| - | --- | --- | --- |
| 1 | **Latent mapping** | Early layers extend the input encoder, mapping raw embeddings into the residual-stream geometry. Skipping is catastrophic. | All models; compressed in [TabICL](qu2025tabicl.md) (column embedder + row-wise interaction) and LimiX-2M (RBF preprocessing). LimiX-16M lacks this preprocessing and depends more on early layers. |
| 2 | **Feature engineering and labeling** | Middle layers rapidly improve [tabular-logit-lens](tabular-logit-lens.md) per-layer decoders; [separation gap](balef2026onelayer.md) grows; labels form after features. | All 6 models. |
| 3 | **Prediction ensembling** | Per-layer decoder saturates while the original decoder keeps improving — representations are reshaped to align with the original decoder. | [TabPFN v2](hollmann2025tabpfnv2.md), TabPFN(2.5); weak elsewhere. |
| 4 | **Prediction calibration** | Per-layer and original decoders match on AUC, but balanced accuracy jumps and entropy keeps shifting — tunes probability calibration, not rank order. | All models, final layers. |

## Compared to LLMs

| Stage | LLM (Lad et al.) | TFM (Balef et al.) |
| --- | --- | --- |
| Early | Detokenization | Latent mapping |
| Middle | Task/entity feature refinement | Feature engineering and labeling |
| Mid-to-late | Prediction ensembling | Prediction ensembling (weaker) |
| Final | Output sharpening | Prediction calibration |

TFMs front-load decisions (high early-exit performance), so middle-stage redundancy is *higher* and final-stage importance is *lower* than in LLMs.

## Appearances in Sources

- [balef2026onelayer](balef2026onelayer.md) — introduces and operationalizes the four-stage taxonomy via embedding similarity, separation gap, and the tabular logit lens.

## Related Concepts

- [tabular-logit-lens](tabular-logit-lens.md) — the gap between per-layer and original decoders demarcates the ensembling stage.
- [looped-transformer-tfm](looped-transformer-tfm.md) — block-redundancy in the middle stages justifies the looped-block design.
