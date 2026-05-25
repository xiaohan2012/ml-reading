---
title: TFM Inference Stages
tags: [tabular, interpretability, mechanistic-interpretability]
sources: [balef2026onelayer]
updated: 2026-05-25
---

# TFM Inference Stages

Four depth-dependent stages of inference observed in tabular foundation models, adapted by [Balef et al. 2026](balef2026onelayer.md) from the LLM stage taxonomy of Lad et al. 2025 (early detokenization → feature refinement → ensembling → output sharpening).

The stages can overlap because of layer redundancy and self-repair.

## 1. Latent mapping

Early layers extend the input encoder, transforming raw input embeddings into representations that fit the residual-stream geometry. Skipping any of these layers is catastrophic.

Models with stronger upstream encoders compress this stage:

- [TabICL](qu2025tabicl.md) — column embedder + row-wise interaction module already produce rich features.
- LimiX-2M — RBF-kernel preprocessing.
- LimiX-16M — lacks the same preprocessing, so it relies more on its early transformer layers.

## 2. Feature engineering and labeling

Middle layers rapidly improve [tabular-logit-lens](tabular-logit-lens.md) per-layer-decoder performance. The [separation gap](balef2026onelayer.md) grows: same-class feature embeddings move closer, different-class embeddings move apart. Label embeddings form iteratively, slightly trailing feature embeddings.

## 3. Prediction ensembling

The per-layer decoder has saturated, but the *original* decoder is still improving. The model is reshaping representations to align with the original decoder. Clearly visible in [TabPFN v2](hollmann2025tabpfnv2.md) and TabPFN(2.5); much less pronounced in other models.

## 4. Prediction calibration

Per-layer and original decoders match on AUC, but balanced accuracy jumps and output entropy continues to shift. The final stage tunes probability calibration rather than rank order.

## Compared to LLMs

| Stage                | LLM (Lad et al.)              | TFM (Balef et al.)                  |
| -------------------- | ----------------------------- | ----------------------------------- |
| Early                | Detokenization                | Latent mapping                      |
| Middle               | Task/entity feature refinement | Feature engineering and labeling   |
| Mid-to-late          | Prediction ensembling         | Prediction ensembling (weaker)      |
| Final                | Output sharpening             | Prediction calibration              |

Differences: TFMs front-load decisions (high early-exit performance), so middle-stage redundancy and final-stage importance are inverted relative to LLMs.

## Appearances in Sources

- [balef2026onelayer](balef2026onelayer.md) — introduces and operationalizes the four-stage taxonomy via embedding similarity, separation gap, and the tabular logit lens.

## Related Concepts

- [tabular-logit-lens](tabular-logit-lens.md) — the gap between per-layer and original decoders demarcates the ensembling stage.
- [looped-transformer-tfm](looped-transformer-tfm.md) — block-redundancy in the middle stages justifies the looped-block design.
