---
title: Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models
tags:
  - tabular
  - interpretability
  - mechanistic-interpretability
  - tabpfn
  - tabicl
  - icml
sources:
  - balef2026onelayer
updated: 2026-05-25
---

# Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models

**Source:** https://arxiv.org/abs/2605.06510
**Title:** Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models
**Date ingested:** 2026-05-25
**Type:** paper
**Authors:** Amir Rezaei Balef, Mykhailo Koshil, Katharina Eggensperger
**Venue:** ICML 2026
**Code:** https://github.com/amirbalef/is_one_layer_enough

## Summary

First mechanistic-interpretability study of tabular foundation models (TFMs). Three questions, three answers:

- **Q1 — How does inference unfold across depth in a TFM?**
    - **Iterative refinement.** Predictions form in the first few layers; later layers add features without overwriting.
    - **Depth-redundant but active.** Middle/late skips are self-repaired downstream; the first layer is the only unrecoverable one.
- **Q2 — Do LLM mechanistic findings transfer to TFMs?**
    - **Partially.** Shared shape: early-critical, block-structured representations, forward-only probe transfer.
    - Diverges from LLMs in three ways: more middle-layer redundancy, less final-layer importance, much higher swap sensitivity (TFMs are order-specialized).
- **Follow-up — Is one layer enough?**
    - **A single looped block matches a 6-layer stack** at matched compute, licensing recurrent / shallow-depth TFM architectures.

**Models studied.** TabPFN v1, TabPFN v2, TabPFN(2.5), TabICL, LIMIX-2M, LIMIX-16M — all encoder-only, set-invariant TFMs.
**Benchmarks.** 15 binary-classification tasks from TabArena (≤10K samples, ≤100 features) and 34 small tasks from PMLBmini (≤500 samples).

## RQ1: How does inference unfold across depth in TFMs?

i.e. as a data point flows through the layers, where and how is the prediction gradually built up? Tackled with six layer-wise experiments that fall into three families.

### Experiments

| Family                     | Experiment                            | Question it answers                                                                       |
| -------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------- |
| A. Look at representations | Embedding similarity                  | Do adjacent layers share a representation space (redundant blocks)?                       |
|                            | Separation gap                        | Does the layer pull classes further apart than the previous one?                          |
| B. Decode representations  | Probing classifiers                   | Is the label linearly decodable here, and are features cumulative or replaced?            |
|                            | Tabular logit lens                    | At what depth is the prediction effectively formed?                                       |
| C. Causal intervention     | Layer ablation (skip / repeat / swap) | Is each layer necessary, reusable, and position-specific?                                 |
|                            | Self-repair                           | Do downstream layers recover a skipped layer's computation (redundancy vs active repair)? |

The three families form a ladder of evidence: **observe → read out → intervene**. A says *what's there*, B says *what can be extracted*, C says *what is causally required* — each rung stronger than the last.

### A. Look at the representations (no intervention)

- **1. Embedding similarity** — compare layer outputs pairwise to find redundant blocks.
    - **Object:** the per-token hidden state $h_\ell(x_i)$ at every layer $\ell$, pooled across all tokens (support + query, all features for per-cell models) and averaged over datasets.
    - **Two metrics, shown together** (lower triangle = cosine, upper triangle = CKA):
        - **Cosine similarity** — point-wise $\cos(h_i^{(t)}, h_j^{(t)})$ averaged over tokens and datasets. Angular alignment; survives non-isotropic scaling.
        - **Linear CKA** — holistic: $\mathrm{CKA}(X_i, X_j) = \frac{\|X_i^\top X_j\|_F^2}{\|X_i^\top X_i\|_F \cdot \|X_j^\top X_j\|_F + \varepsilon}$ on column-centered token matrices. Global geometry; invariant to isotropic scaling + orthogonal transforms.
    - **Diagnostic:** cosine $\gg$ CKA → non-isotropic dimension stretching (vectors aligned, geometry distorted).
    - **Output:** a layer × layer heatmap per model; non-redundant case = sharp diagonal, blue everywhere else; redundant case = thick red blocks along the diagonal.

    ![Embedding similarity heatmaps](assets/balef2026onelayer-embedding-similarity.png)

    **Takeaway 1:** TFMs often form blocks in which the embeddings remain similar. *Upper triangle: linear CKA; lower triangle: cosine.*
- **2. Separation gap** — track how cleanly the model pulls classes apart layer by layer.
    - **Formula:** with $h_\ell(x_i)$ the layer-$\ell$ embedding, $y_i$ its label, and pair sets $\mathcal{P}^{\text{within}}_\ell = \{(x_i,x_j) : y_i = y_j\}$, $\mathcal{P}^{\text{between}}_\ell = \{(x_i,x_j) : y_i \neq y_j\}$:
        $$D^{\text{within}}_\ell = \tfrac{1}{|\mathcal{P}^{\text{within}}_\ell|} \sum_{(x_i,x_j) \in \mathcal{P}^{\text{within}}_\ell} d(h_\ell(x_i), h_\ell(x_j))$$
        $$D^{\text{between}}_\ell = \tfrac{1}{|\mathcal{P}^{\text{between}}_\ell|} \sum_{(x_i,x_j) \in \mathcal{P}^{\text{between}}_\ell} d(h_\ell(x_i), h_\ell(x_j))$$
        $$\boxed{\Delta_\ell = D^{\text{between}}_\ell - D^{\text{within}}_\ell}$$
    - **Defaults:** $d = $ cosine distance; 100 within + 100 between pairs sampled per dataset; PCA at 95% variance applied first to control noise in the high-dim residual stream.
    - **Computed separately** for support vs query, and for per-cell models also feature vs label embeddings — exposes the "features form first, labels follow" pattern.

    ![Separation gap across layers](assets/balef2026onelayer-separation-gap.png)

    **Takeaway 2:** TFMs incrementally increase the distance between samples from different classes. *Bold lines: dataset average; thin lines: individual datasets; label embedding lags feature embedding.*

### B. Decode the representations (read out without changing the model)

- **3. Probing classifiers** — measure how much of the label is already linearly decodable at each layer.
    - **Probe:** logistic regression (also KNN, LDA, fine-tuned decoder in appendix), trained on frozen layer-$i$ embeddings.
    - **Tokens used:** **query embeddings only** (support tokens leak labels). The query set is split into train + validation halves; probe trained on the train half.
    - **Cross-layer matrix (Figure 5):** cell $(i, j)$ = AUC of probe trained on layer $i$, tested on layer $j$. **Y-axis = train layer, X-axis = test layer, color = normalized ROC-AUC.**
    - **Why cross-layer, not just diagonal?** Diagonal ($i=j$) only tells you *when* the label becomes decodable; it cannot distinguish **cumulative features** (each layer adds new directions, keeps old) from **replaced features** (each layer learns different directions) — both give the same diagonal curve. Off-diagonal cells force them apart (see scenarios below).
    - **Signature finding:** red upper triangle ($j > i$ works) + blue lower triangle ($j < i$ fails) → Scenario A holds: each layer **adds** features without overwriting earlier ones (cumulative residual-stream enrichment).

    ![Cross-layer probe transfer matrix](assets/balef2026onelayer-probing.png)

    **Takeaway 3:** Each layer cumulatively enriches the representation by adding new features while preserving previous ones. *Y-axis: train layer; X-axis: test layer; color: normalized ROC-AUC.*

> **Note — reading the asymmetry correctly.** The cumulative-feature claim relies on **long-range** off-diagonal asymmetry, not short-range. Two patterns coexist in the same heatmap:
>
> - **Short-range, symmetric:** diagonal-aligned red blocks (e.g. layers 4-9 all transferring to each other in both directions) — adjacent layers hold near-identical representations, so probes transfer within a block. This is **block redundancy** bleeding through from Experiment 1, *not* the cumulative-feature signal.
> - **Long-range, asymmetric:** compare a far off-diagonal pair like cell $(1, 11)$ vs its mirror $(11, 1)$ — probe trained on layer 1 tested on layer 11 stays warmer than the reverse. This mirrored-pair contrast is the actual cumulative-feature signature.
>
> The paper's "consistent though model-dependent" qualifier reflects this: TabICL and TabPFN v1 show clean long-range asymmetry; TabPFN(2.5), LIMIX-16M show it weakly through block structure; TabPFN v2 is the noisiest. The visually dominant block structure can mask the long-range asymmetry on first glance.

- **4. Tabular Logit Lens** — emit a real prediction after every layer to find *when* the answer is essentially formed.
    - **Why a new lens?** The LLM "logit lens" (reuse the final decoder at every layer) is brittle on TFMs because TFM decoders are tightly tied to the final-layer basis — early-layer hidden states aren't in that basis. The "tuned lens" (a per-layer affine translator) is also weak here for the same reason.
    - **Method:** for each layer $\ell$, train a **fresh decoder** $D_\ell$ (same architecture as the model's own head) on synthetic datasets from the **TabICL prior** to map $h_\ell \to$ class probabilities. Model stays frozen.
    - **Inference:** forward the support+query through layers $1..\ell$, apply $D_\ell$ → a real prediction in the model's own output space at depth $\ell$.
    - **What it measures:** per-layer AUC curve; AUC rises sharply in the first few layers for all six TFMs.
    - **Defines the *prediction-ensembling* stage:** depth range where per-layer decoder AUC has saturated but the *original* decoder is still catching up — the residual stream still needs alignment with the original-decoder basis.
    - **Limitation:** TabICL-prior decoders may under-serve TFMs with richer priors (TabPFN(2.5), LimiX). See [tabular-logit-lens](tabular-logit-lens.md).

    ![Per-layer-decoder ROC-AUC across the six TFMs](assets/balef2026onelayer-logit-lens.png)

    **Takeaway 4:** Representations are already formed for a reliable prediction in the early layers, but not necessarily aligned with the original decoder. *Orange: per-layer decoder; blue: original final decoder; the gap marks the prediction-ensembling stage.*

#### Probe vs lens — when to use which

| Aspect | Probe (Exp 3) | Lens (Exp 4) |
|---|---|---|
| **Intuitive idea** | External classifier on frozen layer | Model emits a prediction *as if* this were the final layer |
| **Asks** | Is the label encoded here? (information-theoretic) | Would the model emit the label here? (functional) |
| **Unique power** | Cross-layer transfer ($i \to j$) → distinguish cumulative vs replaced features | Real prediction in the model's output space → defines the *prediction-ensembling* gap vs the original decoder |

The **gap between them** is itself the diagnostic: where the info is encoded but not yet aligned with the model's own decoder — a finding neither tool delivers alone.

### C. Intervene on the layers (causal tests)

- **5. Layer ablation** — intervene on the forward pass and measure the change in downstream-task performance.
    - **Three interventions, each answering a different question:**
        - **Skip layer $\ell$:** $h_{\ell+1} = h_{\ell-1}$ (residual passes through, layer's transformation removed). Tests **necessity** — if performance drops, the layer's computation was needed.
        - **Repeat layer $\ell$:** $h_{\ell+1} = \text{Layer}_\ell(\text{Layer}_\ell(h_{\ell-1}))$ (apply twice). Tests **reusability** — if performance is preserved or improves, the layer's transformation maps its own output back to a valid input distribution → it can serve as a loop body.
        - **Swap layers $\ell, \ell+1$:** apply $\ell+1$ before $\ell$. Tests **position-specificity** — if performance drops, the layer is specialized to its place in the stack; if it survives, the layer is order-free.
    - **What it gives over Groups A/B:** the first **causal** evidence. A and B can show that a layer's representation *looks* unique or *contains* the label; only ablation shows whether the model's prediction actually *depends* on the layer being there.
    - **Findings:**
        - **Skip:** early layers (especially the first) are catastrophic to skip; middle/late layers are mostly safe. Exception: **TabICL and LIMIX-2M** survive early skips because their upstream encoders (row-wise interaction; RBF preprocessing) already perform the *latent-mapping* role.
        - **Repeat:** helps **LIMIX-16M** and **TabPFN v1** — direct empirical seed for the looped-transformer follow-up in §6.
        - **Swap:** universally hurts across all TFMs; sensitivity is greater than in LLMs, especially pronounced in **TabPFN v2** → TFMs are far more order-specialized than LLMs.
    - **Caveat:** single-layer interventions only — does not test multi-layer ablations, which could compound or cancel.

    ![Layer-ablation results across six TFMs](assets/balef2026onelayer-layer-interventions.png)

    **Takeaway 5:** Early layers contribute the most, while later layers perform iterative refinement of the representation. *TabICL and LIMIX-2M robust to early skips; swaps universally hurt; repeats help LIMIX-16M and TabPFN v1.*
- **6. Self-repair** — after skipping layer $\ell$, run the tabular logit lens at each subsequent layer to see whether later layers *recover* the lost prediction.
    - **Motivation — disambiguate Exp 5.** Exp 5 measures only *final-layer* accuracy, conflating two stories (layer was unnecessary vs. downstream layers compensated). Distinguishing them is what evidences iterative inference and licenses the looped-transformer follow-up.
    - **Technique — lens-after-skip trajectory:**
        1. Pick target layer $\ell^*$ to ablate; apply residual identity $h_{\ell^*+1} = h_{\ell^*-1}$.
        2. At every subsequent layer $\ell \geq \ell^*+1$, apply the Exp 4 per-layer decoder $D_\ell$ → real AUC.
        3. Repeat for every choice of $\ell^*$; compare each post-skip trace against the no-skip baseline.
    - **Reading the trace:**
        - **Dip immediately after skip + recovery toward baseline** → self-repair; depth where AUC re-joins baseline = *repair distance*.
        - **No dip** → pure redundancy (layer wasn't contributing).
        - **Dip with no recovery** → layer uniquely necessary (no safety net).
    - **Why the lens is the right instrument:** it produces a per-layer prediction trajectory, not a single endpoint. Final-layer accuracy alone collapses redundancy and self-repair into the same number; the trajectory makes them visually separable as different curve shapes.
    - **Findings:** early-layer skips show **no recovery** (first layer = unique functionality); middle/late-layer skips show **clear recovery**, especially in TabPFN v2 → middle/late depth-redundancy is real and active, not merely unused.

    ![Self-repair under layer skipping](assets/balef2026onelayer-self-repair.png)

    **Takeaway 6:** Self-repair generally occurs after layer ablations, except for the first layer. *Solid black: no-skip baseline; colored lines: post-skip trajectories, crosses mark the skipped layer; dashed line connects post-skip dip to baseline.*

### How the six combine to answer RQ1

- **A (what's there):** separation grows monotonically; block structure in deep models.
- **B (when it's formed):** probe and lens saturate early; features transfer forward, not backward.
- **C (what's required):** early layers crucial, middle/late skippable but actively self-repairing, swaps universally hurt.
- **→ Iterative inference.** Predictions form early; later layers refine via overlapping (not distinct) computations — depth is largely redundant, motivating the looped-block follow-up.

## Relation to Prior Work

- **LLM mechanistic interpretability** (logit lens, tuned lens, probing classifiers, layer ablation): provides the toolbox; this paper ports the toolbox to TFMs and re-discovers shared structure (early-critical, block-redundant) plus TFM-specific deviations (more middle redundancy, more swap sensitivity).
- **[tabpfnv1](hollmann2023tabpfnv1.md), [tabpfnv2](hollmann2025tabpfnv2.md), [tabicl](qu2025tabicl.md), TabPFN(2.5), LIMIX-2M/16M:** all six are treated as black-box subjects of analysis. The paper does not propose a new TFM; it diagnoses where each one's depth is or is not contributing.
- **Looped transformers:** the layer-repeat experiment provides the empirical seed; the looped-nanoTabPFN follow-up is the constructive proof-of-concept. See [looped-transformer-tfm](looped-transformer-tfm.md).
- **Tuned lens (Belrose 2023) / logit lens (nostalgebraist 2020):** the [tabular-logit-lens](tabular-logit-lens.md) adapts the idea to TFMs by continue-pretraining a fresh decoder per layer on TabICL priors, since TFM final decoders are too tightly tied to the final-layer basis to be reused directly.

## Entities & Concepts

- [tabular-logit-lens](tabular-logit-lens.md) — per-layer decoders for TFM mechanistic interpretability
- [looped-transformer-tfm](looped-transformer-tfm.md) — recurrent single-block TFM design
- [tfm-inference-stages](tfm-inference-stages.md) — four-stage TFM inference taxonomy
- [tabular-learning](tabular-learning.md) — TFM family context
- [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md), [hollmann2025tabpfnv2](hollmann2025tabpfnv2.md), [qu2025tabicl](qu2025tabicl.md), [qu2026tabiclv2](qu2026tabiclv2.md), [muller2022pfn](muller2022pfn.md)
