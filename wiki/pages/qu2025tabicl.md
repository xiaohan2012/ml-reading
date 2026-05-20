---
title: "TabICL: A Tabular Foundation Model for In-Context Learning on Large Data"
tags: [source, tabular, in-context-learning, foundation-model, transformer]
sources: [qu2025tabicl]
updated: 2026-05-06
---

# TabICL: A Tabular Foundation Model for In-Context Learning on Large Data

**Source:** https://arxiv.org/abs/2502.05564
**Title:** TabICL: A Tabular Foundation Model for In-Context Learning on Large Data
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Jingang Qu, Jiashuo Luo, Guo-Wei Yan, Taoran Sun, Chuanhui Hao, Deyuan Xue, Longbing Liu, Junfeng Chen, Yuantao Yin
**Venue:** ICML 2025

![[Pasted image 20260520095310.png]]
## Summary

- **What:** tabpfn-v2's interleaved row/column attention is $O(N^2 M)$ per layer — caps tables at $N \approx 10\text{K}$.
- **How:** 3-stage pipeline that collapses the column dim *before* ICL:
    - *TF_col* — Set Transformer per column → distribution-aware cell embeddings.
    - *TF_row* — per-row Transformer + 4 `[CLS]` → 512-d row vector. Cross-feature interactions extracted once, here.
    - *TF_icl* — Transformer over row vectors only, $O(N^2)$, no $M$.
- **So what:** Scales to $N \le 500\text{K}$; matches v2 overall, beats it on $N > 10\text{K}$; col → row → ICL pattern is the ancestor of KumoRFM-2's Stage 1 + Stage 2.

## Relation to Prior Work

| Method | N limit | Col embedding | Row embedding | ICL stage |
|---|---|---|---|---|
| [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md) | 1K | Joint | Joint | Joint |
| [hollmann2025tabpfnv2](hollmann2025tabpfnv2.md) | 10K | Alternating | Alternating | Alternating |
| **TabICL** | 500K | TF_col | TF_row | TF_icl |

- [hollmann2025tabpfnv2](hollmann2025tabpfnv2.md): TabICL targets the remaining N>10K gap; same PFN/ICL paradigm, more scalable by separating column and ICL stages.
- [fey2025kumorfm2](fey2025kumorfm2.md): KumoRFM-2's hierarchical design mirrors TabICL's 3-stage logic — Stage 1 (col+row attention within a table) then Stage 2 (FK+cross-sample attention across tables), extending the pattern from flat tables to relational databases.

### vs TabPFN v2

| Aspect                       | TabPFN v2                                                               | TabICL                                                                                                                |
| ---------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Token grid                   | Per-cell `(B, S, F, d)` kept through all $L$ layers                     | Per-cell only in TF_col / TF_row; collapsed to `(N, 512)` before TF_icl                                               |
| Attention pattern            | **Alternating loop**: col-attn → row-attn → MLP, repeated $L$ times     | **Sequential pipeline**: TF_col (set-attn over column values) → TF_row (per-row cross-feature) → TF_icl (per-row ICL) |
| Cross-feature × ICL coupling | Interleaved: row-attn at layer $\ell$ can inform col-attn at $\ell{+}1$ | Frozen: feature interactions baked into the 512-d row vector before any ICL context is seen                           |
| Column identity              | Random vector resampled per call (architectural invariance)             | RoPE over column axis + permutation ensembling (partial invariance)                                                   |
| Dominant cost                | $O(N^2 M + N M^2)$ per layer                                            | $O(N^2)$ in TF_icl; column work amortized in $O(NMk)$                                                                 |
| Practical $N$                | ≤ 10K                                                                   | ≤ 500K                                                                                                                |

## Technical Details

| Stage  | Layers                                            | Attends over                  | Input shape  | Output shape                    |
| ------ | ------------------------------------------------- | ----------------------------- | ------------ | ------------------------------- |
| TF_col | ISAB (Set Transformer), $k{=}128$ inducing points | $N$ values within one column  | $(N, M)$ raw | $(N, M, d)$ per-cell embeddings |
| TF_row | 3-layer, 8-head Transformer                       | $M{+}4$ tokens within one row | $(N, M, d)$  | $(N, 512)$ row vectors          |
| TF_icl | 12-layer Transformer                              | $N$ row vectors               | $(N, 512)$   | per-test-row class probs        |

**TF_col — distribution-aware cell embedding.**

- Each column $c_j \in \mathbb{R}^N$ is treated as an unordered set of scalars.
- ISAB reduces $O(N^2) \to O(Nk)$ and conditions each cell on its column's full empirical distribution.
    - **Inducing points** ($k{=}128$): a learned $I \in \mathbb{R}^{k \times d}$ bottleneck shared across all columns.
    - Two linear-in-$N$ attentions — set $\to I$ (summarize) then $I \to$ set (broadcast) — let every value "see" the whole column via $k$ summary vectors instead of full $N \times N$ attention.
- Output per cell: an affine $(W_{ij}, B_{ij})$ applied as $e_{ij} = W_{ij} \odot c_{ij} + B_{ij}$.
- Rows are not mixed; columns are not mixed with each other.

**TF_row — collapse columns into a row vector.**

- Per row $i$: prepend 4 learnable `[CLS]` tokens to the $M$ feature embeddings.
- Add **RoPE (Rotary Positional Embedding)** over the column axis — columns are otherwise exchangeable, so positional identity is needed to avoid representation collapse. (TabPFN v2 solves the same problem with random per-column vectors — see [Appendix: Column identifier choices](#appendix--column-identifier-choices-rope-vs-random-noise).)
- 3-layer Transformer attends features-to-features *within a single row* — the only place cross-feature interactions are extracted.
- The 4 `[CLS]` outputs concatenate into a 512-d row vector. From here, no component can revisit feature $j$ of row $i$.

> **Why 4 `<CLS>` tokens -- 4 × 128 and not 1 × 512?**
> - Attention forces all tokens to share dim — 1 × 512 would run all of TF_row at $d{=}512$, 4× the $O(NM^2 d)$ cost.
> - 4 × 128 keeps TF_row at $d{=}128$, widens only at the final concat → same 512-d output, none of the internal cost.
> - Bonus: the 4 tokens act as parallel pooling queries that can specialize, like multi-head attention.

> **What is RoPE?**
> - Rotates Q/K in 2-d pairs by an angle tied to position $p$ — instead of adding a positional vector.
> - Dot product depends only on **relative** offset $p{-}q$; V untouched, no extra parameters.
> - Standard in modern LLMs (LLaMA, Mistral); from Su et al., RoFormer 2023.
> - **In TabICL:** rotates by column index → columns with identical distributions get distinct attention scores, breaking the symmetry that causes collapse.
> - Cost: permutation invariance lost, partly recovered via column-permutation ensembling.

**TF_icl — in-context learning over rows.**

- Training-row labels are one-hot encoded and added to their row vectors.
- Causal masking: test rows attend to all training rows; training rows attend only among themselves.
- An MLP head maps each test row's final vector to class probabilities.
- $O(N^2)$ in $N$ alone — the $M$ factor is no longer in this stage's attention.

**Complexity per stage** ($N$ rows, $M$ columns, $d$ hidden dim, $k{=}128$ inducing points):

| Stage | Cost | Scaling |
|---|---|---|
| TF_col | $O(N M k d)$ | linear in $N$ — ISAB bottleneck |
| TF_row | $O(N M^2 d)$ | quadratic in $M$, linear in $N$ — applied per row |
| TF_icl | $O(N^2 d)$ | quadratic in $N$, **no $M$** — column dimension already collapsed |

- Total: $O(NMkd + NM^2 d + N^2 d)$; dominated by $N^2 d$ when $N \gg M$.
- vs TabPFN v2: $O(L \cdot (N^2 M + N M^2) \cdot d)$ — the $N^2 M$ term, repeated $L$ times, is what caps v2 at $N \approx 10\text{K}$.

**Pretraining.** Exclusively synthetic: SCMs (TabPFN v1 style) + tree-based SCMs (XGBoost-based generation). Curriculum: 1K → 40K → 60K samples over 3 stages, 20 days on 3×A100. (See [Appendix: Curriculum learning](#appendix--curriculum-learning-1k--40k--60k) for what the curriculum is and why direct training at 60K fails.)

## Experiments

- On TALENT (200 classification datasets): TabICL matches TabPFN v2 overall and outperforms both TabPFN v2 and CatBoost on the 53 large datasets (N>10K).
- Up to 10× faster than TabPFN v2.

## Entities & Concepts

- [tabular-learning](tabular-learning.md)
- [tabular-icl-lineage](tabular-icl-lineage.md) — comparison across the PFN → TabPFN → TabICL lineage
- [tabicl-code](tabicl-code.md) — code-grounded walkthrough of the TabICL architecture

## Appendix — Column identifier choices: RoPE vs random noise

Both TabICL and TabPFN v2 face the same problem: per-column tokens need *some* identifier so attention can tell columns apart. They pick opposite trade-offs.

| Approach | Identifier | Schema invariance | Inference stability | Training |
|---|---|---|---|---|
| **Random noise per call** (TabPFN v2) | Fresh random vector per column, every forward pass | **Architectural** — pretraining can't bake in "column 0 means X" | Multiple calls give different outputs → need ensembling for stable predictions | Harder — model must extract structure from uninformative IDs |
| **RoPE** (TabICL) | Deterministic rotation by column index $j$ | **Statistical only** — pretraining can learn position-specific biases | Stable single pass; column-permutation ensembling needed at inference to recover invariance | Easier — deterministic IDs reduce training noise |

Why each picked what it picked:

- **TabPFN v2:** cells stay tokens through all $L$ layers → any per-column bias compounds. Architectural invariance is more valuable. v2 already ensembles at inference for accuracy → "random per call" is essentially free.
- **TabICL:** already needs a 1K → 40K → 60K curriculum to converge at scale; deterministic identifiers reduce one more axis of training noise. RoPE is also drop-in standard tooling from modern LLMs.

Neither is strictly better — the trade-off is "extra inference ensembling (RoPE)" vs "harder training + per-call randomness (v2-style)."

## Appendix — Curriculum learning ("1K → 40K → 60K")

- **Curriculum learning** (Bengio et al. 2009): present easy examples first, then progressively harder ones, instead of shuffling all difficulties together.
- **In TabICL,** "difficulty" = table size $N$. Three stages:

| Stage | Table size ($N$) | Role |
|---|---|---|
| 1 | up to 1K samples | Learn coarse ICL behavior on small tables (the regime TabPFN v1/v2 already handle) |
| 2 | up to 40K samples | Scale up — model learns to use longer contexts |
| 3 | up to 60K samples | Final tuning at the large-table regime that's TabICL's actual target |
- **Why staged:** small tables give a tight learning signal — the "test row attends to similar training rows" pattern is easy to discover at $N{=}1$K. Direct training at $N{=}60$K fails to converge: gradients are too diffuse, attention patterns too ambiguous from random init.
- **Implication:** scaling tabular ICL is an *optimization* problem, not just a memory/compute one. Most foundation models shuffle scales; TabICL's curriculum requirement is itself a finding.
