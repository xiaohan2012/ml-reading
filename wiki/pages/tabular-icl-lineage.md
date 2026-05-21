---
title: "Tabular ICL Lineage: PFN → TabPFN → TabICL"
tags: [analysis, tabular, in-context-learning, foundation-model, pfn]
sources: [muller2022pfn, hollmann2023tabpfnv1, hollmann2025tabpfnv2, qu2025tabicl, qu2026tabiclv2]
updated: 2026-05-14
---

# Tabular ICL Lineage: PFN → TabPFN → TabICL

A side-by-side comparison of the five papers that define the Prior-Data Fitted Network (PFN) lineage for tabular in-context learning: [muller2022pfn](muller2022pfn.md), [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md), [hollmann2025tabpfnv2](hollmann2025tabpfnv2.md), [qu2025tabicl](qu2025tabicl.md), [qu2026tabiclv2](qu2026tabiclv2.md).

## Shared Backbone

All five papers share the same core recipe:

- **PFN paradigm** ([muller2022pfn](muller2022pfn.md)): train $q_\theta$ to minimize the Prior-Data NLL $\mathbb{E}_{D \sim p(\mathcal{D})}[-\log q_\theta(y \mid x, D)]$ — provably equivalent to minimizing KL against the true posterior predictive distribution. The only prior requirement is the ability to *sample*.
- **Inference recipe:** synthetic-only pretraining; at test time pass $\mathcal{D}_\text{train}$ as context tokens; one forward pass; no gradient updates.
- **Permutation invariance** over context rows (no positional encoding on the row axis).

## Side-by-side Comparison

### Architecture

| Axis | [tabpfnv1](hollmann2023tabpfnv1.md) | [tabpfnv2](hollmann2025tabpfnv2.md) | [tabicl](qu2025tabicl.md) | [tabiclv2](qu2026tabiclv2.md) |
|---|---|---|---|---|
| **Stages** | **1** (one Transformer stack) | **1** stack of `L` layers, each alternating col-attn ↔ row-attn | **3** sequential: $\mathrm{TF}_{\text{col}} \to \mathrm{TF}_{\text{row}} \to \mathrm{TF}_{\text{icl}}$ | Same skeleton as TabICL v1 |
| **Token granularity** | One token *per row* (columns folded into hidden dim by `Linear(M_max → d)`) | One token *per cell* `(N, M, d)` | One token *per cell* `(N, M, d)` | One token *per cell* — but cell = grouped triple |
| **Token shape evolution** | `(N, d)` for all `L` layers | `(N, M, d)` for all `L` layers | `(N, M, d)` → `(N, 512)` after $\mathrm{TF}_{\text{row}}$ | Same as TabICL v1 |
| **Attention factorization** | None — joint attention over `N` rows only | **Interleaved** col-attn ↔ row-attn within each of `L` layers | **Sequential** col → row → ICL, no interleaving | Same as TabICL v1 |
| **Where columns are mixed** | Pre-encoder by a linear layer; never again | Every layer (col-attn) | Once, inside $\mathrm{TF}_{\text{row}}$ (per-row Transformer over the `M` feature tokens) | Same as TabICL v1 |
| **Where ICL happens** | Implicit — same Transformer encodes and reasons | Implicit — alternating loop interleaves them | Explicit dedicated final stage $\mathrm{TF}_{\text{icl}}$ over row-vectors only (no `M` axis) | Same as TabICL v1 |
| **Y-injection** | Additive: `token = embed(x) + embed(y)` at encoder | $y$ as extra column ($F{+}1$), test-row $y$ is NaN | At $\mathrm{TF}_{\text{icl}}$ only ($\mathrm{Embed}_{\text{ICL}}$, additive on row vectors) | **Both** pre-$\mathrm{TF}_{\text{col}}$ (per-cell $\mathrm{Embed}_{\text{TAE}}$) *and* pre-$\mathrm{TF}_{\text{icl}}$ (inherited $\mathrm{Embed}_{\text{ICL}}$) |
| **Attention complexity** | $O(N^2)$ | $O(N^2 M + N M^2)$ × `L` | $O(N M k + N M^2 + N^2)$ | Same as TabICL v1 |

### Scalability, capabilities, training

| Axis | [tabpfnv1](hollmann2023tabpfnv1.md) | [tabpfnv2](hollmann2025tabpfnv2.md) | [tabicl](qu2025tabicl.md) | [tabiclv2](qu2026tabiclv2.md) |
|---|---|---|---|---|
| **Practical $N$** | ~1K | ~10K | ~500K | ~1M |
| **Classes** | ≤10 | ≤10 | hierarchical tree | mixed-radix + hierarchical |
| **Regression** | n/a | learned bar | n/a | 999-quantile + pinball |
| **Schema-agnostic?** | No (fixed column embedding) | Yes (randomized attribute tokens) | Yes (per-column ISAB + RoPE) | Yes |
| **Prior** | SCM + BNN | SCM + BNN, 130M datasets | SCM + tree-SCM | Cauchy-DAG + 8 function families, bootstrap-filtered |
| **Optimizer / training trick** | preprocessing + ensembling | 130M-dataset pretraining | curriculum 1K→40K→60K | Muon + cautious WD + curriculum |
| **Venue** | ICLR 2023 | Nature 2025 | ICML 2025 | ICML 2026 |

## The Research Arc

### 1. muller2022pfn → tabpfnv1 — apply PFN to real tabular data

- Adds the **SCM prior** (causal DAG structure beats plain BNN; ~2% gain).
- **Multi-class** via random interval-mapping of continuous prior labels.
- Preprocessing (z-score, Yeo-Johnson) + 32-way ensembling over feature/class rotations.
- Faster attention pattern: training tokens attend to each other ($n^2$); test tokens attend only to training ($n \cdot m$).

### 2. tabpfnv1 → tabpfnv2 — scale and zero-shot schema

- **Alternating row/column attention** drops complexity from $N^2 M^2$ joint attention to $N^2 M + N M^2$.
- **Randomized attribute tokens** replace learned per-column embeddings — enables arbitrary schemas at inference.
- 130M synthetic datasets; mixed types; regression support; missing values handled in-context.
- New cap: N ≤ 10K.

### 3. tabpfnv2 → tabicl — break the 10K-row wall

The first paper to **decouple column embedding from ICL** with a 3-stage pipeline:

- $\mathrm{TF}_{\text{col}}$ — Set Transformer (ISAB) per column; per-cell distribution-aware embedding.
- $\mathrm{TF}_{\text{row}}$ — collapse features into a 512-dim row vector via 4 [CLS] tokens; RoPE prevents representation collapse when features share distributions.
- $\mathrm{TF}_{\text{icl}}$ — ICL over row embeddings only; column dimension eliminated from the quadratic step.

Result: scales to 500K samples; up to 10× faster than tabpfnv2 on large tables; curriculum learning (1K→40K→60K) is essential.

### 4. tabicl → tabiclv2 — sharpen, scale further, open SOTA

Keeps the 3-stage skeleton; adds five orthogonal improvements:

- **Repeated feature grouping** — overlapping column triples (circular shifts $(0,1,3) \bmod m$) break symmetry without losing per-feature resolution.
- **Target-aware embedding** — inject $y_i$ into training tokens *before* $\mathrm{TF}_{\text{col}}$, so column embedding is task-conditioned (vs. tabpfnv2's late appended-column trick).
- **QASSMax** — query-aware scalable softmax fixes *attention fading* at long context via $\log n$ rescaling + per-element gating.
- **Mixed-radix ensembling** + **999-quantile regression head** — unlock arbitrary class counts and full distributional regression.
- **Muon optimizer** + cautious weight decay; richer prior (Cauchy-graph DAGs + 8 function families); fully open weights.

Beats closed-source RealTabPFN-2.5 with no per-dataset tuning; scales to 1M rows via disk offloading.

## Two Orthogonal Scaling Strategies

The lineage splits into two complementary approaches to attention complexity:

- **TabPFN line (v1 → v2):** keep one Transformer; reduce complexity by **factoring attention** (alternating row || column) and randomize column identities. Practical limit: ~10K rows.
- **TabICL line (v1 → v2):** **stack specialized stages** (col-set → row-CLS → ICL); collapse the column dimension before quadratic ICL. Practical limit: ~1M rows.

Both rest on the muller2022pfn training objective. The differences are entirely in (a) what synthetic prior the network is trained on and (b) how attention is structured to handle larger, more heterogeneous tables.

## Downstream Influence

- [fey2025kumorfm2](fey2025kumorfm2.md): the TabICL 3-stage col→row→ICL pattern is the direct structural ancestor of KumoRFM-2's Stage 1 (intra-table row/column attention) + Stage 2 (cross-sample FK attention), extending the pattern from flat tables to relational databases. Target-aware embedding parallels KumoRFM-2's early label injection.

## Related Pages

- [tabular-learning](tabular-learning.md) — broader survey of tabular foundation model approaches (CARTE, SAINT, etc. beyond the PFN line)
- [relational-foundation-model](relational-foundation-model.md) — extending the FM paradigm from flat tables to multi-table relational data
