---
title: "Transformers Can Do Bayesian Inference"
tags: [source, tabular, in-context-learning, bayesian-inference, pfn]
sources: [muller2022pfn]
updated: 2026-05-08
---

# Transformers Can Do Bayesian Inference

**Source:** https://arxiv.org/abs/2112.10510
**Title:** Transformers Can Do Bayesian Inference
**Date ingested:** 2026-05-08
**Type:** paper
**Authors:** Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, Frank Hutter
**Venue:** ICLR 2022

## Summary

- **What:** small-data prediction needs a useful prior, but both prior design and Bayesian inference under it are hard.
- **How:** amortize inference — pretrain a Transformer on synthetic samples from the prior so it directly approximates the posterior predictive in one forward pass.
- **So what:** Prior-Data Fitted Networks (PFNs); 200–8000× faster than MCMC at matching accuracy; foundational framework behind TabPFN, TabICL, and KumoRFM's ICL module.

![[Pasted image 20260508093314.png]]
## Challenges & Novelty

> *"Use a large amount of [synthetic] data to later yield strong performance on tasks for which only small datasets are available."* — the opening sentence that frames every choice below.

### Three framing questions

- **Why Bayesian?** small-data is the target regime.
    - data alone underdetermines the answer at small N; the prior fills the gap.
    - ERM/GBDTs assume enough data to pin down the function — degrade at small N.
    - Bayes is the only principled prior + data → calibrated prediction.
- **Why is "baking in the prior" the central challenge?** in the PFN framework, the prior is the *only* design knob.
    - objective (Prior-Data NLL), architecture (perm-invariant Transformer), inference (one forward pass) are essentially forced.
    - all inductive bias (smoothness, sparsity, causal structure, Occam) must enter through $p(\mathcal{D})$.
    - wrong prior → calibration guarantees apply to the wrong posterior.
    - the TabPFN/TabICL lineage = a sequence of richer priors (BNN → SCM+BNN → 130M corpus → Cauchy-DAG + 8 function families).
- **Why can't classical Bayes do this?** cost structure is wrong for small-data deployment.
    - MCMC: sequential, slow; VI: biased; both need prior *density*, not just samples.
    - PFNs invert it: huge offline pretraining once → amortized to one forward pass at deploy.
    - only prerequisite: **sample** from the prior (almost always easy).

### Three core contributions

- **Theorem.** Prior-Data NLL ≡ KL to the true PPD.
    - training on synthetic samples provably converges to the exact PPD.
    - elevates the method from a heuristic to principled amortized Bayes.
    - the linchpin every downstream PFN paper reuses without re-proving.
- **Permutation-invariant Transformer.**
    - context $D = \{(x_i, y_i)\}$ has no canonical order → drop positional encodings.
    - matches the i.i.d. assumption baked into the PPD.
- **Riemann Distribution for regression.**
    - discretize continuous target into $B$ equi-probable bins → cross-entropy.
    - approximates any Riemann-integrable density as $B \to \infty$.
    - reused throughout the lineage (interval-mapping in tabpfnv1, 999-quantile head in tabiclv2).

## Relation to Prior Work

| Method                                          | Prior requirement      | Inference cost            | PPD or posterior             |
| ----------------------------------------------- | ---------------------- | ------------------------- | ---------------------------- |
| MCMC (NUTS)                                     | Non-normalized density | High (sequential)         | Posterior samples            |
| Variational Inference                           | Joint density          | Medium (iterative)        | Approximate posterior        |
| Neural Processes (ANP)                          | Amortized samples      | Low (forward pass)        | Approximate PPD              |
| **PFN (this paper)**                            | Samples only           | Low (single forward pass) | Approximate PPD              |
| [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md) | SCM+BNN samples        | Low                       | PPD (tabular classification) |

- [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md): TabPFN is a direct application of PFNs to tabular classification — same paradigm, with a richer prior (SCMs + BNNs with Occam's razor bias) and scaled to real-world benchmarks.
- [hollmann2025tabpfnv2](hollmann2025tabpfnv2.md): extends PFN pretraining with 130M synthetic datasets and alternating row/column attention; PFN framework is unchanged.
- [qu2025tabicl](qu2025tabicl.md): TabICL uses PFN-style ICL but decouples the column embedding from the ICL Transformer, scaling to 500K samples.

## Technical Details

**Notation.** $t$ denotes the latent task — the underlying data-generating mechanism (e.g. a GP with fixed hyperparameters, or a BNN with fixed weights). The prior $p(t)$ encodes beliefs over which tasks are plausible.

**PPD.** The target quantity is the posterior predictive distribution, obtained by marginalizing the likelihood over the posterior on the latent task $t$:

$$p(y \mid x, \mathcal{D}) = \int p(y \mid x, t)\, p(t \mid \mathcal{D})\, dt$$

where $t$ is the latent data-generating mechanism (e.g. a GP with specific hyperparameters, or a BNN with specific weights), and $p(t \mid \mathcal{D}) \propto p(\mathcal{D} \mid t)\, p(t)$ is the posterior over tasks given the observed data. 

This integral is intractable for most priors. PFNs learn to approximate it directly.
**PFN training.** Given a prior $p(\mathcal{D})$ from which datasets can be sampled, train $q_\theta$ to minimize the Prior-Data NLL:

$$\ell_\theta = \mathbb{E}_{D \cup \{x,y\} \sim p(\mathcal{D})}\left[-\log q_\theta(y \mid x, D)\right]$$

![[Pasted image 20260514215851.png]]
**Theoretical guarantee.** Minimizing $\ell_\theta$ is equivalent to minimizing:

$$\mathbb{E}_{x,D \sim p(\mathcal{D})}\left[\text{KL}\!\left(p(\cdot|x,D),\; q_\theta(\cdot|x,D)\right)\right] + \text{const}$$

If the model family $q_\theta$ is expressive enough, the optimum satisfies $q_{\theta^*}(\cdot|x,D) = p(\cdot|x,D)$ exactly.

**Architecture.** Standard Transformer encoder without positional encodings — permutation-invariant by design. Context tokens are $(x_i, y_i)$ pairs; query tokens are $(x_j, ?)$. Context tokens attend to each other and to query tokens; query tokens attend only to context (causal-style masking separating context from query).

**Riemann Distribution.** For regression, the output distribution is a discretized distribution over $B$ bins. Bin boundaries are chosen so each bin has equal prior probability $p(y \in b) = 1/B$ under the prior (estimated from large prior samples). Tails are replaced with half-normal distributions for unbounded support.

> **Note — why equi-probable bins?** Equi-width bins on a skewed prior leave many bins almost empty, so cross-entropy sees little signal there. Equi-probable bins guarantee every bin gets roughly the same number of training samples, so the categorical head is uniformly informed. As $B \to \infty$ the discretization approximates any Riemann-integrable density to arbitrary precision (hence the name) — it's the prototypical trick for using cross-entropy on continuous targets, later reused as interval-mapping in [hollmann2023tabpfnv1](hollmann2023tabpfnv1.md) and generalized to a 999-quantile + pinball-loss head in [qu2026tabiclv2](qu2026tabiclv2.md).

**Prior design.** For GP approximation: sample hyperparameters → sample function → evaluate at random inputs. For BNN: sample architecture and weights → forward-pass random inputs. For tabular: BNN prior with architecture hyper-prior (embedding size, depth, activations sampled jointly).

## Experiments

- GP approximation with fixed hyperparameters: PFN's confidence intervals and means are visually indistinguishable from the exact GP PPD.
- GP with hyper-priors (no closed form): PFN matches or exceeds MLE-II (MAP) at 200× speed; outperforms NUTS at 1000–8000× speed.
- BNN posterior approximation: 1000× faster than Bayes-by-Backprop SVI; 10000× faster than NUTS at the same Prior-Data NLL.
- Small tabular classification (20 OpenML datasets, N=30 training / 70 test): PFN-BNN outperforms XGBoost, CatBoost, GPs, and KNN — without any per-dataset hyperparameter search.
- Few-shot Omniglot (5-way 5-shot): PFN with handwriting stroke prior + fine-tuning reaches state-of-the-art accuracy.

## Entities & Concepts

- [tabular-learning](tabular-learning.md)
- [tabular-icl-lineage](tabular-icl-lineage.md) — comparison across the PFN → TabPFN → TabICL lineage
