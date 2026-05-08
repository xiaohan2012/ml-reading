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

- **What:** The Bayesian PPD $p(y \mid x, \mathcal{D}) = \int p(y \mid x, t)\, p(t \mid \mathcal{D})\, dt$ is intractable for most priors, requiring slow MCMC or biased variational inference.
- **How:** Approximate $p(y \mid x, \mathcal{D})$ directly by training a function $q_\theta$ on synthetic datasets from the prior by minimizing $\ell_\theta = \mathbb{E}_{D \cup \{x,y\} \sim p(\mathcal{D})}[-\log q_\theta(y \mid x, D)]$; 
	- at inference, pass $\mathcal{D}_\text{train}$ as context tokens and predict in a single forward pass — no gradient updates.
- **So what:** Introduces Prior-Data Fitted Networks (PFNs), the foundational framework behind TabPFN and KumoRFM's ICL module; 200–8000× faster than MCMC while matching or exceeding its accuracy.

![[Pasted image 20260508093314.png]]
## Challenges & Novelty

Bayesian inference is theoretically well-grounded but computationally intractable for most real priors — MCMC requires many sequential evaluations, and variational inference requires access to the joint density. PFNs circumvent both requirements: the only prerequisite is the ability to *sample* from the prior, which is almost always easy.

- **Prior-Data NLL as a PPD objective:** minimizing $\ell_\theta = \mathbb{E}_{D \cup \{x,y\} \sim p(\mathcal{D})}[-\log q_\theta(y|x,D)]$ is provably equivalent to minimizing the expected KL divergence between the exact PPD and the model's approximation — the model learns Bayesian inference, not just function fitting.
- **Permutation-invariant Transformer:** the context dataset $D = \{(x_i, y_i)\}$ has no canonical order; dropping positional encodings makes the model invariant to dataset permutation without any additional architectural machinery.
- **Riemann Distribution for regression:** a discretized continuous distribution whose bucket boundaries are chosen to be equi-probable under the prior; provably approximates any Riemann-integrable density to arbitrary precision, while remaining compatible with standard cross-entropy training.

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

**Theoretical guarantee.** Minimizing $\ell_\theta$ is equivalent to minimizing:

$$\mathbb{E}_{x,D \sim p(\mathcal{D})}\left[\text{KL}\!\left(p(\cdot|x,D),\; q_\theta(\cdot|x,D)\right)\right] + \text{const}$$

If the model family $q_\theta$ is expressive enough, the optimum satisfies $q_{\theta^*}(\cdot|x,D) = p(\cdot|x,D)$ exactly.

**Architecture.** Standard Transformer encoder without positional encodings — permutation-invariant by design. Context tokens are $(x_i, y_i)$ pairs; query tokens are $(x_j, ?)$. Context tokens attend to each other and to query tokens; query tokens attend only to context (causal-style masking separating context from query).

**Riemann Distribution.** For regression, the output distribution is a discretized distribution over $B$ bins. Bin boundaries are chosen so each bin has equal prior probability $p(y \in b) = 1/B$ under the prior (estimated from large prior samples). Tails are replaced with half-normal distributions for unbounded support.

**Prior design.** For GP approximation: sample hyperparameters → sample function → evaluate at random inputs. For BNN: sample architecture and weights → forward-pass random inputs. For tabular: BNN prior with architecture hyper-prior (embedding size, depth, activations sampled jointly).

## Experiments

- GP approximation with fixed hyperparameters: PFN's confidence intervals and means are visually indistinguishable from the exact GP PPD.
- GP with hyper-priors (no closed form): PFN matches or exceeds MLE-II (MAP) at 200× speed; outperforms NUTS at 1000–8000× speed.
- BNN posterior approximation: 1000× faster than Bayes-by-Backprop SVI; 10000× faster than NUTS at the same Prior-Data NLL.
- Small tabular classification (20 OpenML datasets, N=30 training / 70 test): PFN-BNN outperforms XGBoost, CatBoost, GPs, and KNN — without any per-dataset hyperparameter search.
- Few-shot Omniglot (5-way 5-shot): PFN with handwriting stroke prior + fine-tuning reaches state-of-the-art accuracy.

## Entities & Concepts

- [tabular-learning](tabular-learning.md)
