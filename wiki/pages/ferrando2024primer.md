---
title: A Primer on the Inner Workings of Transformer-based Language Models
tags: [survey, interpretability, mechanistic-interpretability, llm]
sources: [ferrando2024primer]
updated: 2026-07-14
---

# A Primer on the Inner Workings of Transformer-based Language Models

**Source:** https://arxiv.org/abs/2405.00208
**Title:** A Primer on the Inner Workings of Transformer-based Language Models
**Date ingested:** 2026-05-28
**Type:** survey / primer
**Authors:** Javier Ferrando, Gabriele Sarti, Arianna Bisazza, Marta R. Costa-jussà
**Venue:** arxiv 2024 (concurrent with ICLR 2024 blog ecosystem)

## Summary

- **Two-axis taxonomy** of LM mech-interp: **localize** (which inputs/components drove the prediction) and **decode** (what features live in a representation).
- **Unified residual-stream notation** ties every method (DLA, patching, circuits, lens, SAEs) back to a shared algebraic view.
- **Synthesis catalogue** of discovered inner behaviors — attention head types, FFN neuron types, residual-stream phenomena, multi-component circuits (IOI, factual recall).

## Background: residual-stream view

Every method below is grounded in the residual-stream perspective. Full treatment with derivations, OV/QK factorization, FFN-as-memory, and the shallow-ensemble unroll lives at [transformer-background](transformer-background.md). The minimum required here:

**Notation.**

- $l$ — layer; $h$ — head; $i, n$ — positions ($n$ = predicted).
- $\mathbf{x}^l_i \in \mathbb{R}^d$ — residual stream at layer $l$, position $i$.
- $\mathbf{W}_U \in \mathbb{R}^{d \times |V|}$ — unembedding (residual → logits).
- LN — LayerNorm, applied per-token before each sublayer (Pre-LN).

**Key facts.**

- **Residual stream is additive.** Every sublayer writes back into the same $d$-dim stream:
  $$\mathbf{x}^l_i = \mathbf{x}^{l-1}_i + \text{Attn}^l(\text{LN}(\mathbf{x}^{l-1}_i)) + \text{FFN}^l(\text{LN}(\cdot))$$

- **Attention factorizes into two circuits:**
    - **OV circuit** $\mathbf{W}_V\mathbf{W}_O$ — *what* is written into the stream.
    - **QK circuit** $\mathbf{W}_Q\mathbf{W}_K^\top$ — *where* to read from (drives attention weights).

- **Bases:**
    - FFN neurons = **privileged basis** — readable directly (nonlinearity pins the basis).
    - Residual stream = **no privileged basis** — rotations absorbable into adjacent weights → features must be *discovered* (motivates SAEs).

- **Prediction-as-sum** — the equation every downstream method depends on (Ferrando Eq. 10):
  $$f(\mathbf{x}) = \sum_{l,h} \text{Attn}^{l,h}\mathbf{W}_U + \sum_l \text{FFN}^l\mathbf{W}_U + \mathbf{x}_n\mathbf{W}_U$$
    - **Why it factors.** $\mathbf{W}_U$ is linear, the final residual is a sum of writes — so $\mathbf{W}_U$ distributes.
    - **What each summand is.** One component's (head / FFN / embedding) **direct contribution to the logits**.
    - **Downstream uses.** DLA, DLDA, circuit analysis; applied at *intermediate* layers → the logit-lens family.


## The two-axis taxonomy

| Axis | Question | Representative methods |
|---|---|---|
| **Behavior Localization** | Which *inputs* or *components* are responsible for a prediction? | input attribution, DLA, activation patching, circuit discovery |
| **Information Decoding** | What *features* are encoded in a representation, and where? | probing, SAEs, logit/tuned lens, Patchscopes |

The other two sections of the survey are syntheses — *Discovered Inner Behaviors* (results gathered using these tools) and *Tools* (software libraries).

## Behavior Localization

**Main idea.**

- Given a single prediction, identify *which inputs* (tokens, training examples) and *which model components* (heads, FFNs, neurons, edges) are causally responsible — answering *"why this output?"* rather than *"what does the model know?"*.
- Methods split by **target**: input tokens (attribution) vs. internal components (logit attribution, patching, circuits).

### Input attribution

Estimate contribution of input *tokens* to model output. Faithfulness debates abound; methods often disagree.

- **Gradient-based.** $\|\nabla_{\mathbf{x}_i} f_w(\mathbf{x})\|_p$; sensitivity to embedding $i$. Variants address gradient saturation/shattering:
    - **Integrated Gradients** — integrate gradients along a path from a baseline.
    - **SmoothGrad** — average over input perturbations.
    - **LRP / AttnLRP** — custom propagation rules; sum-preserving.
    - **Gradient × Input** — convert sensitivity to importance.
- **Perturbation-based.** Ablate/noise an input element, measure prediction change. Family includes **LIME** (local surrogate), **SHAP** (Shapley values). Suffers from OOD inputs and cost.
- **Context mixing.** Use attention weights, value-weighted norms, or vector distances to estimate inter-token contribution; aggregate per-layer via **attention rollout** or **DecompX** linearization.
- **Contrastive attribution.** Explain $w$ *instead of* $o$: $\|\nabla_{\mathbf{x}_i}(f_w - f_o)\|_p$. Addresses competing-tokens-in-vocab problem.
- **Training data attribution (TDA).** Influence functions, TRAK, simulator-based methods — locate *training examples* driving a prediction.

**Limitations.** Insensitivity to model/data; methods disagree; provably unreliable for counterfactuals (IG, SHAP); OOD issues for perturbation.

![Three approaches to inter-token contribution: attention weights, value-weighted norms, distance-based (attn_patterns).](assets/ferrando2024primer-attn-patterns.png)

### Component importance

**Progression.** Three subsections form a ladder of increasing structural ambition, then a fourth re-grounds the whole stack:

- **Logit attribution** — *direct* effect of one component on the output logit, computed from a **single forward pass** via residual-stream linearity. No counterfactual needed.
- **Causal interventions / activation patching** — *total* effect of one component, computed by **replacing** its activation with one from a counterfactual run.
- **Circuit analysis** — scale-up to **subgraphs**: which *edges* between components carry the signal.
- **Causal abstraction** — find the *interpretable variables* a circuit implements, not just its components.

#### Logit Attribution

**Motivation**
- Converts the abstract "logit of $w$" into **(component, *source*?, sign, magnitude)** tuples (*source* only via per-path DLA).
- The atomic unit every downstream mech-interp method (circuits, lens, ablation) builds on.

**Reminder — Prediction-as-sum.**
$$f(\mathbf{x}) = \sum_{l,h} \text{Attn}^{l,h}\mathbf{W}_U + \sum_l \text{FFN}^l\mathbf{W}_U + \mathbf{x}_n\mathbf{W}_U$$

**What is a "component"?** Any addend in the equation above — its **write into the residual stream** $f^c(\mathbf{x}) \in \mathbb{R}^d$:

- an attention head: $f^c(\mathbf{x}) = \text{Attn}^{l,h}(\cdots)$,
- an FFN layer: $f^c(\mathbf{x}) = \text{FFN}^l(\cdots)$,
- a single FFN neuron: $f^c(\mathbf{x}) = n_u\,\mathbf{w}_{\text{out}_u}$,
- the input embedding: $f^c(\mathbf{x}) = \mathbf{x}_n$,
- a *path* through several heads (per-path DLA).

**Methods.** All three read off a dot product of a write into the stream against an unembedding column $\mathbf{W}_{U[:,w]} \in \mathbb{R}^d$ (token $w$'s readout direction). Single forward pass, no counterfactual.

**Direct Logit Attribution (DLA).** Component $c$'s push on token $w$:
$$f^c(\mathbf{x})\,\mathbf{W}_{U[:,w]}$$
Sign = promote ($>0$) / suppress ($<0$); magnitude = strength. By Prediction-as-sum, $\text{logit}_w = \sum_c$ of these.

**Direct Logit Difference Attribution (DLDA).** Contrastive: how much $c$ favors $w$ over $o$:
$$f^c(\mathbf{x})(\mathbf{W}_{U[:,w]} - \mathbf{W}_{U[:,o]})$$
Cancels shared baseline directions (e.g. a uniform frequent-token bias).

**Per-path Direct Logit Attribution** (Ferrando et al. 2023). Attribute through source position $j$ inside head $(l,h)$:
$$a^{l,h}_{n,j}\,\mathbf{x}^{l-1}_j\,\mathbf{W}_{OV}^{l,h}\,\mathbf{W}_{U[:,w]}$$
Identifies *which source token*, *through which head*, pushed how hard toward $w$.

#### Causal interventions / activation patching

**What it does.** Replace one component's activation $f^c(\mathbf{x})$ with an alternative $\tilde{\mathbf{h}}$ from another forward pass — formally $f(\mathbf{x} \mid \text{do}(f^c(\mathbf{x}) = \tilde{\mathbf{h}}))$ — and measure the change in prediction. Recovers the **total effect** (direct + indirect via downstream components), in contrast to DLA's purely direct effect. The patch source and direction jointly determine what causal quantity is estimated.

- **Patch sources:**
    - **Resample** — single example from a counterfactual distribution $P_{\text{patch}}$.
    - **Mean** — average activation over $P_{\text{patch}}$.
    - **Zero** — null vector (risk: OOD).
    - **Noise** — Gaussian-perturbed input.
- **Direction:**
    - **Noising** — patch corrupted activation into a clean run; tests **necessity** ("can the component be broken without breaking the prediction?").
    - **Denoising** — patch clean activation into a corrupted run (Causal Tracing); tests **sufficiency** ("can restoring the component alone rescue the prediction?").
- **Component modeling** — learn a linear estimator predicting effects of subset interventions, avoiding combinatorial patching.

#### Circuits Analysis

**What is a circuit?** A subgraph of the forward-pass DAG (nodes = components, edges = residual-stream reads) that *jointly implements* a task — canonical example: the IOI circuit in GPT-2 Small.

**Origin of the concept.**

- **Olah et al., "Zoom In" (Distill 2020)** — coined "circuits" for vision; defined a circuit as an *interpretable subgraph* of a network, with each node carrying a nameable role.
- **Elhage et al., "Mathematical Framework" (Anthropic 2021)** — gave Transformer circuits their *algebraic* meaning: edges = OV/QK composition between heads.
- **Wang et al., IOI (ICLR 2023)** — first fully-specified circuit in a real LM and the source of the three formal criteria below.

**Working definition** (compressed): a circuit is a small subgraph that is

- **sufficient** — running only these nodes reproduces the behavior,
- **necessary** — ablating any of them breaks it,
- **interpretable** — each node has a nameable role (Duplicate Token Head, S-Inhibition Head, …).

**Formal criteria** (Wang et al. 2022, IOI):

- **Faithfulness** — the circuit alone, run on the task, reproduces the target metric (e.g. the Mary − John logit difference). ≈ sufficient.
- **Completeness** — no relevant components are left out; nothing outside the circuit changes the metric.
- **Minimality** — every node is necessary; removing any one breaks the behavior. ≈ necessary.

Interpretability sits on top of the formal triad — it's what makes a circuit an *explanation* rather than just a high-scoring subgraph.

**Discovery algorithms** — scale patching from nodes to edges.

- **Edge patching** — patch a single residual-stream edge between two components, leveraging the additive decomposition of each component's input.
- **Path patching** — generalize to *multi-edge* paths; isolates **direct vs indirect** effects of a sender on a receiver (Pearl mediation).
- **ACDC** (Automatic Circuit DisCovery; Conmy et al. 2023) — iterative edge removal, fully automated but $O(\text{edges})$ forward passes — impractical for large models.
- **EAP** (Edge Attribution Patching; Syed et al. 2023) — linear approximation of patching with **one forward + one backward pass**; orders of magnitude cheaper than ACDC.
- **EAP-IG** (EAP with Integrated Gradients; Hanna et al. 2024) — EAP combined with Integrated Gradients; demonstrably **more faithful** circuits than vanilla EAP.
- **AtP\*** (Attribution Patching\*; Kramár et al. 2024) — patches two known false-negative modes of attribution patching while keeping the efficiency.
- **Information Flow Routes** (Ferrando & Voita 2024) — patch-free, **single forward pass**; extracts a subnetwork via context-mixing aggregation, with no counterfactual dataset and no self-repair risk.

**Worked example — ACDC** (the reference implementation; everything else is defined relative to it).

- **Inputs.**
    - *Clean* dataset — prompts exhibiting the behavior (IOI sentences).
    - *Corrupted* dataset — counterfactual prompts that destroy it (swap names).
    - *Metric* — typically the logit difference on the clean run.
    - *Threshold* $\tau$ — tolerated metric drop per pruned edge.
- **Algorithm.**
    1. Cache all activations on clean + corrupted runs.
    2. Sort edges in **reverse topological order** (output → input).
    3. For each edge $A \to B$: patch $A$'s contribution into $B$'s input using the *corrupted* activation; re-run; measure metric drop $\Delta$.
    4. If $\Delta < \tau$ → drop the edge; else → keep it.
    5. Surviving subgraph = the circuit.
- **Why reverse order.** Pruning from the output backwards means each edge is judged against the *already-pruned* downstream — an edge survives only if it still matters in the current candidate circuit. Forward order would overestimate importance.
- **Single-edge patch (concretely).** Downstream input is additive over upstream writes (residual-stream identity), so:
    $$\text{input}_B = \sum_{A' \ne A} f^{A'}(\mathbf{x}_{\text{clean}}) + f^A(\mathbf{x}_{\text{corrupted}})$$
    Only $A$'s contribution to $B$ uses the corrupted prompt; everything else stays clean.
- **Threshold trade-off.** Small $\tau$ → conservative, large circuit, high faithfulness. Large $\tau$ → aggressive prune, small circuit, lower faithfulness. Sweep to get a ROC-style curve.
- **Cost.** $O(|E|)$ forward passes per task (~30k for GPT-2 Small). This is the bottleneck **EAP** removes — same target, 2 passes total.
- **Validation.** Recovers Wang et al.'s IOI circuit unsupervised — the methodology check that justified the whole automated-discovery program.
- **Failure modes.** Greedy (never re-evaluates a pruned edge); threshold-sensitive; depends on a well-designed corrupted dataset; **self-repair** can mask edges that *would* matter without backup pathways; doesn't interpret — still need DLA + attention-pattern inspection for step 5 of the recipe.

**Shared limitations of circuit discovery.** (1) requires designing $P_{\text{patch}}$, (2) needs human inspection for subgraph isolation and node-role labeling, (3) interventions can trigger **second-order self-repair** that confounds the analysis.

#### Causal Abstraction

A complement to circuits: rather than locating *which* components matter, find the **high-level interpretable variables** they implement. Bridges component-level mechanism to feature-level explanation.

- **Subspace patching / DII** (Geiger 2023) — intervene only on a learned linear subspace $U \subset \mathbb{R}^d$ of an activation, not the whole vector; respects the **linear representation hypothesis**.
- **DAS / Boundless DAS** — distributed alignment search: find non-basis-aligned subspaces with causal influence via gradient descent. Empirically the strongest causal-intervention method across syntactic / mathematical / attribute benchmarks.
- **Causal Proxy Models (CPMs)** — interpretable proxies trained to mimic the original model's counterfactual behavior.

## Information Decoding

### Probing

- A supervised classifier $p: f^l(\mathbf{x}) \mapsto z$ on frozen activations.
- **Tension:** probe accuracy conflates *amount of info in $f^l$* with *probe expressivity*. Mitigations:
    - **Control tasks** (random labels) and **control functions** baselines.
    - **MDL probes** — measure description length, not just accuracy.
- **Limits.** Correlation, not causation — high probe accuracy doesn't mean the model *uses* that feature.

→ Full page: [probing-classifier](probing-classifier.md).

### Linear Representation Hypothesis & SAEs

**LRH.** Features encoded as **linear subspaces** (directions) of the representation space.

- **Evidence.** Word2vec analogies; concept directions found via linear probes; interpretable monosemantic neurons.
- **Linear interventions.**
    - **Concept erasure** (Ravfogel, LEACE, OracleLEACE) — project out a direction; verifies probed info is used.
    - **Activation addition / steering** — add $-\alpha \mathbf{u}$ to flip sentiment, refusal, etc.
    - **Difference-in-means** vectors for refusal, truthfulness.

**Polysemanticity & superposition.** Models pack *more features than dimensions* by storing them in non-orthogonal directions — explains why most neurons fire on apparently unrelated inputs.

**Sparse Autoencoders (SAEs).** Disentangle superposed features via overcomplete dictionary learning. Trained to reconstruct $\mathbf{z}$ with sparsity penalty on activations $h(\mathbf{z})$.

**Variants:**

| SAE | Activation | Sparsity term | Strength |
|---|---|---|---|
| **Standard** (ReLU) | ReLU | $\alpha \|h\|_1$ | baseline; suffers *shrinkage* |
| **Gated (GSAE)** | ReLU × gated step | $\|h\|_1$ on gate path | decouples magnitude from detection; Pareto-improves baseline |
| **TopK** | TopK | none — only reconstruction | constrains exactly $k$ active features; outperforms ReLU |
| **BatchTopK** | TopK over batch | none | per-sample flexibility |
| **JumpReLU** | JumpReLU($\theta$) | $L_0$ on $\theta$-gated features | decouples activation decision from magnitude |

- **Evaluation:** Pareto frontier of **L0 norm** vs **loss recovered**; feature density histogram; manual or LLM-automated interpretability scoring.
- **Caveats:** SAE error term ($\epsilon$) shifts predictions more than random noise — faithfulness concern. Marks et al. 2024 incorporate $\epsilon$ as a node in causal graph for **sparse feature circuits**.

![Linear representations & superposition (left); SAE architecture (right) — features_graphs_sae.](assets/ferrando2024primer-sae.png)

### Decoding in Vocabulary Space

Use $\mathbf{W}_U$ (or $\mathbf{W}_E$) as an interpretable readout from any representation.

- **Logit lens** (nostalgebraist) — project intermediate $\mathbf{x}^l$ by $\mathbf{W}_U$. Fails on some models (Belrose).
- **Tuned lens** (Belrose et al. 2023) — learned affine *translator* before $\mathbf{W}_U$.
- **Attention lens** — translator on attention head outputs.
- **Patchscopes** — generalize patching: pick target model $f^*$, target prompt $\mathbf{x}^*$, target component $c^*$, mapping $m$. Decodes information context-independently. **Future lens** is a special case.
- **Decoding weights.** Project $\mathbf{W}_{OV}, \mathbf{W}_{\text{out}}$ rows through $\mathbf{W}_U$ to inspect what tokens they promote. **SVD decomposition** reveals dominant directions; **low-rank approximation** improves accuracy in later layers (LASER).
- **Logit spectroscopy** — split right singular vectors of $\mathbf{W}_U$ into bands; subspace-level lens.
- **Maximally-activating inputs** — examples that fire a neuron/feature most strongly; risk of "interpretability illusions" across activation ranges.
- **Natural-language self-explanation** — prompt an LLM to describe what a neuron fires for (GPT-4 → GPT-2 XL). Often unfaithful.
- **Backward lens** — project FFN *gradient* matrices to study how new information is stored.

→ Related page: [tabular-logit-lens](tabular-logit-lens.md) (TFM adaptation).

## Discovered Inner Behaviors

Findings catalogued by the survey — exploration is non-exhaustive but representative.

### Attention head types

- **Interpretable attention patterns:**
    - **Positional heads** — attend to self / previous / next token.
    - **Previous-token heads** — substrate for induction; copy previous token info.
    - **Subword joiner heads** — attend to prior subwords of the same word.
    - **Syntactic heads** — specialize in `obj`, `nsubj`, `advmod`, `amod` dependencies; emerge suddenly during training.
    - **Duplicate token heads** — attend to prior occurrences of current token.
- **Interpretable QK/OV circuits:**
    - **Copying heads** — diagnosed by positive eigenvalues of $\mathbf{W}_E \mathbf{W}_{OV} \mathbf{W}_U$.
    - **Induction heads** — prev-token + copying composition; substrate of ICL.
    - **Name mover / S-inhibition / negative mover heads** — IOI circuit components.
    - **Copy suppression heads** — downweight tokens already attended.
    - **Attention sink** — disproportionate attention to BOS / first tokens; acts as no-op valve.

### FFN neuron types

- **Input-side:** position-range neurons, skill neurons, concept-specific neurons (Python, French, German), grammatical/linguistic neurons.
- **Output-side:** knowledge neurons (factual associations), linguistic-acceptability neurons, space/time neurons, language-specific neurons, **token-frequency neurons** (push toward/from unigram), suppression-of-improbable-continuations neurons.
- **Polysemantic / structural:** n-gram detector neurons in early layers, dead neurons (OPT), **de-/re-tokenization neurons**, **universal neurons** (1-5% shared across seeds), **entropy neurons** (modulate output uncertainty via $\mathbf{W}_U$ null space).
- **High-level structure.** Early ≈ sensory (n-grams), middle ≈ abstract concepts, late ≈ motor (push token distribution).

### Residual stream phenomena

- **Exponential norm growth** along layers (in $\mathbf{x}$ and in output matrices $\mathbf{W}_O, \mathbf{W}_{\text{out}}$).
- **Memory management** — components with negative-eigenvalue OV or anti-aligned FFN keys/values *delete* prior writes.
- **Outlier dimensions** — rogue dims with anisotropic effect; ablating breaks performance; correlate with training frequency; complicate quantization.
- **SAE features in residual stream** — local context features, partition features (promote/suppress dual sets), suppression features, abstract / safety / code-error features (Anthropic Claude-3 SAEs).

### Emergent multi-component behaviors

- **IOI circuit** (Wang et al. 2023) — duplicate token + induction → S-inhibition → name mover; canonical mech-interp case study.
- **Function / task vectors** (Hendel, Todd) — middle-layer activations that, when patched into novel zero-shot prompts, induce the task.
- **Factual recall circuit** (Meng, Geva, Chughtai) — early-middle FFNs at subject-last-token enrich subject; later attention heads extract attribute via additive subject-head + relation-head mechanism.
- **In-context vs memory heads** — competing mechanisms for grounded vs memorized answers.
- **Retrieval heads** — long-context Needle-in-Haystack solvers.
- **Grokking as circuit emergence** — generalizing sparse circuit replaces dense memorizing subnetwork.
- **Circuit generality** — same low-level components participate in many tasks; circuits robust to fine-tuning.

## Tools

| Category | Tools |
|---|---|
| Input attribution | Captum, Transformers Interpret, ferret, Ecco, Inseq, SHAP, Saliency, LIT |
| Component importance / circuits | **TransformerLens**, **NNsight**, **Pyvene**, Pyreft |
| SAE training / inspection | SAELens, dictionary-learning, sae-vis, Neuronpedia |
| Visualization | BERTViz, exBERT, InterpreT, LM-Debugger, VISIT, Ecco, Tuned Lens, CircuitsVis, Penzai, LM-TT, TDB |
| RASP / Tracr | human-readable Transformer spec → compiled weights |
| Vision/multimodal | ViT Prisma, MAIA |

## Conclusion & Future Directions

- **From functional grounding → actionable insights.** Move beyond toy benchmarks to debugging real models.
- **From component space → feature/NL space.** LMs as verbalizers; feature-based circuits.
- **Open access to LM internals** — prerequisite for the field.

## Bridge: techniques used in [balef2026onelayer](balef2026onelayer.md) vs not

| Family | Technique | balef uses? | Notes |
|---|---|---|---|
| **Input attribution** | Gradient / IG / SHAP / LIME / LRP | ❌ | tabular ICL has no LLM-style input-token sequence |
| | Context mixing (attention rollout, ALTI) | ❌ | |
| | Training data attribution | ❌ | |
| **Component importance** | DLA / DLDA | ❌ | no per-component logit decomposition |
| | Activation patching (resample/mean) with $P_{\text{patch}}$ | ❌ | no counterfactual datasets |
| | Zero/identity intervention | ⚠️ partial | [layer-ablation](layer-ablation.md): skip/repeat/swap ≈ identity patching |
| | Subspace patching / DII | ❌ | |
| | Path / edge patching | ❌ | |
| | ACDC / EAP / EAP-IG / AtP* | ❌ | no circuit discovery |
| | DAS / causal abstraction | ❌ | |
| **Decoding — probing** | Linear probe per layer + cross-layer matrix | ✅ | [probing-classifier](probing-classifier.md) |
| | Control tasks / MDL | ❌ | |
| | Concept erasure (LEACE) | ❌ | |
| **Decoding — linear rep** | SAEs (any variant) | ❌ | |
| | Activation addition / steering | ❌ | |
| | Difference-in-means / refusal direction | ❌ | |
| **Decoding — vocab space** | Logit lens | ✅ (adapted) | [tabular-logit-lens](tabular-logit-lens.md) — closer to tuned lens (trained translator) |
| | Tuned lens | ✅ (closest analog) | |
| | Patchscopes / SVD weights / spectroscopy / max-activating inputs | ❌ | |
| **Discovered behaviors** | Head taxonomy (induction, copy-suppression, …) | ❌ | depth-level only, no per-head analysis |
| | Knowledge / entropy / language neurons | ❌ | |
| | Residual-stream norm growth, outlier dims | ❌ | |
| | Multi-component circuits (IOI, factual recall) | ❌ | TFM ICL queries lack a clean (subject, relation, attribute) factorization |
| | **Self-repair (Hydra)** | ✅ | [self-repair](self-repair.md) — lens-after-skip trajectory |
| **Representation analysis** | Representation similarity (CKA / cosine) | ✅ | [representation-similarity](representation-similarity.md) — not central in this survey but adjacent |

**Headline.** balef occupies a narrow **layer-level, behavioral** slice — probing + lens + ablation + self-repair + repr-similarity. Skipped LLM toolboxes:

- **SAEs / linear-rep tooling** — no feature-direction dictionary.
- **Counterfactual interventions** — TFM ICL lacks clean templates like IOI / `(s, r, a)`.
- **Circuit discovery** — depth-level, not head-level granularity.
- **Input attribution** — TFM "tokens" are columns + rows, not a 1-D text sequence.

*Why:* TFMs are small enough that depth-level behavior is the open frontier, and tabular ICL makes feature/circuit decomposition methodologically awkward.

**Update.** [gupta2026tabpfnheads](gupta2026tabpfnheads.md) fills three of the ❌ rows balef left open on TabPFN-2.5: **counterfactual activation patching** (with clean/corrupt runs), **per-head analysis** (head taxonomy — one dominant head + late computation heads), and **activation addition / steering** (which it finds *does not transfer*, attributed to the absence of LLM-style function-vector heads). It remains single-module and two-task, so circuit discovery and SAE tooling stay unexplored.

## Entities & Concepts

- [probing-classifier](probing-classifier.md)
- [tabular-logit-lens](tabular-logit-lens.md)
- [self-repair](self-repair.md)
- [layer-ablation](layer-ablation.md)
- [representation-similarity](representation-similarity.md)
- [balef2026onelayer](balef2026onelayer.md)
- [gupta2026tabpfnheads](gupta2026tabpfnheads.md) — ports activation patching + per-head analysis + steering to TabPFN-2.5
