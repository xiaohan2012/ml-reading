---
title: "LSPE: Graph Neural Networks with Learnable Structural and Positional Representations"
tags: [source, positional-encoding, graph-transformer, graph-neural-network]
sources: [dwivedi2022lspe]
updated: 2026-05-07
---

# LSPE: Graph Neural Networks with Learnable Structural and Positional Representations

**Source:** https://arxiv.org/abs/2110.07875
**Title:** Graph Neural Networks with Learnable Structural and Positional Representations
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, Xavier Bresson
**Venue:** ICLR 2022

## Summary

- **What:** Existing GNNs and GTs conflate structural features (task-relevant node attributes) and positional encodings (node identity within graph) into a single representation, making it harder to learn each independently and using a static PE that doesn't adapt to the task.
- **How:** Maintain two separate streams — $\mathbf{h}_v$ for structural/feature representations and $\mathbf{p}_v$ for positional representations — that evolve in parallel through all network layers with separate learnable update functions.
- **So what:** 1.79%–64.14% performance improvements across 7 molecular benchmarks when applied to GatedGCN, PNA, Transformer, and SAN-GT baselines; directly influenced GraphGPS's architectural separation of MPNN and global attention streams.

## Challenges & Novelty

Standard practice initializes LapPE at the input layer and concatenates it with node features — the network then merges them into a single vector and cannot refine positional information independently. This has two problems: (1) the PE becomes entangled with task features, making it harder to learn either well; (2) the PE is static — it doesn't evolve to reflect the task, meaning the same LapPE is used regardless of whether the task is local or global.

- **Decoupled streams allow independent learning:** the positional stream $\mathbf{p}_v$ is updated at every layer by its own aggregation rule and MLP, conditioned on its current neighbors' positions — the PE adapts to the learned representation rather than staying fixed as a preprocessing step.
- **Structural stream conditions on current positions:** the structural update $\mathbf{h}_v^{(l+1)}$ uses the current $\mathbf{p}_v^{(l)}$ — the positions gate the structural aggregation at each layer, creating a feedback loop between the two streams.
- **Generic architectural pattern:** LSPE is not a new base model — it augments existing models (GatedGCN, PNA, Transformer, SAN) by splitting their node representation into two streams, without changing the base architecture's attention or aggregation mechanism.

## Relation to Prior Work

| PE approach | Static/Adaptive | Decoupled stream | Base model | ICLR 2022 |
|---|---|---|---|---|
| LapPE ([dwivedi2020benchmarking](dwivedi2020benchmarking.md)) | Static | No | Any | — |
| SignNet ([lim2022signnet](lim2022signnet.md)) | Static (invariant) | No | Any | Yes |
| **LSPE** | Adaptive (per-layer) | Yes | Any | Yes |
| RWSE ([rampavsek2022graphgps](rampavsek2022graphgps.md)) | Static | No | GPS | 2022 |

- [dwivedi2020benchmarking](dwivedi2020benchmarking.md): LSPE builds directly on LapPE introduced there — same lead author, extends the PE concept from static input feature to dynamic learnable stream.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GraphGPS incorporates the decoupled PE philosophy; its separation of MPNN and global attention components is architecturally related to LSPE's two-stream design.
- [lim2022signnet](lim2022signnet.md): both appear at ICLR 2022 and both address LapPE limitations; SignNet fixes the sign symmetry problem, LSPE fixes the static/entanglement problem — complementary solutions.
- [kreuzer2021san](kreuzer2021san.md): LSPE applied to SAN's GT backbone shows the decoupled PE stream improves even spectral-based Transformers.

## Technical Details

**Dual-stream update.** At each layer $l$, two parallel updates:

**Positional stream:**
$$\mathbf{p}_v^{(l+1)} = f_p\!\left(\mathbf{p}_v^{(l)},\; \left\{\!\!\left\{\mathbf{p}_u^{(l)} : u \in \mathcal{N}(v)\right\}\!\!\right\}\right)$$

where $f_p$ is the same type of aggregation as the base model (e.g., GatedGCN-style gated aggregation, PNA-style multi-aggregation, or sparse attention).

**Structural stream (conditioned on current PE):**
$$\mathbf{h}_v^{(l+1)} = f_h\!\left(\mathbf{h}_v^{(l)},\; \mathbf{p}_v^{(l)},\; \left\{\!\!\left\{(\mathbf{h}_u^{(l)}, \mathbf{p}_u^{(l)}) : u \in \mathcal{N}(v)\right\}\!\!\right\}\right)$$

**Initialization:** $\mathbf{h}_v^{(0)} = \text{MLP}(\mathbf{x}_v)$ (node features), $\mathbf{p}_v^{(0)} = \text{MLP}(\text{LapPE}(v))$ (Laplacian eigenvectors).

**Readout:** concatenate $[\mathbf{h}_v^{(L)} \| \mathbf{p}_v^{(L)}]$ for node-level predictions; pool over nodes for graph-level.

**Positional loss (auxiliary).** An additional loss term encourages the learned PE to remain informative: predict the original LapPE from the learned PE at each layer. This prevents the PE stream from collapsing into the structural stream.

## Experiments

- ZINC (graph regression): LSPE improves GatedGCN by 1.79% and Transformer by 42.3% — structural tasks benefit most.
- AQSOL (molecular solubility): LSPE + PNA achieves 64.14% improvement over PNA-no-PE — the single largest gain reported.
- PATTERN/CLUSTER (node classification): LSPE consistently improves all four base models; SAN-GT with LSPE outperforms all prior GTs.
- Ablation: decoupled streams > concatenated (single-stream) PE by 2–15% across tasks; per-layer PE update > static PE by 1–8%.

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-transformer](graph-transformer.md)
- [graph-neural-network](graph-neural-network.md)
