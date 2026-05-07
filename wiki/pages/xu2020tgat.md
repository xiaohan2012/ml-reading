---
title: "TGAT: Inductive Representation Learning on Temporal Graphs"
tags: [source, temporal-graph, attention, time-encoding, ctdg, inductive]
sources: [xu2020tgat]
updated: 2026-05-06
---

# TGAT: Inductive Representation Learning on Temporal Graphs

**Source:** https://arxiv.org/abs/2002.07962
**Title:** Inductive Representation Learning on Temporal Graphs
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Xu, Ruan, Korpeoglu, Kumar, Achan
**Venue:** ICLR 2020

## Summary

- **What:** Static graph attention (GAT) cannot handle continuous-time dynamics; standard discrete positional encodings are incompatible with continuous timestamps.
- **How:** Grounds time encoding in Bochner's theorem (any shift-invariant kernel is the Fourier transform of a non-negative measure), then applies multi-head attention over causally-valid temporal neighborhoods.
- **So what:** First Transformer-based CTDG architecture; SOTA at ICLR 2020 on transductive/inductive link prediction; directly precedes TGN, which subsumes TGAT as a special case.

## Challenges & Novelty

Prior inductive dynamic graph methods (CTDNE) relied on random walks on snapshots and couldn't generalize to unseen nodes or process continuous timestamps. Static GNNs (GAT, GraphSAGE) require retraining when new nodes appear and have no notion of temporal ordering.

- **Bochner time encoding is theoretically motivated:** any continuous, shift-invariant kernel $\mathcal{K}(\Delta t) = \psi(t_1 - t_2)$ is the Fourier transform of a non-negative measure. Approximating this via Monte Carlo gives a $d$-dimensional time feature vector; the parameters $\omega_1, \ldots, \omega_d$ are learned end-to-end, making the encoding task-adaptive.
- **Temporal causality constraint:** attention is restricted to neighbors with timestamps strictly before the query time $t$ — enforced at the subgraph batching level using BFS with chronological ordering. This is the CTDG equivalent of causal masking in language models.
- **Inductive by construction:** because TGAT computes embeddings via forward aggregation over sampled neighbors (no per-node parameters), any new node can be embedded as soon as it has interactions — same mechanism as GraphSAGE but time-aware.

## Relation to Prior Work

| Model | Inductive | Continuous time | Temporal attention | Memory |
|---|---|---|---|---|
| GraphSAGE ([hamilton2017graphsage](hamilton2017graphsage.md)) | Yes | No | No | No |
| GAT ([velickovic2018gat](velickovic2018gat.md)) | Yes | No | Yes | No |
| CTDNE | No | Yes | No | No |
| **TGAT** | Yes | Yes | Yes | No |
| [rossi2020tgn](rossi2020tgn.md) (TGN-no-mem) | Yes | Yes | Yes | No (≡ TGAT) |
| [rossi2020tgn](rossi2020tgn.md) (TGN-attn) | Yes | Yes | Yes | Yes |

- [rossi2020tgn](rossi2020tgn.md): TGN explicitly shows TGAT = TGN with no memory and attn embedding; adding memory allows 1 layer to match 2-layer TGAT performance at 30× speed.
- [velickovic2018gat](velickovic2018gat.md): TGAT extends GAT to temporal graphs — replaces position-agnostic aggregation with time-aware attention over causal neighbors.
- [yu2023dygformer](yu2023dygformer.md): DyGFormer reuses TGAT's Bochner time encoding but replaces multi-hop aggregation with long first-hop Transformer sequences.

## Technical Details

**Bochner time encoding.** For elapsed time $\Delta t = t - t'$:

$$\Phi_d(\Delta t) = \frac{1}{\sqrt{d}}\big[\cos(\omega_1 \Delta t),\, \sin(\omega_1 \Delta t),\, \ldots,\, \cos(\omega_{d/2} \Delta t),\, \sin(\omega_{d/2} \Delta t)\big]$$

Parameters $\omega_1, \ldots, \omega_{d/2}$ are learned. The inner product $\langle \Phi_d(t_1), \Phi_d(t_2) \rangle \approx \mathcal{K}(t_1 - t_2)$ is translation-invariant: only the time gap $\Delta t$ matters, not absolute timestamps.

**TGAT layer.** For target node $v_0$ at time $t$, with temporal neighborhood $\mathcal{N}(v_0; t) = \{(v_1, t_1), \ldots, (v_N, t_N)\}$ (all with $t_j < t$):

$$\mathbf{Z}(t) = \big[\mathbf{h}_0^{(l-1)}(t) \| \Phi(0),\;\; \mathbf{h}_1^{(l-1)}(t_1) \| \Phi(t - t_1),\;\; \ldots,\;\; \mathbf{h}_N^{(l-1)}(t_N) \| \Phi(t - t_N)\big]$$

Query = row 0; Keys/Values = rows 1:N. Standard scaled dot-product attention produces neighborhood summary $\mathbf{h}(t)$, which is concatenated with raw features and passed through FFN:

$$\tilde{\mathbf{h}}_0^{(l)}(t) = \text{FFN}\!\left(\mathbf{h}(t) \| \mathbf{x}_0\right)$$

Edge features are appended to each neighbor's row in $\mathbf{Z}$. Stacking $L$ layers aggregates $L$-hop temporal neighborhoods.

**No memory = staleness vulnerability.** Between events, $v_0$'s embedding is not updated — to compute $\mathbf{h}_0(t)$ at any query time, TGAT must re-run the full neighborhood aggregation. This is the weakness TGN's memory module directly addresses.

## Experiments

- SOTA at ICLR 2020: Wikipedia transductive AP 95.34%, inductive AP 93.99% (vs. GAT transductive 93.21%).
- Outperforms GraphSAGE and CTDNE on both transductive and inductive future link prediction.
- Subsequently surpassed by TGN-attn (+~3% AP on Wikipedia) which uses 1 layer + memory vs. TGAT's 2 layers.

## Entities & Concepts

- [temporal-graph](temporal-graph.md)
- [positional-encoding](positional-encoding.md)
