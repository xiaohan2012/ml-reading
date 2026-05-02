---
title: "TGAT: Inductive Representation Learning on Temporal Graphs"
tags: [source, temporal-graph, attention, time-encoding, ctdg, inductive]
sources: [tgat]
updated: 2026-04-29
---

# TGAT: Inductive Representation Learning on Temporal Graphs

**Source:** https://arxiv.org/abs/2002.07962
**Title:** Inductive Representation Learning on Temporal Graphs
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Xu, Ruan, Korpeoglu, Kumar, Achan
**Venue:** ICLR 2020
**Year:** 2020

## Summary

Xu, Ruan, Korpeoglu, Kumar, and Achan (Walmart Labs) introduce TGAT — the first transformer-based architecture for inductive representation learning on continuous-time dynamic graphs. The two core innovations are: (1) a theoretically grounded **functional time encoding** that replaces positional encodings, and (2) a **temporal graph attention layer** that aggregates causally valid (temporally earlier) neighbors weighted by both feature relevance and time proximity.

**Functional time encoding.** Standard Transformer positional encodings are discrete and position-based; time in graphs is continuous. TGAT derives time encoding from Bochner's theorem: any continuous, translation-invariant kernel $\mathcal{K}(t_1, t_2) = \psi(t_1 - t_2)$ is the Fourier transform of a non-negative measure $p(\omega)$. Approximating via Monte Carlo:

$$\Phi_d(t) = \frac{1}{\sqrt{d}}\big[\cos(\omega_1 t),\, \sin(\omega_1 t),\, \ldots,\, \cos(\omega_d t),\, \sin(\omega_d t)\big]$$

where $\omega_1, \ldots, \omega_d$ are learnable parameters (treated as free parameters rather than a parameterized distribution — more efficient). The inner product $\langle \Phi_d(t_1), \Phi_d(t_2) \rangle \approx \mathcal{K}(t_1 - t_2)$ is translation-invariant: what matters is the time gap, not absolute timestamps.

**TGAT layer.** For target node $v_0$ at time $t$, with temporal neighborhood $\mathcal{N}(v_0; t) = \{v_1, \ldots, v_N\}$ (only interactions *before* $t$):

$$\mathbf{Z}(t) = \left[\mathbf{h}_0^{(l-1)}(t) \| \Phi(0),\; \mathbf{h}_1^{(l-1)}(t_1) \| \Phi(t-t_1),\; \ldots,\; \mathbf{h}_N^{(l-1)}(t_N) \| \Phi(t-t_N)\right]$$

Query = row 0 of $\mathbf{Z} \cdot \mathbf{W}_Q$; Keys/Values = rows 1:N. Standard scaled dot-product attention gives neighborhood summary $\mathbf{h}(t)$, which is concatenated with raw node features and passed through FFN:

$$\tilde{\mathbf{h}}_0^{(l)}(t) = \text{FFN}\!\left(\mathbf{h}(t) \| \mathbf{x}_0\right)$$

Stacking $L$ layers aggregates $L$-hop temporal neighborhoods with causal constraints. Multi-head ($k$ heads) improves stability. Edge features are incorporated by appending them to each neighbor's row in $\mathbf{Z}$.

**Inductive.** Because TGAT computes embeddings via a forward pass over sampled temporal neighbors (no node-specific parameters), any new node can be embedded as soon as it has interactions — same mechanism as GraphSAGE but time-aware.

**Relationship to TGN.** TGN (Rossi et al. 2020) later shows TGAT is a special case of TGN with no memory module. Without memory, TGAT requires 2 graph attention layers to match TGN-attn's 1 layer — memory acts as a compressed long-term summary that reduces the need for deep neighborhood aggregation.

**Results.** SOTA at publication on transductive and inductive future link prediction on Reddit, Wikipedia (70/15/15 chronological split). Outperforms GAT, GraphSAGE, and CTDNE on both tasks. Subsequently surpassed by TGN-attn (by ~3% absolute AP on Wikipedia).

## Key Takeaways

- **Bochner time encoding is theoretically motivated**: treats time as a learnable kernel rather than hand-crafted sinusoids — the ω parameters are optimized end-to-end; translation invariance ensures relative time gaps matter, not absolute timestamps.
- **Temporal causality constraint**: attention only over neighbors with $t_j < t$ — enforced at the subgraph batching level using BFS with chronological ordering.
- **No memory = staleness vulnerability**: TGAT's embedding at time $t$ only reflects the sampled temporal neighborhood, not the full history — the weakness TGN's memory module directly addresses.
- **2-layer TGAT vs 1-layer**: TGAT needs 2 layers to perform well; TGN with memory needs only 1 (and is 30× faster per epoch).
- **Foundation for TGN's attn embedding**: the temporal graph attention computation in TGN-attn is a direct extension of TGAT's layer, augmented with node memories as initial features ($\mathbf{h}_j^{(0)}(t) = \mathbf{s}_j(t) + \mathbf{v}_j(t)$).

## Entities & Concepts

- [temporal-graph](temporal-graph.md)

## Relation to Other Wiki Pages

- [temporal-graph](temporal-graph.md): TGAT introduces the first attention-based CTDG embedding; defines the temporal neighborhood aggregation paradigm.
- [tgn](tgn.md): TGN generalizes TGAT by adding the memory module; TGAT = TGN-no-mem; TGN-attn is strictly better and faster.
- [gat](gat.md): TGAT is GAT extended to temporal graphs — replaces standard positional encoding with Bochner time encoding; adds causal masking over temporal neighborhoods.
- [positional-encoding](positional-encoding.md): Bochner time encoding is an alternative to sinusoidal/learnable PE for the temporal domain — kernel-theoretic grounding instead of heuristic design.
