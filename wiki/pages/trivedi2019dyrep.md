---
title: "DyRep: Learning Representations over Dynamic Graphs"
tags: [source, temporal-graph, dynamic-graph, ctdg, temporal-point-process]
sources: [trivedi2019dyrep]
updated: 2026-04-29
---

# DyRep: Learning Representations over Dynamic Graphs

**Source:** https://arxiv.org/abs/1803.04051
**Title:** DyRep: Learning Representations over Dynamic Graphs
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Trivedi, Farajtabar, Biswal, Zha
**Venue:** ICLR 2019
**Year:** 2019

## Summary

Trivedi, Farajtabar, Biswal, and Zha (Georgia Tech / Google Brain) introduce DyRep, an inductive representation learning framework for continuous-time dynamic graphs that models two distinct interaction processes — structural evolution and communication — as a temporal point process.

**Two-process framework.** DyRep decomposes dynamic graph interactions into two types, indexed by $k$:
- **Association** ($k=0$): slow structural changes forming new graph edges (long-range dependencies)
- **Communication** ($k=1$): fast transient interactions along existing edges (short-range, high-frequency)

Both processes share the same embedding infrastructure but use separate event intensity functions.

**Embedding update.** When node $v$ is involved in event $e_p$ at time $t_p$ (with counterpart $u$):
$$z^v(t_p) = \sigma\!\left(W_{\text{struct}} \cdot h^u_{\text{struct}}(t_p^-) + W_{\text{rec}} \cdot z^v(t_{p-1}) + W_t \cdot (t_p - t_{p-1})\right)$$
- $h^u_{\text{struct}}$: local structural neighborhood aggregation over $u$'s current neighbors (attention-weighted)
- $W_{\text{rec}} \cdot z^v(t_{p-1})$: recurrent update from previous embedding (memory-like)
- $W_t \cdot \Delta t$: explicit temporal decay

**Intensity-based attention.** The structural neighborhood aggregation uses a stochastic matrix $S$ whose $(i,j)$ entry is the interaction intensity between nodes $i$ and $j$, softmax-normalized per row. This makes attention weights history-driven: nodes with more recent/frequent interactions receive higher weights.

**Temporal point process.** Event intensity at time $t$ for event type $k$ between nodes $u$ and $v$:
$$\lambda_{uv}^k(t) = f\!\left(g^k\!\left(z^u(t^-), z^v(t^-)\right)\right)$$
where $g^k$ is a learned bilinear/Hadamard compatibility function and $f$ is a softplus. The log-likelihood is:
$$\ell = \sum_p \log \lambda_{u_p v_p}^{k_p}(t_p) - \int_0^T \sum_{u,v,k} \lambda_{uv}^k(t)\,dt$$

The integral is approximated via Monte Carlo sampling.

**Inductive setting.** Embeddings are derived from the structural neighborhood at inference, not stored node lookup vectors — so the model generalizes to unseen nodes.

**Results.** On Social Evolution and GitHub event datasets, DyRep outperforms DeepCoevolve and other temporal baselines on future event prediction (time and event type).

## Key Takeaways

- **First to model two distinct interaction speeds**: association (topology-changing) vs. communication (signal-propagating) as separate intensity processes sharing a unified embedding.
- **Recurrent + structural + temporal**: the three-term update combines GRU-style recurrence, attention over current neighbors, and explicit elapsed-time encoding — each term addressing a different aspect of temporal dynamics.
- **Intensity-based attention is data-driven**: the stochastic matrix $S$ is updated from event history, so neighbor weights reflect actual interaction frequency rather than learned scalars.
- **DyRep = TGN special case**: TGN (2020) shows DyRep is equivalent to TGN with GRU memory + graph attention in the message function, with the association/communication distinction collapsed into edge features.
- **Temporal point process framing**: jointly models *when* and *what type* of event occurs, not just link probability at fixed time steps.

## Entities & Concepts

- [temporal-graph](temporal-graph.md)

## Relation to Other Wiki Pages

- [temporal-graph](temporal-graph.md): DyRep is an early CTDG model; predates TGAT/TGN but introduces the memory+attention combination later formalized by TGN.
- [rossi2020tgn](rossi2020tgn.md): TGN explicitly subsumes DyRep as a special case (GRU memory + graph attention in message function); DyRep's two-process distinction maps to edge feature conditioning.
- [xu2020tgat](xu2020tgat.md): TGAT (2020) replaces DyRep's recurrent update with pure Bochner time encoding + attention, removing the RNN dependency.
- [yu2023dygformer](yu2023dygformer.md): DyGFormer (2023) eliminates both RNN and multi-hop aggregation; DyRep's recurrent update is replaced by long first-hop Transformer sequences.
