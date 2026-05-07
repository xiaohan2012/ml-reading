---
title: "DyRep: Learning Representations over Dynamic Graphs"
tags: [source, temporal-graph, dynamic-graph, ctdg, temporal-point-process]
sources: [trivedi2019dyrep]
updated: 2026-05-06
---

# DyRep: Learning Representations over Dynamic Graphs

**Source:** https://arxiv.org/abs/1803.04051
**Title:** DyRep: Learning Representations over Dynamic Graphs
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Trivedi, Farajtabar, Biswal, Zha
**Venue:** ICLR 2019

## Summary

- **What:** Dynamic graphs contain two distinct interaction processes — topology changes (slow, structural) and communication (fast, transient) — that prior methods conflate into a single representation update.
- **How:** Models both processes as temporal point processes with separate intensity functions sharing a unified recurrent embedding that combines structural neighborhood aggregation, GRU-style recurrence, and elapsed-time encoding.
- **So what:** Outperforms DeepCoevolve on Social Evolution and GitHub datasets; later shown by TGN to be a special case of the general CTDG framework.

## Challenges & Novelty

Prior dynamic graph methods (DeepCoevolve, CTDNE) either ignored temporal ordering or could not model the qualitative difference between interactions that change graph topology (forming a new friendship) and interactions that signal along existing topology (sending a message). Conflating them forces the embedding to serve two incompatible objectives simultaneously.

- **Two-process decomposition:** association events ($k=0$) form new edges and capture long-range structural evolution; communication events ($k=1$) propagate signals along existing edges at high frequency. Separate intensity functions let the model specialize each process.
- **Intensity-based attention is data-driven:** the stochastic interaction matrix $S$ is updated from event history, so neighbor weights reflect actual interaction frequency — not learned static scalars — making the attention adaptive to graph dynamics.
- **Temporal point process framing:** jointly models *when* and *what type* of event occurs, not just link probability at a fixed time step. This enables event time prediction in addition to link prediction.

## Relation to Prior Work

| Model | Memory | Temporal point process | Two interaction types | Inductive |
|---|---|---|---|---|
| CTDNE | No | No | No | No |
| DeepCoevolve | RNN | Yes | No | No |
| **DyRep** | GRU (recurrent update) | Yes | Yes | Yes |
| [xu2020tgat](xu2020tgat.md) | No | No | No | Yes |
| [rossi2020tgn](rossi2020tgn.md) | GRU memory | No | Via edge features | Yes |

- [rossi2020tgn](rossi2020tgn.md): TGN shows DyRep is a special case — GRU memory + graph attention in the message function, with association/communication distinction collapsed into edge features.
- [xu2020tgat](xu2020tgat.md): TGAT (contemporaneous) replaces DyRep's recurrent update with pure Bochner attention — no RNN dependency.
- [yu2023dygformer](yu2023dygformer.md): DyGFormer eliminates both RNN and multi-hop aggregation; DyRep's recurrent update is the design choice most aggressively deprecated.

## Technical Details

**Two-process framework.** Interactions are typed: $k=0$ (Association, topology-changing) or $k=1$ (Communication, signal-propagating).

**Embedding update.** When node $v$ is involved in event $e_p = (v, u, k, t_p)$:

$$z^v(t_p) = \sigma\!\left(W_\text{struct} \cdot h^u_\text{struct}(t_p^-) + W_\text{rec} \cdot z^v(t_{p-1}) + W_t \cdot (t_p - t_{p-1})\right)$$

Three terms:
- $W_\text{struct} \cdot h^u_\text{struct}$: attention-weighted aggregation of $u$'s structural neighborhood
- $W_\text{rec} \cdot z^v(t_{p-1})$: GRU-style recurrent carry of $v$'s previous embedding
- $W_t \cdot \Delta t$: explicit temporal decay

**Intensity-based attention.** The structural aggregation uses a stochastic matrix $S$ where $S_{ij}$ is the softmax-normalized interaction intensity between nodes $i$ and $j$. Updated after each event — neighbors with more frequent recent interactions receive higher attention weights.

**Temporal point process.** Event intensity at time $t$ for type $k$ between nodes $u$ and $v$:

$$\lambda_{uv}^k(t) = f\!\left(g^k\!\left(z^u(t^-), z^v(t^-)\right)\right)$$

where $g^k$ is a learned bilinear/Hadamard compatibility function and $f$ is a softplus. Log-likelihood:

$$\ell = \sum_p \log \lambda_{u_p v_p}^{k_p}(t_p) - \int_0^T \sum_{u,v,k} \lambda_{uv}^k(t)\,dt$$

The integral is approximated via Monte Carlo sampling.

## Experiments

- Outperforms DeepCoevolve on future event prediction (both event time and event type) on Social Evolution and GitHub event datasets.
- Two-process decomposition ablation: using a single process (no $k$ distinction) degrades performance, validating the two-speed hypothesis.

## Entities & Concepts

- [temporal-graph](temporal-graph.md)
