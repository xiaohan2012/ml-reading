---
title: "MPNN: Neural Message Passing for Quantum Chemistry"
tags: [source, gnn, message-passing, framework, molecular]
sources: [gilmer2017mpnn]
updated: 2026-04-29
---

# MPNN: Neural Message Passing for Quantum Chemistry

**Source:** https://arxiv.org/abs/1704.01212
**Title:** Neural Message Passing for Quantum Chemistry
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Gilmer, Schütt, Riley, Vinyals, Sadowski, Ghahramani
**Venue:** ICML 2017
**Year:** 2017

## Summary

Gilmer et al. (ICML 2017, Google Brain) unify a large class of graph neural networks under a single **Message Passing Neural Network (MPNN)** framework and apply it to quantum chemistry. The framework abstracts the common structure of at least 8 prior models — including GCN, Gated Graph Neural Networks, Interaction Networks, Deep Tensor Neural Networks, and Molecular Graph Convolutions — into a clean two-phase formulation. This serves both as a retrospective unification and as a template for designing new variants.

**Message passing phase** (T steps):

```
m_v^{t+1} = sum_{w in N(v)} M_t(h_v^t, h_w^t, e_vw)
h_v^{t+1} = U_t(h_v^t, m_v^{t+1})
```

**Readout phase:**

```
y_hat = R({h_v^T | v in G})
```

where `M_t` (message function), `U_t` (update function), and `R` (readout function) are all learned differentiable functions. `R` must be permutation-invariant. Edge features `e_vw` are first-class inputs to `M_t`, enabling richer molecular representations. The paper also introduces the concept of updating edge hidden states alongside node states.

The paper's own MPNN variant achieves SOTA on QM9 (130k molecules, 13 quantum properties), predicting DFT values to within chemical accuracy on 11/13 targets, compared to previous best of 0/13.

## Key Takeaways

- Canonical MPNN formulation: `m_v^{t+1} = sum M_t(h_v^t, h_w^t, e_vw)`, `h_v^{t+1} = U_t(h_v^t, m_v^{t+1})`
- Edge features `e_vw` are explicit inputs — distinguishes MPNN from simpler aggregation-only GNNs
- Eight prior models shown as special cases by choosing M_t, U_t appropriately
- R must be permutation-invariant; sum works (GIN later proves it's the right choice)
- Achieves chemical accuracy on 11/13 QM9 targets (first model to do so)
- Foundational reference that GIN, GAT, GraphSAGE, Graphormer all cite

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md) — MPNN is the canonical unifying GNN framework
- [graph-transformer](graph-transformer.md) — Graphormer and GPS describe themselves as MPNN extensions

## Relation to Other Wiki Pages

- [xu2019gin](xu2019gin.md): GIN's proof that sum aggregation = 1-WL completeness is stated in MPNN language; sum readout in MPNN is what GIN formally justifies
- [velickovic2018gat](velickovic2018gat.md): GAT's attention-weighted sum is a special case of M_t with learned scalar weights
- [kipf2017gcn](kipf2017gcn.md): GCN's normalized mean aggregation is a special case with `M_t(h_v, h_w) = c_vw · h_w` and `U_t = sigma(W · m_v)`
- [hamilton2017graphsage](hamilton2017graphsage.md): GraphSAGE's LSTM/mean/max aggregators are special cases of M_t + U_t choices
- [ying2021graphormer](ying2021graphormer.md): Graphormer explicitly shows it can simulate GCN/GIN/GraphSAGE as special cases of its attention — itself built on the MPNN intuition
- [relational-deep-learning](relational-deep-learning.md): RDL's graph learning is implemented via HeteroGraphSAGE, which is an MPNN on a heterogeneous REG
