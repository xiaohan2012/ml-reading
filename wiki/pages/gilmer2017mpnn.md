---
title: "MPNN: Neural Message Passing for Quantum Chemistry"
tags: [source, gnn, message-passing, framework, molecular]
sources: [gilmer2017mpnn]
updated: 2026-05-07
---

# MPNN: Neural Message Passing for Quantum Chemistry

**Source:** https://arxiv.org/abs/1704.01212
**Title:** Neural Message Passing for Quantum Chemistry
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Gilmer, Schütt, Riley, Vinyals, Sadowski, Ghahramani
**Venue:** ICML 2017

## Summary

- **What:** At least 8 GNN variants had been independently proposed for molecular property prediction with no unified view of their shared structure.
- **How:** Abstract all neighborhood-aggregation GNNs into three learnable functions — message function $M_t$, update function $U_t$, and readout function $R$ — and show each prior model is a special case.
- **So what:** Canonical GNN framework used to organize the entire field; the paper's own MPNN variant achieves chemical accuracy on 11/13 QM9 targets, the first model to do so.

## Challenges & Novelty

Prior molecular GNN papers (Duvenaud 2015, Kearnes 2016, DTNN 2017, Interaction Networks) each introduced distinct notation and architectural choices, making comparison difficult. There was no principled way to identify which design decisions drove performance.

- **Unification reveals hidden commonality:** by defining $M_t$, $U_t$, $R$ as abstract functions, MPNN subsumes GCN ($M_t = c_{vw}W h_w$, $U_t = \sigma(Wm)$), GIN (MLP update, sum aggregation), GAT (attention-weighted $M_t$), GGNN (GRU update), Interaction Networks, DTNN, and more — 8+ models as special cases.
- **Edge features are first-class:** the message function $M_t(\mathbf{h}_v, \mathbf{h}_w, \mathbf{e}_{vw})$ explicitly takes edge features as input, unlike GCN/GAT which treat all edges identically. This is critical for molecules where bond type, distance, and angle are primary signals.
- **Readout must be permutation-invariant:** $R$ operates on the set $\{h_v^T\}$; the paper uses a learned invariant aggregation (Set2Set) for graph-level prediction.

## Relation to Prior Work

| Model | Special case of MPNN | $M_t$ form | $U_t$ form | Edge features |
|---|---|---|---|---|
| [kipf2017gcn](kipf2017gcn.md) | Yes | $\hat{A}_{vw} W h_w$ | Sum + activation | No |
| [hamilton2017graphsage](hamilton2017graphsage.md) | Yes | $W h_w$ | Concat + MLP | No |
| [velickovic2018gat](velickovic2018gat.md) | Yes | $\alpha_{vw} W h_w$ | Weighted sum | No |
| [xu2019gin](xu2019gin.md) | Yes | $h_w$ | $(1+\varepsilon)h_v + \text{sum}$, MLP | No |
| GGNN | Yes | $W_e h_w$ (edge-typed) | GRU | No |
| DTNN | Yes | Gaussian-weighted | Sum | Yes |
| **MPNN (paper variant)** | — | $A(h_v, h_w) e_{vw}$ | GRU | Yes |

- [xu2019gin](xu2019gin.md): GIN's theoretical contribution (sum = injective = WL power) is stated in MPNN language — it answers *which* $M_t$/$U_t$ choice maximizes expressive power.
- [velickovic2018gat](velickovic2018gat.md): GAT's attention-weighted mean is a special case with $M_t(h_v, h_w) = \alpha_{vw}(h_v, h_w) W h_w$.
- [kipf2017gcn](kipf2017gcn.md): GCN's normalized aggregation is a special case with fixed scalar $M_t = \hat{A}_{vw} W h_w$ and $U_t = \sigma(\cdot)$.

## Technical Details

**Message passing phase** ($T$ steps):

$$\mathbf{m}_v^{t+1} = \sum_{w \in \mathcal{N}(v)} M_t\!\left(\mathbf{h}_v^t,\, \mathbf{h}_w^t,\, \mathbf{e}_{vw}\right)$$

$$\mathbf{h}_v^{t+1} = U_t\!\left(\mathbf{h}_v^t,\, \mathbf{m}_v^{t+1}\right)$$

**Readout phase:**

$$\hat{y} = R\!\left(\left\{\mathbf{h}_v^T \mid v \in G\right\}\right)$$

where $R$ must be permutation-invariant. The paper uses Set2Vec (a recurrent attentional set aggregation) as $R$ for graph-level regression.

**Paper's own MPNN.** $M_t(\mathbf{h}_v, \mathbf{h}_w, \mathbf{e}_{vw}) = A(\mathbf{e}_{vw})\mathbf{h}_w$ where $A(\mathbf{e}_{vw})$ is a learned edge-specific matrix (different from a shared $W$). $U_t$ is a GRU: $\mathbf{h}_v^{t+1} = \text{GRU}(\mathbf{h}_v^t, \mathbf{m}_v^{t+1})$. Virtual nodes (connected to all real nodes) are added to propagate global information.

**QM9 task.** 130k molecules, 13 quantum mechanical properties (energy, polarizability, etc.) computed by DFT. Chemical accuracy = within 1 kcal/mol for energy targets.

## Experiments

- MPNN achieves chemical accuracy on 11/13 QM9 targets — prior best was 0/13 at the time of submission.
- Ablation: removing edge features (edge matrix $A(e_{vw})$ → fixed $W$) drops accuracy on 6/13 targets, confirming bond-type specificity is essential.
- Removing the virtual node degrades global property predictions (dipole moment, HOMO/LUMO gap) more than local ones (atomization energy).
- Set2Vec readout outperforms plain sum readout on 8/13 targets.

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md)
- [graph-transformer](graph-transformer.md)
