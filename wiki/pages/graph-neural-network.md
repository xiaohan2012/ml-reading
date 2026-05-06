---
title: Graph Neural Network
tags: [entity, concept, gnn, message-passing]
sources: [kipf2017gcn, hamilton2017inductive, velickovic2018gat, xu2019gin, gilmer2017mpnn, schlichtkrull2018rgcn, morris2019kgnn]
updated: 2026-04-29
---

# Graph Neural Network

## Description

A Graph Neural Network (GNN) is a neural architecture that learns node (and graph) representations by iteratively aggregating information from local neighborhoods — implementing a form of learned message passing. Each layer updates each node's representation by aggregating its neighbors' representations from the previous layer, then combining with the node's own state.

The general update rule:

$$h_v^{(k)} = \phi\!\left(h_v^{(k-1)},\, f\!\left(\{h_u^{(k-1)} : u \in \mathcal{N}(v)\}\right)\right)$$

where $f$ is the **aggregation function** (mean, sum, max, attention-weighted sum, …) and $\phi$ combines the aggregated neighborhood with the node's own state.

**Key design choices:**
- **Aggregator**: determines what structural information is preserved (GIN proves sum > mean > max in expressive power)
- **Combination**: how to merge self and neighbor representations (concatenate vs. sum vs. gating)
- **Normalization**: degree normalization (GCN), none (GIN), or learned (GAT attention weights)

**Expressiveness bound (GIN):** Any aggregation-based GNN is at most as powerful as the Weisfeiler-Lehman (WL) graph isomorphism test. GIN achieves this upper bound using sum aggregation + MLP.

## Foundational Variants

| Model | Aggregator | Expressive power | Setting |
|-------|-----------|-----------------|---------|
| GCN | Degree-normalized mean | < WL | Transductive |
| GraphSAGE | Mean / LSTM / Max (sampled) | < WL | Inductive |
| GAT | Attention-weighted sum | < WL | Both |
| GIN | Sum + MLP | = WL | Both |
| R-GCN | Relation-typed sum | = WL per relation | Inductive |
| k-GNN | Tuple-level sum | > WL (k≥2) | Both |

## Appearances in Sources

- [kipf2017gcn](kipf2017gcn.md) — introduces spectral GNNs via Chebyshev approximation; symmetric degree normalization; transductive only
- [hamilton2017graphsage](hamilton2017graphsage.md) — inductive GNNs via neighborhood sampling + learnable aggregation; foundation for HeteroGraphSAGE (RDL baseline)
- [velickovic2018gat](velickovic2018gat.md) — replaces isotropic aggregation with learned attention weights; first GNN to assign different importance per neighbor without precomputed structure
- [xu2019gin](xu2019gin.md) — proves theoretical upper bound on GNN expressive power; shows GCN/GraphSAGE mean aggregation is not injective; GIN achieves WL-level power via sum+MLP
- [gilmer2017mpnn](gilmer2017mpnn.md) — Gilmer et al. (ICML 2017): canonical MPNN framework unifying 8+ prior models; edge features as first-class inputs; SOTA on QM9
- [schlichtkrull2018rgcn](schlichtkrull2018rgcn.md) — Schlichtkrull et al. (ESWC 2018): relation-specific weight matrices for multi-relational/knowledge graphs; basis decomposition for parameter efficiency
- [morris2019kgnn](morris2019kgnn.md) — Morris et al. (AAAI 2019): proves 1-GNNs ≡ 1-WL; k-GNNs operate on k-tuples to exceed 1-WL; 54% MAE reduction on QM9

## Limitations

- **Over-squashing** ([alon2020bottleneck](alon2020bottleneck.md)): information from exponentially-growing receptive fields is compressed into fixed-size vectors; long-range signals fail to propagate; GCN/GIN more susceptible than GAT/GGNN. Motivates global attention in Graph Transformers.
- **Expressiveness ceiling** ([xu2019gin](xu2019gin.md)): no MPNN exceeds 1-WL; [morris2019kgnn](morris2019kgnn.md) and [kim2022pure](kim2022pure.md) break this ceiling via higher-order methods.

## Related Concepts

- [graph-transformer](graph-transformer.md) — extends GNNs with global self-attention; overcomes GNNs' local neighborhood limitation
- [positional-encoding](positional-encoding.md) — augments GNNs/GTs with structural/positional signals beyond node features
- [relational-deep-learning](relational-deep-learning.md) — applies GNNs (specifically HeteroGraphSAGE) to relational database graphs
