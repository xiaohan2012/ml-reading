---
title: "Graph Attention Networks (GAT)"
tags: [source, gnn, attention, node-classification]
sources: [velickovic2018gat]
updated: 2026-05-07
---

# Graph Attention Networks (GAT)

**Source:** https://arxiv.org/abs/1710.10903
**Title:** Graph Attention Networks
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Veličković, Cucurull, Casanova, Romero, Liò, Bengio
**Venue:** ICLR 2018

## Summary

- **What:** GCN assigns all neighbors the same degree-normalized weight, ignoring that different neighbors carry different relevance to the prediction task.
- **How:** Compute learned scalar attention coefficients $\alpha_{ij}$ from node feature pairs via a shared feedforward network + softmax; aggregate with attention-weighted sums using multi-head attention for stability.
- **So what:** SOTA on transductive (Cora, Citeseer, Pubmed) and inductive (PPI) node classification at ICLR 2018; first feature-based attention mechanism for graph-structured data.

## Challenges & Novelty

GCN's propagation rule $\hat{A}_{vw}Wh_w$ is isotropic — edge weight is determined entirely by node degrees, not by feature compatibility. This is both a modeling limitation (two chemically distinct bonds get the same weight) and a motivating failure case (degree-1 noise nodes receive equal weight to high-degree meaningful neighbors).

- **No global graph structure required:** attention coefficients $\alpha_{ij}$ are computed locally from $(\mathbf{h}_i, \mathbf{h}_j)$ — no eigendecomposition or full graph Laplacian needed. GAT is immediately applicable to inductive and heterogeneous settings.
- **Multi-head stabilizes noisy attention:** a single attention head is high-variance; $K$ independent heads run in parallel, concatenated at hidden layers and averaged at the final layer, yielding stable and expressive representations.
- **Dropout on attention coefficients:** applying dropout to $\alpha_{ij}$ (not just node features) is critical for generalization with small training sets — each training step, each node sees a stochastically sampled neighborhood.

## Relation to Prior Work

| Model | Neighbor weighting | Inductive | Edge types | Global structure |
|---|---|---|---|---|
| [kipf2017gcn](kipf2017gcn.md) | Degree-normalized (fixed) | No | No | Yes (full $\hat{A}$) |
| [hamilton2017graphsage](hamilton2017graphsage.md) | Uniform or max-pool | Yes | No | No |
| **GAT** | Learned attention ($\alpha_{ij}$) | Yes | No | No |
| [schlichtkrull2018rgcn](schlichtkrull2018rgcn.md) | Uniform (per relation type) | No | Yes | No |
| [hu2020hgt](hu2020hgt.md) | Attention (per meta-relation) | Yes | Yes | No |

- [kipf2017gcn](kipf2017gcn.md): GAT replaces GCN's fixed degree normalization with learned attention at the same $O(|\mathcal{E}|)$ complexity per layer.
- [hamilton2017graphsage](hamilton2017graphsage.md): both handle inductive settings; GAT attends over the full neighborhood while GraphSAGE samples a fixed-size subset.
- [xu2019gin](xu2019gin.md): GIN's theory shows attention-weighted mean (GAT) is not injective — GAT improves capacity over GCN but remains strictly below WL expressive power.
- [hu2020hgt](hu2020hgt.md): HGT extends GAT-style attention to heterogeneous graphs using type-specific key/query projection matrices per meta-relation triplet (src_type, edge_type, dst_type).

## Technical Details

**Attention coefficient computation.** For edge $(i, j)$:

$$e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}^T\!\left[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j\right]\right)$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

where $\mathbf{W} \in \mathbb{R}^{F' \times F}$ is a shared linear transformation and $\mathbf{a} \in \mathbb{R}^{2F'}$ is a learnable attention vector. LeakyReLU (negative slope 0.2) allows non-zero gradients on negative activations.

**Aggregation:**

$$\mathbf{h}'_i = \sigma\!\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}\mathbf{h}_j\right)$$

**Multi-head attention.** Run $K$ independent attention mechanisms; concatenate outputs at intermediate layers:

$$\mathbf{h}'_i = \Big\|_{k=1}^K \sigma\!\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k\mathbf{h}_j\right)$$

At the final (prediction) layer, average instead of concatenate to control output dimensionality.

**Complexity:** $O(|\mathcal{V}|FF' + |\mathcal{E}|F')$ per head — same order as GCN. Fully parallelizable over all edges.

**GAT v2 (Brody et al. 2022):** shows the original attention is "static" — $e_{ij}$ depends on $\mathbf{h}_i$ and $\mathbf{h}_j$ through their projection, but can be decomposed as a ranking over neighbors that doesn't change with the query. GATv2 fixes this with $e_{ij} = \mathbf{a}^T \text{LeakyReLU}(\mathbf{W}[\mathbf{h}_i \| \mathbf{h}_j])$ (attention vector applied after activation, not before).

## Experiments

- Transductive: Cora 83.0%, Citeseer 72.5%, Pubmed 79.0% — SOTA or tied at ICLR 2018 on all three.
- Inductive PPI (20 training graphs, 2 unseen test graphs): micro-F1 = 0.973 vs. GraphSAGE-LSTM 0.612 — demonstrates that learned attention substantially outperforms fixed aggregation on multi-label graph tasks.
- Ablation: multi-head ($K=8$) consistently outperforms single-head; removing dropout on $\alpha_{ij}$ degrades Citeseer/Cora performance by 0.5–1.5%.
- Attention visualizations on Cora confirm that semantically similar nodes (same paper category) receive higher attention weights.

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md)
