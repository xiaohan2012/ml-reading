---
title: "Hierarchical Graph Transformer with Adaptive Node Sampling"
tags: [source, graph-transformer, scalability]
sources: [zhang2022hierarchical]
updated: 2026-05-07
---

# Hierarchical Graph Transformer with Adaptive Node Sampling

**Source:** https://arxiv.org/abs/2210.03930
**Title:** Hierarchical Graph Transformer with Adaptive Node Sampling
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Zaixi Zhang, Qi Liu, Qingyong Hu, Chee-Kong Lee
**Venue:** NeurIPS 2022

## Summary

- **What:** Existing Graph Transformer sampling strategies are static and topology-agnostic — they sample neighbors randomly without considering node importance to the learning objective, and capture only local structure.
- **How:** Train an adaptive sampling policy that samples nodes proportionally to their training-time importance (gradient-based signal); hierarchically aggregate these samples at local and global levels via a two-level GT architecture.
- **So what:** Competitive with or outperforms prior scalable GTs on node classification benchmarks while capturing both local and long-range structural dependencies.

## Challenges & Novelty

GraphSAGE and Gophormer sample fixed-size neighborhoods uniformly — ignoring the fact that different nodes contribute very differently to the training loss. Hub nodes or structurally informative nodes should be sampled more frequently. Additionally, sampling-based methods inherently miss long-range dependencies because they are bounded by the sampling horizon.

- **Adaptive sampling focuses compute on important nodes:** gradient-based node importance scoring prioritizes nodes whose features most influence the training objective — the sampler is trained jointly with the GT and updates its policy as the model improves.
- **Hierarchical two-level design captures multi-scale structure:** local GT attention operates on the sampled neighborhood (short-range, fine-grained); global level aggregation (hierarchical clustering) captures coarse community structure at longer range.
- **Complementary to sampling-only methods:** pure sampling (NAGphormer, Gophormer) loses global context; pure clustering (zhu2023hierarchical) is agnostic to importance weighting. This paper combines both.

## Relation to Prior Work

| Model | Sampling | Importance-aware | Long-range | Scalability |
|---|---|---|---|---|
| [zhao2021gophormer](zhao2021gophormer.md) | Ego-graph (uniform) | No | No | Yes |
| [chen2022nagphormer](chen2022nagphormer.md) | Hop aggregation | No | Limited | Yes |
| [zhu2023hierarchical](zhu2023hierarchical.md) | Clustering | No | Yes | Yes |
| **Zhang et al.** | Adaptive (gradient) | Yes | Yes | Yes |

- [chen2022nagphormer](chen2022nagphormer.md): NAGphormer uses hop-aggregated tokens; this paper uses importance-adaptive sampling — different inductive biases for which neighbors contribute to each node's representation.
- [zhu2023hierarchical](zhu2023hierarchical.md): both use hierarchical designs; this paper adds importance-weighted sampling on top of the hierarchical structure.
- [alon2020bottleneck](alon2020bottleneck.md): the hierarchical global level directly addresses over-squashing by providing long-range shortcuts beyond the sampling horizon.

## Technical Details

**Adaptive importance sampling.** Each node $v$ has an importance score $s_v$ updated during training:

$$s_v^{(t)} = \alpha s_v^{(t-1)} + (1 - \alpha) \left\|\frac{\partial \mathcal{L}}{\partial \mathbf{h}_v}\right\|$$

The sampling distribution for node $u$'s neighborhood: $P(v | u) \propto s_v \cdot \mathbf{1}[v \in \mathcal{N}(u)]$ — neighbors with high gradient norm are sampled more frequently.

**Local GT level.** Standard sparse GT attention over the sampled neighborhood $\hat{\mathcal{N}}(v)$:

$$\mathbf{h}_v^{(1)} = \text{SparseAttn}\!\left(\mathbf{h}_v^{(0)}, \{\mathbf{h}_u^{(0)} : u \in \hat{\mathcal{N}}(v)\}\right)$$

**Global level.** Hierarchical clustering (e.g., METIS or spectral clustering) produces clusters $\{C_1, \ldots, C_K\}$. Cluster representation: $\mathbf{h}_{C_k} = \text{mean}(\{\mathbf{h}_v^{(1)} : v \in C_k\})$. Cluster-level attention:

$$\mathbf{h}_{C_k}^{(2)} = \text{GlobalAttn}(\mathbf{h}_{C_k}^{(1)}, \{\mathbf{h}_{C_j}^{(1)} : j = 1, \ldots, K\})$$

Node-level update: $\mathbf{h}_v^{(2)} = \text{MLP}([\mathbf{h}_v^{(1)} \| \mathbf{h}_{C(v)}^{(2)}])$ where $C(v)$ is $v$'s cluster.

## Experiments

- ogbn-arxiv (170K nodes): outperforms GCN, GraphSAGE, and Gophormer by 1–3% accuracy; competitive with NAGphormer.
- ogbn-products (2.4M nodes): scales without memory overflow; importance sampling reduces required sample size by 30–40% at same accuracy vs. uniform sampling.
- Ablation: removing adaptive sampling (uniform replacement) degrades accuracy 1.5%; removing global level degrades 2.1% — both components contribute independently.
- Training convergence: adaptive sampling stabilizes after ~10 epochs as importance scores become reliable.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
