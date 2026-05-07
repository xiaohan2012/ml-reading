---
title: "Graph Transformers for Large Graphs"
tags: [source, graph-transformer, scalability, benchmark]
sources: [dwivedi2023largegraphs]
updated: 2026-05-07
---

# Graph Transformers for Large Graphs

**Source:** https://arxiv.org/abs/2312.11109
**Title:** Graph Transformers for Large Graphs
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Vijay Prakash Dwivedi, Yozen Liu, Anh Tuan Luu, Xavier Bresson, Neil Shah, Tong Zhao
**Venue:** arXiv 2023

## Summary

- **What:** Standard GT benchmarks (Benchmarking-GNNs, OGB) cover graphs with thousands to hundreds of thousands of nodes — but real-world applications (social networks, web graphs, biological networks) have millions to billions of nodes where $O(N^2)$ attention is infeasible.
- **How:** Introduce new large-scale graph benchmarks and systematically evaluate four GT scalability strategies: sparse attention (Exphormer), hierarchical clustering, neighborhood sampling (Gophormer, NAGphormer), and linear attention approximations.
- **So what:** Documents accuracy-scalability tradeoffs across strategies; no single approach dominates — choice depends on task type and graph structure.

## Challenges & Novelty

The standard Graph Transformer benchmarks test models on graphs with at most a few thousand nodes, where even $O(N^2)$ attention is feasible. Extending these results to large graphs requires fundamentally different scalability strategies — sampling, hierarchical clustering, and sparse attention — whose tradeoffs were not systematically studied.

- **Scale gap between benchmarks and real applications:** PCQM4Mv2 has 3.7M graphs but each has ~27 nodes — not large graphs. Real large-graph tasks (paper citation networks, social networks) have millions of nodes per graph. Filling this gap requires new datasets and evaluation protocols.
- **No single winner across scalability strategies:** sparse attention (Exphormer) preserves expressiveness but memory grows with edge count; sampling (Gophormer, NAGphormer) scales to arbitrary size but loses global context; linear attention scales cheaply but approximates the attention kernel; hierarchical methods capture multi-scale structure but add complexity.
- **Extends the Dwivedi/Bresson benchmarking tradition:** same lead author as Benchmarking-GNNs (2020) and LSPE (2022), applying the same systematic evaluation philosophy to the large-graph regime.

## Relation to Prior Work

| Paper | Focus | Graph size | Scalability strategy |
|---|---|---|---|
| [dwivedi2020benchmarking](dwivedi2020benchmarking.md) | Accuracy + PE | Small/medium | Not applicable |
| [muller2023attending](muller2023attending.md) | Taxonomy | Small/medium | Covered conceptually |
| **This paper** | Scalability + accuracy | Large (M–B nodes) | All four strategies |
| [shirzad2023exphormer](shirzad2023exphormer.md) | Sparse attention | Large | Sparse + virtual nodes |
| [wu2023sgformer](wu2023sgformer.md) | Minimal architecture | Very large (111M) | Single global layer |

- [dwivedi2020benchmarking](dwivedi2020benchmarking.md): same lead author; extends benchmark focus from small/medium to large-scale graphs.
- [shirzad2023exphormer](shirzad2023exphormer.md): Exphormer is evaluated as the sparse attention representative; shown to be the best accuracy-scalability tradeoff in the sparse category.
- [zhang2022hierarchical](zhang2022hierarchical.md), [zhao2021gophormer](zhao2021gophormer.md), [chen2022nagphormer](chen2022nagphormer.md): evaluated as representatives of hierarchical and sampling-based strategies.

## Technical Details

**Four scalability strategies evaluated:**

1. *Sparse attention* (Exphormer-style): attention only over expander-graph edges + virtual global nodes. $O(N)$ complexity. Preserves structure; limited by edge count for dense graphs.

2. *Hierarchical clustering* (Zhang et al. 2022-style): coarsen graph into clusters; attend within clusters; aggregate across levels. $O(N \cdot C)$ where $C$ = cluster size. Captures multi-scale context; additional clustering overhead.

3. *Neighborhood sampling* (Gophormer, NAGphormer): each node attends to a fixed-size sampled neighborhood. $O(N \cdot k)$ where $k$ = sample size. Scales to any graph; loses long-range dependencies.

4. *Linear attention approximation* (kernel attention, Performer-style): approximate $\text{softmax}(QK^T)$ with a kernel function. $O(N \cdot d)$. Cheapest; approximation error can be significant.

**New benchmarks.** Large-scale node classification datasets with 1M–1B nodes spanning social (Facebook, Reddit), citation (Papers100M), and biological (protein interaction) graphs. Fixed evaluation protocol: train/val/test split by timestamp or random; metrics = accuracy/F1.

## Experiments

- Sparse attention (Exphormer) achieves best accuracy across all benchmark datasets; highest scalability cost among the four strategies due to edge-based memory.
- Linear attention (SGFormer-style) scales to the largest graphs (111M nodes) with competitive accuracy; 10–140× faster inference than dense attention.
- Sampling-based methods (NAGphormer) scale well but accuracy degrades on tasks requiring long-range reasoning (SPD > 5 hops).
- Hierarchical methods offer a middle ground: better accuracy than sampling, better scalability than sparse attention for power-law degree distributions.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
