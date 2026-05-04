---
title: "RelGT Intellectual Lineage"
tags: [query, analysis, graph-transformer, relational-deep-learning, positional-encoding]
sources: [relational-graph-transformer, graphgps, graphormer, san, generalization-transformer-graphs, dwivedi2020benchmarking, dwivedi2022graph, dwivedi2023graph, limsign, huangstability, kanatsoulis2025learning, canturk2023graph, zhao2021gophormer, chen2022nagphormer, zhang2022hierarchical, wu2023sgformer, alon2020bottleneck, kim2022pure, mao2023hinormer, zhu2025hhgt, muller2023attending, hgt, rgcn, tgat, relational-deep-learning-position, relbench-v1]
updated: 2026-05-04
---

# RelGT Intellectual Lineage

How each RelGT design decision traces back to prior work, organized by thread.

---

## Thread 1: Problem Framing — Relational Deep Learning

```
Cvitkovic 2020 (first REG formulation)
    ↓
Fey et al. 2023 [relational-deep-learning-position](relational-deep-learning-position.md)
(RDL position paper: end-to-end learning from relational DBs)
    ↓
Robinson et al. 2025 / RelBench [relbench-v1](relbench-v1.md)
(benchmark: temporal splits, multi-table tasks)
    ↓
RelGT: targets REGs with schema-defined heterogeneity, temporality, and scale
```

The three challenges RelGT must solve — schema-defined, temporal, multi-type heterogeneity — were all articulated by the RDL position paper.

---

## Thread 2: Why GNNs Aren't Enough

```
Gilmer et al. 2017 [mpnn](mpnn.md) (MPNN: message passing framework)
    ↓
Xu et al. 2018 [gin](gin.md) (1-WL expressiveness ceiling)
    ↓
Alon & Yahav 2021 [alon2020bottleneck](alon2020bottleneck.md)
(over-squashing: O(exp(K)) receptive field → fixed vector; FA layer fix)
    ↓
RelGT motivation: GNNs cannot distinguish non-isomorphic REG subgraphs;
global attention needed
```

---

## Thread 3: Graph Transformer Foundations → Hybrid Architecture

```
Vaswani et al. 2017 (Transformer: attention as pairwise interaction)
    ↓
Dwivedi & Bresson 2021 [generalization-transformer-graphs](generalization-transformer-graphs.md)
(first GT: LapPE + sparse local attention + BatchNorm + edge-gated attention)
    ↓
  ┌─────────────────────────────────────────────┐
  │                                             │
Ying et al. 2021 [graphormer](graphormer.md)   Kreuzer et al. 2021 [san](san.md)
(global full attention; SPD bias;              (full Laplacian spectrum as LPE;
 centrality encoding)                           full attention; motivates RWSE move)
  │                                             │
  └──────────────┬──────────────────────────────┘
                 ↓
    Rampášek et al. 2022 [graphgps](graphgps.md)
    (hybrid: local MPNN + global Transformer in parallel;
     PE/SE taxonomy: LapPE, RWSE, SignNet; O(N+E))
                 ↓
    RelGT: adopts GPS hybrid architecture —
    local K=300 all-pair attention + global EMA centroid attention
```

---

## Thread 4: Positional Encodings → Subgraph GNN PE

```
Dwivedi et al. 2020 [dwivedi2020benchmarking](dwivedi2020benchmarking.md)
(LapPE: Laplacian eigenvectors as node PE; first systematic GNN benchmark)
    ↓
Dwivedi et al. 2022 [dwivedi2022graph](dwivedi2022graph.md) / LSPE
(decoupled structural+positional streams; learnable PE through layers)
    ↓
Lim et al. 2022 [limsign](limsign.md) / SignNet
(ρ(φ(v)+φ(−v)): sign-invariant aggregation; universally expressive)
    ↓
Huang et al. 2024 [huangstability](huangstability.md) / SPE
(first provably stable AND expressive PE; eigenvalue soft-partitioning)
    ↓
Kanatsoulis et al. 2025 [kanatsoulis2025learning](kanatsoulis2025learning.md) / PEARL
(GNNs as eigenvector mappings; random init + pooling; O(N) complexity)
    ↓
RelGT: Subgraph GNN PE = PEARL applied locally
    h_pe(vj) = GNN(A_local, Z_random)_j
    Z_random resampled each training step → stochastic robustness
    no global eigendecomposition needed at REG scale
```

Side branch: [canturk2023graph](canturk2023graph.md) / GPSE — pre-trained multi-PSE encoder combining RWSE+LapPE+SignNet; informs RelGT's decision to use GNN-based rather than spectral PE.

---

## Thread 5: Scalable Subgraph Attention → Local Module

```
Zhao et al. 2021 [zhao2021gophormer](zhao2021gophormer.md) / Gophormer
(ego-graph subgraph sampling; each node attends only to local ego-graph;
 first scalable GT via subgraph decomposition)
    ↓
Chen et al. 2022 [chen2022nagphormer](chen2022nagphormer.md) / NAGphormer
(Hop2Token: compress k-hop neighborhood into per-hop token sequence;
 memory-efficient; ICLR 2023)
    ↓
RelGT Local Attention Module:
    K=300 nearest neighbors via temporal-aware BFS sampling
    full all-pair attention within local neighborhood
```

---

## Thread 6: Global Centroid Attention → Global Module

```
Zhang et al. 2022 [zhang2022hierarchical](zhang2022hierarchical.md)
(hierarchical GT; adaptive coarsening for scale)
    ↓
Kong et al. 2023 / GOAT
(EMA centroids as global summary tokens; nodes attend to B learned centroids;
 O(NB) instead of O(N²))
    ↓
Dwivedi et al. 2023 [dwivedi2023graph](dwivedi2023graph.md)
(GT for large graphs: survey + benchmarks; validates centroid-based global attention)
    ↓
Wu et al. 2023 [wu2023sgformer](wu2023sgformer.md) / SGFormer
(single global attention layer; 141× speedup; scales to 111M nodes)
    ↓
RelGT Global Attention Module:
    B=4096 EMA centroids updated via exponential moving average
    each node attends to all B centroids (not all N nodes)
```

---

## Thread 7: Heterogeneous Handling → Node Type Encoder

```
Schlichtkrull et al. 2018 [rgcn](rgcn.md) / R-GCN
(first message passing for heterogeneous graphs; relation-type-specific weights)
    ↓
Hu et al. 2020 [hgt](hgt.md) / HGT
(meta-relation parameterization: W(τ(s),φ(e),τ(t)); type-dependent attention)
    ↓
Mao et al. 2023 [mao2023hinormer](mao2023hinormer.md) / HINormer
(local HIN encoder + global Transformer; decouples heterogeneity from global attention)
    ↓
RelGT: node type as explicit token element (one of 5);
    type-conditioned projection; handles REG's schema-defined types
```

[muller2023attending](muller2023attending.md) survey confirms: none of ~40 surveyed GTs handles REG's combination of heterogeneity + temporality + scale.

---

## Thread 8: Temporal Handling → Relative Time Encoder

```
Xu et al. 2020 [tgat](tgat.md) / TGAT
(time2vec: sinusoidal time encoding; attention conditioned on Δt)
    ↓
RelGT Relative Time Encoder:
    Δt = t_seed − t_neighbor encoded as token feature
    temporal leakage prevention: mask future edges during training
```

---

## Thread 9: Multimodal Features → MultiModalEncoder

```
Hu et al. 2024 / PyTorch Frame
(unified tabular features: numerical, categorical, text, embedding;
 per-column encoders → unified representation)
    ↓
RelGT MultiModalEncoder:
    wraps PyTorch Frame; handles heterogeneous column types per entity table
```

---

## Summary Table

| RelGT Design Choice | Key Predecessors |
|---|---|
| Hybrid local+global attention | [graphgps](graphgps.md) |
| Local ego-graph attention | [zhao2021gophormer](zhao2021gophormer.md), [chen2022nagphormer](chen2022nagphormer.md) |
| EMA centroid global tokens | GOAT → [dwivedi2023graph](dwivedi2023graph.md) |
| Subgraph GNN PE | [kanatsoulis2025learning](kanatsoulis2025learning.md) / PEARL |
| Node type encoding | [hgt](hgt.md) |
| Temporal Δt encoding | [tgat](tgat.md) |
| Multimodal node features | PyTorch Frame |
| Problem definition | [relational-deep-learning-position](relational-deep-learning-position.md) |
| GNN limitation motivation | [alon2020bottleneck](alon2020bottleneck.md), [gin](gin.md) |

RelGT's novelty is not any single component but the integration: no prior GT handled the REG setting's simultaneous requirements of schema-defined heterogeneity, temporal ordering, and graph scale.
