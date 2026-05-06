---
title: "RelGT Intellectual Lineage"
tags: [query, analysis, graph-transformer, relational-deep-learning, positional-encoding]
sources: [dwivedi2025relgt, rampavsek2022graphgps, ying2021graphormer, kreuzer2021san, dwivedi2021graph, dwivedi2020benchmarking, dwivedi2022graph, dwivedi2023graph, lim2022signnet, huang2024stability, kanatsoulis2025learning, canturk2023graph, zhao2021gophormer, chen2022nagphormer, zhang2022hierarchical, wu2023sgformer, alon2020bottleneck, kim2022pure, mao2023hinormer, zhu2025hhgt, muller2023attending, hu2020hgt, schlichtkrull2018rgcn, xu2020tgat, fey2024rdlposition, robinson2024relbench]
updated: 2026-05-06
---

# RelGT Intellectual Lineage

How each RelGT design decision traces back to prior work, organized by thread.

---

## Thread 1: Problem Framing — Relational Deep Learning

> Cvitkovic 2020 — first REG formulation
> ↓
> [fey2024rdlposition](fey2024rdlposition.md) (Fey et al. 2023) — RDL position paper: end-to-end learning from relational DBs
> ↓
> [robinson2024relbench](robinson2024relbench.md) (Robinson et al. 2025) — benchmark: temporal splits, multi-table tasks
> ↓
> **[dwivedi2025relgt](dwivedi2025relgt.md)** — targets REGs with schema-defined heterogeneity, temporality, and scale

The three challenges RelGT must solve — schema-defined, temporal, multi-type heterogeneity — were all articulated by the RDL position paper.

---

## Thread 2: Why GNNs Aren't Enough

> [gilmer2017mpnn](gilmer2017mpnn.md) (Gilmer et al. 2017) — MPNN: message passing framework
> ↓
> [xu2019gin](xu2019gin.md) (Xu et al. 2018) — 1-WL expressiveness ceiling
> ↓
> [alon2020bottleneck](alon2020bottleneck.md) (Alon & Yahav 2021) — over-squashing: O(exp(K)) receptive field → fixed vector; FA layer fix
> ↓
> **RelGT motivation** — GNNs cannot distinguish non-isomorphic REG subgraphs; global attention needed

---

## Thread 3: Graph Transformer Foundations → Hybrid Architecture

> Vaswani et al. 2017 — Transformer: attention as pairwise interaction
> ↓
> [dwivedi2021graph](dwivedi2021graph.md) (Dwivedi & Bresson 2021) — first GT: LapPE + sparse local attention + BatchNorm + edge-gated attention
> ↓

| [ying2021graphormer](ying2021graphormer.md) (Ying et al. 2021) | [kreuzer2021san](kreuzer2021san.md) (Kreuzer et al. 2021) |
|---|---|
| global full attention; SPD bias; centrality encoding | full Laplacian spectrum as LPE; full attention; motivates RWSE move |

> ↓
> [rampavsek2022graphgps](rampavsek2022graphgps.md) (Rampášek et al. 2022) — hybrid: local MPNN + global Transformer in parallel; PE/SE taxonomy; O(N+E)
> ↓
> **RelGT** — adopts GPS hybrid architecture: local K=300 all-pair attention + global EMA centroid attention

---

## Thread 4: Positional Encodings → Subgraph GNN PE

> [dwivedi2020benchmarking](dwivedi2020benchmarking.md) (Dwivedi et al. 2020) — LapPE: Laplacian eigenvectors as node PE; first systematic GNN benchmark
> ↓
> [dwivedi2022graph](dwivedi2022graph.md) / LSPE (Dwivedi et al. 2022) — decoupled structural+positional streams; learnable PE through layers
> ↓
> [lim2022signnet](lim2022signnet.md) / SignNet (Lim et al. 2022) — ρ(φ(v)+φ(−v)): sign-invariant aggregation; universally expressive
> ↓
> [huang2024stability](huang2024stability.md) / SPE (Huang et al. 2024) — first provably stable AND expressive PE; eigenvalue soft-partitioning
> ↓
> [kanatsoulis2025learning](kanatsoulis2025learning.md) / PEARL (Kanatsoulis et al. 2025) — GNNs as eigenvector mappings; random init + pooling; O(N) complexity
> ↓
> **RelGT Subgraph GNN PE** — PEARL applied locally; Z_random resampled each training step; no global eigendecomposition

Side branch: [canturk2023graph](canturk2023graph.md) / GPSE — pre-trained multi-PSE encoder combining RWSE+LapPE+SignNet; informs RelGT's decision to use GNN-based rather than spectral PE.

---

## Thread 5: Scalable Subgraph Attention → Local Module

> [zhao2021gophormer](zhao2021gophormer.md) / Gophormer (Zhao et al. 2021) — ego-graph subgraph sampling; each node attends only to local ego-graph; first scalable GT via subgraph decomposition
> ↓
> [chen2022nagphormer](chen2022nagphormer.md) / NAGphormer (Chen et al. 2022) — Hop2Token: compress k-hop neighborhood into per-hop token sequence; memory-efficient; ICLR 2023
> ↓
> **RelGT Local Attention Module** — K=300 nearest neighbors via temporal-aware BFS sampling; full all-pair attention within local neighborhood

---

## Thread 6: Global Centroid Attention → Global Module

> [zhang2022hierarchical](zhang2022hierarchical.md) (Zhang et al. 2022) — hierarchical GT; adaptive coarsening for scale
> ↓
> Kong et al. 2023 / GOAT — EMA centroids as global summary tokens; nodes attend to B learned centroids; O(NB) instead of O(N²)
> ↓
> [dwivedi2023graph](dwivedi2023graph.md) (Dwivedi et al. 2023) — GT for large graphs: survey + benchmarks; validates centroid-based global attention
>
> [wu2023sgformer](wu2023sgformer.md) / SGFormer (Wu et al. 2023) — single global attention layer; 141× speedup; scales to 111M nodes
> ↓
> **RelGT Global Attention Module** — B=4096 EMA centroids updated via exponential moving average; each node attends to all B centroids (not all N nodes)

---

## Thread 7: Heterogeneous Handling → Node Type Encoder

> [schlichtkrull2018rgcn](schlichtkrull2018rgcn.md) / R-GCN (Schlichtkrull et al. 2018) — first message passing for heterogeneous graphs; relation-type-specific weights
> ↓
> [hu2020hgt](hu2020hgt.md) / HGT (Hu et al. 2020) — meta-relation parameterization: W(τ(s),φ(e),τ(t)); type-dependent attention
> ↓
> [mao2023hinormer](mao2023hinormer.md) / HINormer (Mao et al. 2023) — local HIN encoder + global Transformer; decouples heterogeneity from global attention
> ↓
> **RelGT** — node type as explicit token element (one of 5); type-conditioned projection; handles REG's schema-defined types

[muller2023attending](muller2023attending.md) survey confirms: none of ~40 surveyed GTs handles REG's combination of heterogeneity + temporality + scale.

---

## Thread 8: Temporal Handling → Relative Time Encoder

> [xu2020tgat](xu2020tgat.md) / TGAT (Xu et al. 2020) — time2vec: sinusoidal time encoding; attention conditioned on Δt
> ↓
> [rossi2020tgn](rossi2020tgn.md) / TGN (Rossi et al. 2020) — memory module + temporal graph attention; unifies TGAT/DyRep; 30× faster than TGAT
>
> [yu2023dygformer](yu2023dygformer.md) / DyGFormer (Yu et al. 2023) — first-hop Transformer + patching + neighbor co-occurrence encoding; NeurIPS 2023 SOTA
> ↓
> HTGformer (Wang, SIGIR 2025) — heterogeneous + temporal GT; handles both jointly via separate iterative modules; no PE; not yet ingested
> ↓
> **RelGT Relative Time Encoder** — Δt = t_seed − t_neighbor encoded as token feature; temporal leakage prevention: mask future edges during training; unlike HTGformer, integrates PE + temporal + heterogeneity in a unified tokenization

---

## Thread 9: Multimodal Features → MultiModalEncoder

> Hu et al. 2024 / PyTorch Frame — unified tabular features: numerical, categorical, text, embedding; per-column encoders → unified representation
> ↓
> **RelGT MultiModalEncoder** — wraps PyTorch Frame; handles heterogeneous column types per entity table

---

## Summary Table

| RelGT Design Choice | Key Predecessors |
|---|---|
| Hybrid local+global attention | [rampavsek2022graphgps](rampavsek2022graphgps.md) |
| Local ego-graph attention | [zhao2021gophormer](zhao2021gophormer.md), [chen2022nagphormer](chen2022nagphormer.md) |
| EMA centroid global tokens | GOAT → [dwivedi2023graph](dwivedi2023graph.md) |
| Subgraph GNN PE | [kanatsoulis2025learning](kanatsoulis2025learning.md) / PEARL |
| Node type encoding | [hu2020hgt](hu2020hgt.md) |
| Temporal Δt encoding | [xu2020tgat](xu2020tgat.md), [rossi2020tgn](rossi2020tgn.md), [yu2023dygformer](yu2023dygformer.md) |
| Closest prior heterogeneous+temporal GT | HTGformer (Wang 2025) — lacks PE, separate modules |
| Multimodal node features | PyTorch Frame |
| Problem definition | [fey2024rdlposition](fey2024rdlposition.md) |
| GNN limitation motivation | [alon2020bottleneck](alon2020bottleneck.md), [xu2019gin](xu2019gin.md) |

RelGT's novelty is not any single component but the integration: no prior GT handled the REG setting's simultaneous requirements of schema-defined heterogeneity, temporal ordering, and graph scale.
