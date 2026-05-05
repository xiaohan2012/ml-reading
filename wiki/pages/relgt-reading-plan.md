---
title: "Reading Plan: Graph Transformers & RelGT"
tags: [query, analysis, graph-transformer, relational-deep-learning, positional-encoding]
sources: [alon2020bottleneck, dwivedi2021graph, dwivedi2020benchmarking, lim2022sign, kanatsoulis2025learning, rampavsek2022graphgps, zhao2021gophormer, chen2022nagphormer, hgt, relational-graph-transformer, muller2023attending, graphormer, san]
updated: 2026-05-04
---

# Reading Plan: Graph Transformers & RelGT

Sequenced reading path from GNN limitations to RelGT, ~10–12 papers. Each stage builds on the previous.

---

## Stage 1: Why Graph Transformers Exist (2 papers)

1. **[alon2020bottleneck](alon2020bottleneck.md)** — Over-squashing: exponential receptive field compressed to fixed-size vector; the core GNN failure mode that motivates global attention. Short, crisp argument.
2. **[dwivedi2021graph](dwivedi2021graph.md)** — Dwivedi & Bresson (AAAI 2021): first GT formalization; sparse attention + LapPE + BatchNorm + edge-gated attention. Sets the vocabulary for everything that follows.

---

## Stage 2: The PE Research Arc (pick 2 of 3)

3. **[dwivedi2020benchmarking](dwivedi2020benchmarking.md)** — Origin of LapPE; systematic benchmark methodology; catalyzed all subsequent PE research. Read if you want the historical foundation.
4. **[lim2022sign](lim2022sign.md)** — SignNet: ρ(φ(v)+φ(−v)) sign-invariant aggregation; universally expressive; the best single PE paper conceptually.
5. **[kanatsoulis2025learning](kanatsoulis2025learning.md)** — PEARL: GNN-based PE with linear complexity; random init + statistical pooling; direct ancestor of RelGT's subgraph GNN PE. Read this one if skipping the others.

---

## Stage 3: Scalable GT Architectures (2 papers)

6. **[rampavsek2022graphgps](rampavsek2022graphgps.md)** — The hybrid recipe: MPNN ∥ GlobalAttn per layer; PE/SE taxonomy; O(N+E). Most influential modern GT; directly adopted by RelGT as its backbone.
7. **[zhao2021gophormer](zhao2021gophormer.md)** or **[chen2022nagphormer](chen2022nagphormer.md)** — Subgraph ego-graph sampling (Gophormer) or hop tokenization (NAGphormer). Either covers the subgraph-sampling intuition behind RelGT's local K=300 attention.

---

## Stage 4: Heterogeneous Graphs (1 paper)

8. **[hgt](hgt.md)** — HGT: meta-relation parameterization W(τ(s),φ(e),τ(t)); type-dependent attention; standard baseline for heterogeneous GTs. Understand this before RelGT's node type encoder.

---

## Stage 5: RelGT (1 paper)

9. **[relational-graph-transformer](relational-graph-transformer.md)** — With Stages 1–4 complete, all five tokenization elements and both attention modules will make immediate sense. Read the lineage analysis [relgt-lineage](relgt-lineage.md) alongside.

---

## Optional Deepening

- **[muller2023attending](muller2023attending.md)** — Survey + taxonomy of ~40 GTs; PE choice dominates; confirms no prior GT handles REGs. Good landscape map.
- **[graphormer](graphormer.md)** — Full global attention via SPD bias + centrality encoding; good contrast with GPS's hybrid approach.
- **[san](san.md)** — Full Laplacian spectrum as LPE; motivates GraphGPS's shift away from LapPE toward RWSE.
- **[dwivedi2023graph](dwivedi2023graph.md)** — Survey of scalability strategies for large graphs; contextualizes RelGT's EMA centroid global module.

---

## Estimated Time

| Stage | Papers | Est. reading time |
|---|---|---|
| 1: Foundations | 2 | 3–4 hrs |
| 2: PE arc | 2 | 3–4 hrs |
| 3: Scalable GTs | 2 | 3–4 hrs |
| 4: Heterogeneous | 1 | 1–2 hrs |
| 5: RelGT | 1 | 2–3 hrs |
| **Total** | **8** | **~12–17 hrs** |
