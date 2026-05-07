---
title: "Reading Plan: Graph Transformers & RelGT"
tags: [query, analysis, graph-transformer, relational-deep-learning, positional-encoding]
sources: [alon2020bottleneck, dwivedi2021gt, dwivedi2020benchmarking, lim2022signnet, kanatsoulis2025pearl, rampavsek2022graphgps, zhao2021gophormer, chen2022nagphormer, hu2020hgt, dwivedi2025relgt, muller2023attending, ying2021graphormer, kreuzer2021san]
updated: 2026-05-04
---

# Reading Plan: Graph Transformers & RelGT

Sequenced reading path from GNN limitations to RelGT, ~10–12 papers. Each stage builds on the previous.

---

## Stage 1: Why Graph Transformers Exist (2 papers)

1. **[alon2020bottleneck](alon2020bottleneck.md)** — Over-squashing: exponential receptive field compressed to fixed-size vector; the core GNN failure mode that motivates global attention. Short, crisp argument.
2. **[dwivedi2021gt](dwivedi2021gt.md)** — Dwivedi & Bresson (AAAI 2021): first GT formalization; sparse attention + LapPE + BatchNorm + edge-gated attention. Sets the vocabulary for everything that follows.

---

## Stage 2: The PE Research Arc (pick 2 of 3)

3. **[dwivedi2020benchmarking](dwivedi2020benchmarking.md)** — Origin of LapPE; systematic benchmark methodology; catalyzed all subsequent PE research. Read if you want the historical foundation.
4. **[lim2022signnet](lim2022signnet.md)** — SignNet: ρ(φ(v)+φ(−v)) sign-invariant aggregation; universally expressive; the best single PE paper conceptually.
5. **[kanatsoulis2025pearl](kanatsoulis2025pearl.md)** — PEARL: GNN-based PE with linear complexity; random init + statistical pooling; direct ancestor of RelGT's subgraph GNN PE. Read this one if skipping the others.

---

## Stage 3: Scalable GT Architectures (2 papers)

6. **[rampavsek2022graphgps](rampavsek2022graphgps.md)** — The hybrid recipe: MPNN ∥ GlobalAttn per layer; PE/SE taxonomy; O(N+E). Most influential modern GT; directly adopted by RelGT as its backbone.
7. **[zhao2021gophormer](zhao2021gophormer.md)** or **[chen2022nagphormer](chen2022nagphormer.md)** — Subgraph ego-graph sampling (Gophormer) or hop tokenization (NAGphormer). Either covers the subgraph-sampling intuition behind RelGT's local K=300 attention.

---

## Stage 4: Heterogeneous & Temporal Graphs (2 papers)

8. **[hu2020hgt](hu2020hgt.md)** — HGT: meta-relation parameterization W(τ(s),φ(e),τ(t)); type-dependent attention; standard baseline for heterogeneous GTs. Understand this before RelGT's node type encoder.
9. **HTGformer** (Wang, SIGIR 2025) — the closest prior work to RelGT: heterogeneous + temporal GT, but handles both via separate iterative modules and lacks PE. RelGT directly positions itself against this. *Not yet ingested — worth reading before RelGT.*

---

## Stage 5: RelGT (1 paper)

10. **[dwivedi2025relgt](dwivedi2025relgt.md)** — With Stages 1–4 complete, all five tokenization elements and both attention modules will make immediate sense. Read the lineage analysis [relgt-lineage](relgt-lineage.md) alongside.

---

## Optional Deepening

- **[rossi2020tgn](rossi2020tgn.md)** — TGN: memory module + temporal graph attention; deeper background on continuous-time temporal graphs before RelGT's temporal encoding.
- **[yu2023dygformer](yu2023dygformer.md)** — DyGFormer: Transformer for temporal graphs; most recent SOTA before RelGT; good contrast with RelGT's unified approach.
- **[muller2023attending](muller2023attending.md)** — Survey + taxonomy of ~40 GTs; PE choice dominates; confirms no prior GT handles REGs. Good landscape map.
- **[ying2021graphormer](ying2021graphormer.md)** — Full global attention via SPD bias + centrality encoding; good contrast with GPS's hybrid approach.
- **[kreuzer2021san](kreuzer2021san.md)** — Full Laplacian spectrum as LPE; motivates GraphGPS's shift away from LapPE toward RWSE.
- **[dwivedi2023largegraphs](dwivedi2023largegraphs.md)** — Survey of scalability strategies for large graphs; contextualizes RelGT's EMA centroid global module.

---

## Estimated Time

| Stage | Papers | Est. reading time |
|---|---|---|
| 1: Foundations | 2 | 3–4 hrs |
| 2: PE arc | 2 | 3–4 hrs |
| 3: Scalable GTs | 2 | 3–4 hrs |
| 4: Heterogeneous & Temporal | 2 | 2–3 hrs |
| 5: RelGT | 1 | 2–3 hrs |
| **Total** | **9** | **~13–18 hrs** |
