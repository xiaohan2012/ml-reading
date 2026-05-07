---
title: "GraphGPS vs Gophormer vs NAGphormer: Three Scalability Strategies"
tags: [query, analysis, graph-transformer, scalability]
sources: [rampavsek2022graphgps, zhao2021gophormer, chen2022nagphormer]
updated: 2026-05-06
---

# GraphGPS vs Gophormer vs NAGphormer: Three Scalability Strategies

## Common Ground

All three papers solve the same root problem: standard Graph Transformers require O(N²) attention and can't scale to large graphs. All three cite the same baseline GT papers as prior work — Dwivedi & Bresson's GT ([dwivedi2021gt](dwivedi2021gt.md)), SAN ([kreuzer2021san](kreuzer2021san.md)), and Graphormer ([ying2021graphormer](ying2021graphormer.md)).

## Direct Citation Chain

**NAGphormer explicitly critiques Gophormer** in its related work section:

> "Recent work [Gophormer] samples several ego-graphs... However, the sampling process is also time-consuming... Moreover, the sampled ego-graphs only contain limited neighborhood information due to the fixed and small graph size."

NAGphormer's Hop2Token is a direct counter-proposal: instead of sampling a subgraph (Gophormer's approach), pre-aggregate multi-hop features and attend *between hops*, not between nodes.

**GraphGPS does not cite Gophormer or NAGphormer** — published May 2022, after Gophormer (Oct 2021) but before NAGphormer's ICLR 2023 appearance. It pursues a third path: linear attention (Performer/BigBird) in the global stream rather than restricting attention to subgraphs or hops.

## Three Distinct Strategies

| Paper | Strategy | Attention Scope | Complexity |
|---|---|---|---|
| [rampavsek2022graphgps](rampavsek2022graphgps.md) | Linear attention (Performer/BigBird) | Global, all nodes | O(N+E) via kernel |
| [zhao2021gophormer](zhao2021gophormer.md) | Ego-graph sampling | Local subgraph per node | O(K²) per node |
| [chen2022nagphormer](chen2022nagphormer.md) | Hop tokenization | K hops within a node | O(K²) per node; no graph access at train time |

## Relevance to RelGT

- RelGT's local K=300 neighborhood attention is closest to Gophormer's ego-graph design
- NAGphormer's hop tokenization noted as a possible future direction for REGs
- GraphGPS is RelGT's direct backbone (MPNN ∥ GlobalAttn structure)
