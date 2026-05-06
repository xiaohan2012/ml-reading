---
title: "GraphGPS PE/SE Taxonomy"
tags: [query, analysis, concept, positional-encoding, graph-transformer]
sources: [rampavsek2022graphgps, dwivedi2020benchmarking, lim2022signnet, kanatsoulis2025learning, huang2024stability, dwivedi2022graph, mialon2021graphit, graphormer]
updated: 2026-05-06
---

# GraphGPS PE/SE Taxonomy

[rampavsek2022graphgps](rampavsek2022graphgps.md) introduces the first systematic categorization of positional encodings (PE) and structural encodings (SE) for graphs. The taxonomy organizes encodings along two axes: **type** (PE vs. SE) and **scope** (local / global / relative), giving 6 cells.

---

## PE vs. SE: the core distinction

**PE** gives a notion of *position in space*: close nodes in the graph should have similar PEs. Analogy: position of a word in a text.

**SE** gives a notion of *structural similarity*: nodes embedded in similar subgraphs should have similar SEs. Analogy: grammatical role of a word regardless of position.

They are **complementary** — one can sometimes approximate the other (nearby nodes tend to have similar local structure; similar structures tend to cluster), but their objectives are distinct and they capture different graph information.

---

## The 6-cell taxonomy

### Positional Encodings (PE) — node/edge features representing *position*

| Scope | Feature type | Description | Examples |
|---|---|---|---|
| **Local PE** | node features | Position within a local cluster. Within a cluster, closer nodes → more similar PE. | Sum of non-diagonal columns of $m$-step RW matrix; distance to cluster centroid |
| **Global PE** | node features | Position within the whole graph. Closer nodes in the graph → more similar PE. | LapPE (Laplacian eigenvectors) [dwivedi2020benchmarking](dwivedi2020benchmarking.md); SignNet [lim2022signnet](lim2022signnet.md); PEARL [kanatsoulis2025learning](kanatsoulis2025learning.md); distance-matrix eigenvectors; unique component ID |
| **Relative PE** | edge features | Pairwise distance or directional relationship between two nodes. | Shortest-path distances (Graphormer [graphormer](graphormer.md)); heat kernel distances; random-walk pairwise distances [mialon2021graphit](mialon2021graphit.md); eigenvector gradient; PEG layer |

### Structural Encodings (SE) — node/edge/graph features representing *structure*

| Scope | Feature type | Description | Examples |
|---|---|---|---|
| **Local SE** | node features | Subgraph membership around a node. Nodes in similar $m$-hop subgraphs → similar SE. | Diagonal of $m$-step RW matrix (RWSE) [dwivedi2022graph](dwivedi2022graph.md) — for odd $m$, indicates membership in $m$-long cycle; node degree; triangle/ring counts; Ricci curvature |
| **Global SE** | graph features | Global structure of the entire graph. Similar graphs → similar SE. | Eigenvalues of Laplacian/adjacency [san](san.md); graph diameter, girth, #components, #nodes/#edges |
| **Relative SE** | edge features | Structural difference between two nodes. | Pairwise gradient of any local SE; boolean: do two nodes share a sub-structure? |

---

## Practical focus and empirical findings

GPS empirically evaluates only **Global PE**, **Relative PE**, and **Local SE** — the three cells with the strongest known impact. The other three (Local PE, Global SE, Relative SE) are catalogued but not ablated.

**Ablation ranking on ZINC/PCQM4Mv2/ogbg-molhiv/LRGB** (from GPS paper):

| Encoding | Type | ZINC MAE | Notes |
|---|---|---|---|
| None | — | 0.113 | baseline |
| RWSE | Local SE | **0.070** | most robust, cheapest; no eigendecomposition |
| LapPE | Global PE | 0.116 | unstable; sign ambiguity; $O(N^3)$ precompute |
| SignNet-MLP | Global PE | 0.090 | fixes sign ambiguity; moderate cost |
| SignNet-DeepSets | Global PE | **0.079** | best single encoding; highest cost |

Key finding: **RWSE is the best practical choice** for molecular graphs — cheaper and more robust than LapPE. **SignNet-DeepSets is the best absolute encoding** when compute budget allows. LapPE alone is not reliable.

---

## Why PE/SE are needed: two motivating examples from the paper

**1. Circular Skip Link (CSL) graphs** — two non-isomorphic graphs where every node has the same 1-WL color. An MPNN cannot distinguish them. A *global PE* (LapPE) assigns each node a unique position → distinction possible. A *local SE* (RWSE) captures different skip-link lengths → distinction possible via subgraph structure.

**2. Decalin molecule** — bicyclic graph where nodes $a \cong b$ and $c \cong d$ under 1-WL and local SE. Link prediction cannot distinguish $(a,d)$ from $(b,d)$. A *relative PE* (pairwise distances as edge features) or *global PE* (eigenvectors) breaks the symmetry → distinction possible.

This motivates using PE/SE at **multiple levels simultaneously** — no single level is universally sufficient.

---

## Where each encoding lives in GPS

- **Node features** ← Local PE + Global PE + Local SE (all concatenated with input features before the first GPS layer)
- **Edge features** ← Relative PE + Relative SE (used only by the MPNN stream, not global attention)
- **Graph features** ← Global SE (pooled into graph-level representation)

The modular design means any cell of the taxonomy can be swapped independently — PEARL [kanatsoulis2025learning](kanatsoulis2025learning.md) replaces LapPE/SignNet in the Global PE cell with a GNN-based linear-complexity alternative. RelGT [relational-graph-transformer](relational-graph-transformer.md) replaces the entire PE/SE framework with schema-aware encodings (node type, hop distance, temporal distance, subgraph GNN PE) suited to heterogeneous temporal REGs.

---

## Related concepts

- [positional-encoding](positional-encoding.md) — the wiki's broader PE/SE concept page (predates this taxonomy)
- [rampavsek2022graphgps](rampavsek2022graphgps.md) — source paper introducing this taxonomy
- [lim2022signnet](lim2022signnet.md) — SignNet: best Global PE in GPS ablations
- [kanatsoulis2025learning](kanatsoulis2025learning.md) — PEARL: linear-complexity replacement for Global PE cell
- [dwivedi2022graph](dwivedi2022graph.md) — LSPE: RWSE origin; Local SE
- [subgraph-gnn-pe](subgraph-gnn-pe.md) — RelGT's analog of Local SE for the RDL setting
