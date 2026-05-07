---
title: "R-GCN: Modeling Relational Data with Graph Convolutional Networks"
tags: [source, gnn, heterogeneous-graph, knowledge-graph, relational]
sources: [schlichtkrull2018rgcn]
updated: 2026-05-07
---

# R-GCN: Modeling Relational Data with Graph Convolutional Networks

**Source:** https://arxiv.org/abs/1703.06103
**Title:** Modeling Relational Data with Graph Convolutional Networks
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Schlichtkrull, Kipf, Bloem, van den Berg, Titov, Welling
**Venue:** ESWC 2018

## Summary

- **What:** Standard GCN uses a single shared weight matrix for all edges, ignoring that edges in knowledge graphs have different semantic relation types (*born_in*, *works_at*, *married_to*).
- **How:** Assign relation-type-specific weight matrices $W_r^l$ with basis or block-diagonal decomposition to control parameter growth; combine with DistMult decoder for link prediction.
- **So what:** First GCN for multi-relational graphs; achieves SOTA on entity classification and link prediction on FB15k-237 and WN18; foundational precursor to HGT's meta-relation parameterization.

## Challenges & Novelty

Knowledge graphs and relational databases have edge types that encode fundamentally different semantics. Applying GCN uniformly across all edge types blends incompatible signals — a *born_in* edge and a *works_at* edge should not be aggregated with the same matrix. With potentially thousands of relation types, simply assigning one matrix per relation leads to quadratic parameter growth and poor generalization on rare relations.

- **Relation-specific aggregation:** separate $W_r^l$ matrices per relation type gives each relation type its own message, allowing the model to distinguish *born_in* from *works_at* at the representation level.
- **Basis decomposition regularizes rare relations:** $W_r^l = \sum_{b=1}^B a_{rb}^l V_b^l$ — all relations share a small set of basis matrices $V_b$; only the scalar coefficients $a_{rb}$ are relation-specific. Rare relations borrow capacity from frequent relations via shared bases.
- **Block-diagonal decomposition enforces sparsity:** $W_r^l = \text{diag}(Q_{1r}^l, \ldots, Q_{Br}^l)$ — each relation operates on independent sub-spaces of the embedding, preventing cross-feature contamination across relations.

## Relation to Prior Work

| Model | Edge types | Aggregation | Parameter sharing | Task |
|---|---|---|---|---|
| [kipf2017gcn](kipf2017gcn.md) | No | Normalized mean | Shared $W$ | Node classification |
| **R-GCN** | Yes (by $W_r$) | Per-relation mean | Basis / block-diag | Entity class. + link pred. |
| [hu2020hgt](hu2020hgt.md) | Yes (+ node types) | Attention per meta-relation | Type-specific $K$/$Q$/$V$ | Node classification |
| [chen2025relgnn](chen2025relgnn.md) | Yes (bridge/hub routes) | Route-specific + FUSE | Shared route weights | RDL forecasting |

- [kipf2017gcn](kipf2017gcn.md): R-GCN replaces GCN's single $W$ with $\{W_r\}$ — adding relation specificity while keeping the same normalized aggregation structure.
- [hu2020hgt](hu2020hgt.md): HGT extends R-GCN's relation typing to triplets (src_type, edge_type, dst_type) and replaces uniform aggregation with attention, giving finer-grained heterogeneous message passing.
- [chen2025relgnn](chen2025relgnn.md): RelGNN's composite routes on PK-FK graphs can be seen as a successor to R-GCN's relation-specific aggregation, specialized for the bridge/hub topology of relational databases.
- [fey2024rdlposition](fey2024rdlposition.md): RDL's blueprint identifies R-GCN as a key ancestor; HeteroGraphSAGE (the RelBench baseline) is an inductive extension of the same multi-relational idea.

## Technical Details

**R-GCN layer update:**

$$\mathbf{h}_i^{(l+1)} = \sigma\!\left(\sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)} \frac{1}{c_{i,r}} W_r^{(l)} \mathbf{h}_j^{(l)} + W_0^{(l)} \mathbf{h}_i^{(l)}\right)$$

where $\mathcal{N}_r(i)$ is the set of neighbors of $i$ under relation $r$, $c_{i,r} = |\mathcal{N}_r(i)|$ normalizes by relation-specific degree, and $W_0^{(l)}\mathbf{h}_i^{(l)}$ is a self-loop term.

**Basis decomposition:**

$$W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)} V_b^{(l)}$$

Parameters: $B$ shared basis matrices $V_b$ + per-relation scalar coefficients $a_{rb}$. When $B \ll |\mathcal{R}|$, this drastically reduces parameter count and forces sharing across relations.

**Block-diagonal decomposition:**

$$W_r^{(l)} = \text{diag}\!\left(Q_{1r}^{(l)}, \ldots, Q_{Br}^{(l)}\right)$$

Each relation operates in $B$ independent sub-spaces of size $d/B$. More efficient than basis decomposition when relations are diverse.

**Encoder-decoder for link prediction.** R-GCN as encoder generates entity embeddings $\{\mathbf{h}_i\}$; DistMult as decoder scores triples: $f(s, r, o) = \mathbf{h}_s^T \text{diag}(w_r) \mathbf{h}_o$. Trained with cross-entropy on negative-sampled triples.

## Experiments

- Entity classification on FB15k-237: R-GCN outperforms Feat (raw features) and DeepWalk by 4–6% accuracy; basis decomposition consistently outperforms block-diagonal.
- Link prediction: R-GCN+DistMult achieves 29.8% MRR on FB15k-237, a 29.8% relative improvement over DistMult alone (decoder-only baseline).
- On WN18: similar gains; R-GCN encoder adds largest benefit when entities have few labeled training triples (sparse-relation regime).
- Basis decomposition with $B=2$ competitive with $B=100$; increasing $B$ beyond 10 shows diminishing returns.

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md)
- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md)
- [relational-entity-graph](relational-entity-graph.md)
