---
title: "HINormer: Representation Learning on Heterogeneous Information Networks with Graph Transformer"
tags: [source, graph-transformer, heterogeneous-graph]
sources: [mao2023hinormer]
updated: 2026-05-06
---

# HINormer: Representation Learning on Heterogeneous Information Networks with Graph Transformer

**Source:** https://arxiv.org/abs/2302.11329
**Title:** HINormer: Representation Learning On Heterogeneous Information Networks with Graph Transformer
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Qiheng Mao, Zemin Liu, Chenghao Liu, Jianling Sun
**Venue:** WWW 2023

## Summary

- **What:** Message-passing HGNNs on heterogeneous information networks suffer from limited expressiveness, over-smoothing, and over-squashing due to local aggregation over type-mixed neighborhoods.
- **How:** HINormer combines a local structure encoder (type-aware message passing within a neighborhood window) with a heterogeneous relation encoder (global Transformer attention across node types), mirroring GraphGPS's local+global split for heterogeneous graphs.
- **So what:** HINormer outperforms GNN baselines on HIN node classification benchmarks (ACM, DBLP, Freebase) by capturing both local structural heterogeneity and long-range cross-type dependencies.

## Challenges & Novelty

Heterogeneous graph neural networks (HGNNs) use meta-path-based or relation-type-aware message passing, but remain subject to the fundamental limitations of local aggregation: over-smoothing (features homogenize with depth) and over-squashing (distant node information is compressed through narrow bottlenecks). A GT approach for heterogeneous graphs needs to handle incompatible type-specific feature spaces while enabling global attention.

- **Over-smoothing and over-squashing in HGNNs:** stacking many layers of type-aware message passing produces uniform representations; long-range cross-type dependencies require exponentially many hops.
- **Type incompatibility for global attention:** standard Transformers apply uniform attention across all nodes, ignoring that different node types (papers, authors, venues) have different feature distributions and semantic roles.
- **Meta-path dependency:** prior HGNNs require manual meta-path specification; HINormer learns type-specific representations without hand-crafted relational paths.

## Relation to Prior Work

| Model | Heterogeneous | Global attention | Temporal |
|---|---|---|---|
| RGCN ([schlichtkrull2018rgcn](schlichtkrull2018rgcn.md)) | Yes | No | No |
| [hu2020hgt](hu2020hgt.md) | Yes | No (local) | Yes (RTE) |
| **HINormer** | Yes | Yes | No |
| [dwivedi2025relgt](dwivedi2025relgt.md) | Yes | Yes | Yes |

- [hu2020hgt](hu2020hgt.md): HGT uses meta-relation triplet parameterization for type-specific attention but operates locally (no global attention); HINormer adds a global cross-type attention stream.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): same local+global design philosophy applied to heterogeneous graphs — GPS is the homogeneous counterpart.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT subsumes HINormer's heterogeneous GT design but additionally handles temporality, schema-defined structure, and multimodal attributes that HINormer ignores.

## Technical Details

**Local structure encoder.** Type-aware message passing within a local neighborhood window captures heterogeneous structural patterns (analogous to the MPNN component in GraphGPS). Node-type-specific projections allow features from different types to be aggregated meaningfully.

**Heterogeneous relation encoder.** A global Transformer attention module attends across all node types, capturing long-range cross-type dependencies that local message passing cannot reach. Type embeddings are injected into the attention computation to preserve type awareness in global attention.

**Two-stream combination.** The local and global outputs are combined (summed or concatenated) before an MLP, mirroring the GPS layer design:
$$h_v^{(\ell+1)} = \text{MLP}(h_v^{\text{local}} + h_v^{\text{global}})$$

## Experiments

- Outperforms GNN-based HGNNs (HAN, HGT, MAGNN) on ACM, DBLP, and Freebase node classification.
- Global relation encoder provides consistent gains over local-only baselines, especially on tasks requiring cross-type long-range reasoning.
- Evaluated on standard HIN benchmarks only; no temporal graphs or relational database tasks.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md)
