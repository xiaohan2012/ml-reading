---
title: Graph Transformer (GT)
tags: [concept, graph-transformer, gnn, attention]
sources: [relational-graph-transformer, relational-transformer, rampavsek2022graphgps, dwivedi2021graph, graphormer, san, attention-graphs-gt-interpretability]
updated: 2026-04-30
---

# Graph Transformer (GT)

## Description

Graph Transformers extend the self-attention mechanism from sequence models to graph-structured data. They offer more expressive alternatives to standard message-passing GNNs by enabling longer-range information flow and richer structural encodings via positional/structural encodings (PEs/SEs).

Key design axes in the literature:
- **Local vs. global attention**: early GTs restricted attention to local neighborhoods (acting like attention-based GNNs); later models (Graphormer, GraphGPS) use full global attention or hybrid local+global designs
- **Positional encodings**: Laplacian eigenvectors, random-walk PEs, node2vec — inject topology but are expensive to precompute and don't generalize to heterogeneous/temporal/large graphs
- **Scalability strategies**: hierarchical clustering ([zhang2022hierarchical](zhang2022hierarchical.md), [zhu2023hierarchical](zhu2023hierarchical.md)), sparse attention ([shirzad2023exphormer](shirzad2023exphormer.md)), neighborhood sampling ([zhao2021gophormer](zhao2021gophormer.md), [chen2022nagphormer](chen2022nagphormer.md)), single-layer global ([wu2023sgformer](wu2023sgformer.md))
- **Tokenization granularity**: row-level (most GTs, including RelGT) vs. cell-level ([Relational Transformer](relational-transformer.md))

**Limitations on relational data**: most GT designs assume static, homogeneous, small-scale graphs. Applying them to [Relational Entity Graphs](relational-entity-graph.md) fails because: (i) PEs don't generalize to heterogeneous/temporal graphs, (ii) precomputation is prohibitive at scale, (iii) temporal dynamics and schema constraints are ignored.

**Two contrasting approaches to relational GTs:**
- [RelGT](relational-graph-transformer.md): row-level tokens, rich 5-element positional encodings, schema-specific, best supervised performance.
- [Relational Transformer (RT)](relational-transformer.md): cell-level tokens, no positional encodings, schema-agnostic, enables zero-shot transfer. RT's neighbor attention is equivalent to GNN message passing.

Notable GT models:
- **[TokenGT](kim2022pure.md)** (Kim et al., NeurIPS 2022): pure Transformer — nodes+edges as tokens with orthonormal node identifiers + type identifiers; provably ≥ 2-IGN (> all MPNNs); compatible with linear attention
- **[Graph Transformer](dwivedi2021graph.md)** (Dwivedi & Bresson, AAAI 2021): foundational formalization; sparse neighborhood attention + LapPE + BatchNorm; establishes LapPE as canonical graph PE (LapPE itself introduced by [dwivedi2020benchmarking](dwivedi2020benchmarking.md))
- **[GraphGPS](rampavsek2022graphgps.md)** (Rampášek et al., NeurIPS 2022): hybrid local-global recipe; GPS layer = MPNN ∥ GlobalAttn; O(N+E) with Performer/BigBird; SOTA on 11/16 benchmarks; foundational PE/SE taxonomy
- **HGT** (Hu et al.): designed for heterogeneous graphs but underperforms HeteroGNN on RelBench; high overhead with Laplacian PE
- **[Graphormer](graphormer.md)** (Ying et al., NeurIPS 2021): full global attention with SPD-based spatial encoding + centrality + edge encoding; SOTA on OGB-LSC; subsumes GCN/GIN/GraphSAGE as special cases
- **[SAN](san.md)** (Kreuzer et al., NeurIPS 2021): full Laplacian spectrum as LPE; provably exceeds 1-WL; first fully-connected model competitive on graph benchmarks

## Appearances in Sources

- [relational-graph-transformer](relational-graph-transformer.md) — surveys GT limitations for REGs; RelGT adapts row-level GT design to the RDL setting
- [relational-transformer](relational-transformer.md) — RT contrasts with row-level GTs; cell-level tokenization as an alternative for foundation model pretraining
- [rampavsek2022graphgps](rampavsek2022graphgps.md) — foundational GPS recipe: MPNN ∥ GlobalAttn per layer; PE/SE taxonomy; O(N+E) with linear attention; SOTA on 11/16 benchmarks
- [dwivedi2021graph](dwivedi2021graph.md) — Dwivedi & Bresson (AAAI 2021): first GT formalization; sparse neighborhood attention + LapPE + BatchNorm; edge features × attention scores
- [graphormer](graphormer.md) — Ying et al. (NeurIPS 2021): full attention with 3 structural encodings (centrality, spatial SPD, edge path); SOTA on OGB-LSC; covers GCN/GIN/GraphSAGE as special cases
- [san](san.md) — Kreuzer et al. (NeurIPS 2021): full Laplacian spectrum as LPE; fully-connected attention; first to exceed 1-WL provably
- [attention-graphs-gt-interpretability](attention-graphs-gt-interpretability.md) — El et al. (2025): Attention Graphs framework; reveals that DL/DLB models achieve same accuracy via divergent internal strategies; DL models <4% structure recovery

## Related Concepts

- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md) — GT variant for heterogeneous graphs (HGT)
- [multi-element-tokenization](multi-element-tokenization.md) — RelGT's row-level tokenization scheme
- [relational-attention](relational-attention.md) — RT's cell-level attention mechanism; neighbor attention ≈ GNN message passing
- [subgraph-gnn-pe](subgraph-gnn-pe.md) — RelGT's positional encoding strategy
- [relational-entity-graph](relational-entity-graph.md) — the graph type GTs must handle in RDL
