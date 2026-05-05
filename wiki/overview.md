---
title: Overview
tags: [overview, synthesis]
sources: [relational-deep-learning-position, relbench-v1, relational-graph-transformer, tgm-temporal-graph-modelling, relational-transformer, relbench-v2, relgnn, griffin, rampavsek2022graphgps, gcn, hamilton2017inductive, gat, gin, tgat, tgn, dygformer, tabpfnv2, tabicl, carte, graphormer, san, mpnn, rgcn, k-gnn, tgb, tabpfnv1, saint, cvitkovic2020, dyrep, graphmixer]
updated: 2026-04-29
---

# ML Reading and Research (GNN & Interpretability) — Overview

> Evolving synthesis of everything in the wiki. Updated by wiki-ingest when sources shift the understanding.

## Current Understanding

The wiki covers Relational Deep Learning (RDL) from its founding papers through the current frontier. The field was established by Fey et al. (ICML 2024) with a clear blueprint: represent any relational database as a [Relational Entity Graph](pages/relational-entity-graph.md), specify tasks via [Training Tables](pages/training-table.md), and apply heterogeneous temporal GNNs that learn to perform SQL `JOIN`+`AGGREGATE` end-to-end. Robinson et al. (NeurIPS 2024) empirically validated this by showing RDL matches or outperforms a skilled data scientist on all 30 RelBench v1 tasks with 96% fewer human hours.

The central question across later sources is: *how do you design architectures that exploit relational structure without expensive global precomputation, while generalizing across diverse schemas?*

Three contrasting answers have emerged:
- **Schema-specific + rich encodings** ([RelGT](pages/relational-graph-transformer.md)): row-level tokenization with 5-element positional encodings (type, hop, time, subgraph GNN PE); best supervised performance on RelBench.
- **Schema-agnostic + no encodings** ([Relational Transformer](pages/relational-transformer.md)): cell-level tokenization with Relational Attention masks; enables zero-shot transfer across databases at 22M params outperforming 27B LLMs.
- **Fixed GNN topology** ([RelGNN](pages/relgnn.md)): identifies bridge/hub structural patterns specific to PK-FK graphs; composite message passing (FUSE operation) replaces 2-hop aggregation; achieves SOTA on 27/30 RelBench v1 tasks without foundation model pretraining.

Additionally, [Griffin](pages/griffin.md) (Wang et al. 2025, ICML) is the first explicit attempt at a pretrained foundation model for diverse RDBs — combining unified text+float encoders with cross-attention over row cells and GNN message passing across tables. Pretraining on 150M+ nodes helps most in low-data regimes; gap narrows with more task data, consistent with general FM behavior.

**Tabular foundation models.** Three distinct approaches compete: TabPFN v2 (Nature 2025, ~130M synthetic datasets, alternating row/col attention, best small-table ICL but capped at 10K samples); TabICL (ICML 2025, three-Transformer decoupled architecture, scales to 500K samples, outperforms TabPFNv2 on large data); and CARTE (ICML 2024, graph-per-row pretraining on YAGO knowledge base, no schema matching, best for string-heavy tables). All three challenge gradient-boosted tree dominance on tabular data, but each targets a different regime.

**Temporal graph learning.** The CTDG literature spans four generations: DyRep (ICLR 2019, two-process temporal point process — Association for topology-changing events, Communication for transient interactions — with recurrent+structural+temporal embedding update); TGAT (ICLR 2020, Bochner time encoding + temporal graph attention, no memory); TGN (2020, adds a per-node GRU memory that resolves TGAT's staleness problem — 1 layer with memory beats 2 layers without, 30× faster; explicitly subsumes DyRep as a special case); and DyGFormer (NeurIPS 2023, eliminates multi-hop aggregation in favor of long first-hop sequences fed into a Transformer via patching + neighbor co-occurrence encoding). GraphMixer (ICLR 2023) challenges all of these: a pure MLP-Mixer over sorted temporal links + mean-pooled neighbors matches or beats TGAT/TGN/DyGFormer with 2–5× faster training, suggesting architectural complexity may be unnecessary for link prediction.

**Foundational GNN layer.** The field rests on four canonical message-passing architectures: GCN (spectral, transductive, isotropic mean), GraphSAGE (inductive via neighborhood sampling, three aggregator variants), GAT (learned per-neighbor attention weights, inductive), and GIN (sum+MLP, provably WL-level expressive). GIN's theoretical result (ICLR 2019) establishes that any aggregation-based GNN is at most as powerful as the WL isomorphism test, and that sum aggregation achieves this bound while mean/max do not. Gilmer et al.'s MPNN framework (ICML 2017) unifies GCN/GAT/GIN/GraphSAGE and 5+ other models under a single message-passing formulation with edge features. R-GCN (ESWC 2018) extends GCN to multi-relational data via relation-specific weight matrices — a direct precursor to HGT. k-GNN (AAAI 2019) proves 1-GNNs ≡ 1-WL and proposes tuple-level message passing to exceed the 1-WL bound.

**Graph Transformer landscape.** Beyond GraphGPS (MPNN ∥ GlobalAttn), two contrasting GT designs address structural encoding: Graphormer (NeurIPS 2021) encodes graph structure as attention biases — shortest-path distance, centrality, and edge path features — and subsumes GCN/GIN/GraphSAGE as special cases; SAN (NeurIPS 2021) uses the full Laplacian spectrum as a learned positional encoding and proves provable beyond-1-WL expressiveness via fully-connected attention.

**Temporal graph benchmarking.** TGB (NeurIPS 2023) provides large-scale (4M–67M edges) standardized evaluation for temporal graph models with realistic negative sampling. Its key finding — that simple baselines often match or outperform TGAT/TGN/DyGFormer on node prediction — calibrates expectations for the field's architectural complexity.

**Tabular learning: deeper history.** TabPFN v1 (ICLR 2023) established the Prior-Data Fitted Network paradigm: a Transformer pretrained on SCM+BNN synthetic data approximates Bayesian posterior prediction for small tabular datasets (N≤1000) in a single forward pass. SAINT (2021) introduced intersample (row-level) attention across the batch alongside standard column-level attention, beating boosted trees on average — the architectural insight that TabICL scales up. Cvitkovic (2020) first proposed treating relational database supervised learning as GNN node classification via RDB-as-graph, directly anticipating the RDL blueprint's formalization.

The benchmark ([RelBench](pages/relbench.md), now at v2) is growing in scale (7→11 datasets, 22M+ rows) and task diversity (forecasting + autocomplete) to support both supervised evaluation and foundation model pretraining. The [TGM library](pages/tgm-temporal-graph-modelling.md) provides parallel infrastructure for temporal graph learning (CTDG + DTDG unified), with TGB datasets now bridging into the RelBench ecosystem.

RDL consistently outperforms single-table baselines (LightGBM) across all task types, confirming that relational structure carries informative signal.

## Open Questions

- How do RelGNN's atomic routes interact with Griffin's cross-attention within-row encoding — could composite message passing be applied on top of Griffin's hierarchical aggregation?
- How does RelGT perform on RelBench v2's new datasets and autocomplete tasks?
- Can RT's zero-shot approach match RelGT's supervised performance after fine-tuning at scale?
- What makes the global attention module in RelGT helpful for some tasks and harmful for others?
- Would TGM's hook system generalize to the RDL pipeline (temporal-aware subgraph sampling as a hook)?
- How does snapshot time granularity (TGM finding: daily > hourly for GCN on Wikipedia) interact with RDL's temporal neighbor sampling window?
- The RDL blueprint paper flagged fine-tuning vs. freezing multimodal column encoders as an open question — how much does end-to-end encoder training improve RelBench results?
- The blueprint calls for self-supervised pretraining via automatically mined training tables — how does this compare to RT's masked token prediction objective?

## Key Entities / Concepts

- [Tabular Learning](pages/tabular-learning.md) — ICL/PFN paradigm; TabPFNv1→v2/TabICL/CARTE/SAINT; GBDT vs. deep learning competition
- [Temporal Graph](pages/temporal-graph.md) — CTDG/DTDG; TGAT/TGN/DyGFormer; TGB benchmark calibration
- [Graph Neural Network](pages/graph-neural-network.md) — GCN/GraphSAGE/GAT/GIN/MPNN/R-GCN/k-GNN; message-passing family; WL expressiveness hierarchy
- [GraphGPS](pages/rampavsek2022graphgps.md) — foundational GT recipe: MPNN ∥ GlobalAttn; PE/SE taxonomy; O(N+E) complexity
- [Relational Deep Learning](pages/relational-deep-learning.md) — the paradigm; from GNNs to GTs to foundation models
- [Training Table](pages/training-table.md) — (EntityID, seed\_time, label); task specification + temporal leakage prevention in one abstraction
- [Relational Entity Graph](pages/relational-entity-graph.md) — the graph abstraction of a relational database
- [Graph Transformer](pages/graph-transformer.md) — key architecture family; row-level vs. cell-level tokenization tradeoff
- [Multi-Element Tokenization](pages/multi-element-tokenization.md) — RelGT's 5-element schema-specific token decomposition
- [Subgraph GNN PE](pages/subgraph-gnn-pe.md) — most critical RelGT component; random resampled init beats Laplacian PE
- [Relational Attention](pages/relational-attention.md) — RT's schema-agnostic attention over column/row/FK links
- [Relational Foundation Model](pages/relational-foundation-model.md) — the frontier; RT achieves 93% supervised AUROC zero-shot
- [Autocomplete Tasks](pages/autocomplete-tasks.md) — new RelBench v2 task type; predict existing column values from relational context
- [Temporal Graph](pages/temporal-graph.md) — broader concept; TGM unifies CTDG+DTDG; REGs are a special case
- [RelBench](pages/relbench.md) — benchmark (v1+v2); 11 datasets, forecasting + autocomplete + recommendation
