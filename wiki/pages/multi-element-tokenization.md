---
title: Multi-Element Tokenization
tags: [concept, tokenization, graph-transformer, relational-deep-learning]
sources: [relational-graph-transformer]
updated: 2026-04-28
---

# Multi-Element Tokenization

## Description

Multi-element tokenization is [RelGT](relational-graph-transformer.md)'s approach to converting nodes in a [Relational Entity Graph](relational-entity-graph.md) into tokens for a Transformer. Standard GTs use a 2-element representation (node features + a single positional encoding). RelGT instead uses a **5-element token** that explicitly decouples different properties of relational data:

| Element | What it captures | Encoder |
|---|---|---|
| Node features $x_{v_j}$ | Entity attributes (multimodal) | PyTorch Frame multimodal encoder |
| Node type $\phi(v_j)$ | Table-based heterogeneity | Learned embedding via one-hot |
| Hop distance $p(v_i, v_j)$ | Structural proximity to seed node | Learned embedding via one-hot |
| Relative time $\tau(v_j) - \tau(v_i)$ | Temporal dynamics | Linear projection |
| Subgraph GNN PE | Local graph structure (cycles, parent-child) | [Subgraph GNN PE](subgraph-gnn-pe.md) |

The five encodings are each projected to $\mathbb{R}^d$, concatenated to $\mathbb{R}^{5d}$, then mixed by a learned linear map $O \in \mathbb{R}^{5d \times d}$.

**Key advantage**: no global precomputation — all encodings operate on the sampled local subgraph of K=300 nodes, making complexity independent of the full REG size.

**Ablation findings** (RelBench, 8 tasks):
- Removing subgraph GNN PE: −5.95% average (most critical)
- Removing relative time: −9.91% average (large penalty on `rel-hm item-sales`)
- Removing node type: −1.75% average
- Removing hop distance: −2.19% average
- Removing global module: −3.87% average (task-dependent)

## Appearances in Sources

- [relational-graph-transformer](relational-graph-transformer.md) — core contribution; described in Section 3.1

## Related Concepts

- [subgraph-gnn-pe](subgraph-gnn-pe.md) — the most critical of the five elements
- [graph-transformer](graph-transformer.md) — the architecture this tokenization feeds into
- [relational-entity-graph](relational-entity-graph.md) — the input structure being tokenized
