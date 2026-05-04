---
title: "Exphormer: Sparse Transformers for Graphs"
tags: [source, graph-transformer, scalability]
sources: [shirzad2023exphormer]
updated: 2026-05-04
---

# Exphormer: Sparse Transformers for Graphs

**Source:** https://arxiv.org/abs/2303.06147
**Title:** Exphormer: Sparse Transformers for Graphs
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Hamed Shirzad, Ameya Velingker, Balaji Venkatachalam, Danica J. Sutherland, Ali Kemal Sinop
**Venue:** ICML 2023

## Summary

Exphormer builds scalable Graph Transformers via a principled **sparse attention mechanism** that replaces quadratic all-pairs attention with two complementary components: **(1) virtual global nodes** (a small set of nodes that attend to all other nodes, providing global context at O(N) cost), and **(2) expander graph attention** (sparse attention following edges drawn from an expander graph, whose spectral expansion properties guarantee good information mixing). Together, these give O(N) complexity while maintaining strong expressive power — expander graphs have near-optimal mixing properties by construction.

Exphormer serves as a drop-in replacement for the global attention module in GraphGPS (replacing the Performer/BigBird approximation), achieving better accuracy and O(N+E) total complexity. It scales to graphs with millions of nodes while remaining competitive with full-attention methods on standard benchmarks.

## Key Takeaways

- **Expander graph attention**: sparse connections with provably good spectral mixing; not arbitrary sparse connections but mathematically motivated by expansion properties.
- **Virtual global nodes**: each attends to all real nodes — provides global context at O(N) cost.
- Replaces full $O(N^2)$ attention with $O(N+E)$ without significant accuracy loss.
- Drop-in for GraphGPS global attention; improves over Performer/BigBird on accuracy.
- Demonstrates that carefully chosen sparse connections can match dense attention for graph tasks.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [positional-encoding](positional-encoding.md)

## Relation to Other Wiki Pages

- [graphgps](graphgps.md): Exphormer replaces the global attention in GraphGPS, demonstrating that its Performer/BigBird approximation can be bettered with expander-based sparsity.
- [graph-transformer](graph-transformer.md): Exphormer is one of several scalability strategies (sparse attention approach, complementing hierarchical and sampling approaches).
- [alon2020bottleneck](alon2020bottleneck.md): Exphormer's global nodes directly address over-squashing by providing a shortcut for long-range signals.
