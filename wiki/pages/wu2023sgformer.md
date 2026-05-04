---
title: "SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations"
tags: [source, graph-transformer, scalability]
sources: [wu2023sgformer]
updated: 2026-05-04
---

# SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations

**Source:** https://arxiv.org/abs/2306.10759
**Title:** SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Qitian Wu, Wentao Zhao, Chenxiao Yang, Hengrui Zhang, Fan Nie, Haitian Jiang, Yatao Bian, Junchi Yan
**Venue:** NeurIPS 2023

## Summary

SGFormer takes a deliberately minimalist approach to scaling Graph Transformers: it shows that even a **single-layer global attention module** combined with one-hop message passing is sufficient to achieve state-of-the-art performance on large-scale graph benchmarks, scaling to the web-scale ogbn-papers100M (111M nodes). The key insight is that deep multi-head attention stacks, as used in standard Transformers, are unnecessary for graph tasks when the local graph structure already provides relevant inductive biases.

SGFormer's architecture has one global attention layer (handling all-pairs long-range dependencies) and one GNN layer (handling local structure). The global attention uses a simplified kernel-based formulation for linear complexity: $O(N)$ instead of $O(N^2)$. This yields **up to 141× inference acceleration** over state-of-the-art Transformers on medium-sized graphs.

The main theoretical argument is that graph structure imposes strong relational constraints that reduce the effective information complexity — so one layer of global attention can capture what multiple layers of NLP-style attention cannot, because the graph edges guide where to aggregate.

## Key Takeaways

- **Single global attention layer + one GNN layer** sufficient for SOTA on large graphs — challenges the deep-stack paradigm.
- Linear $O(N)$ complexity via kernel attention approximation; 141× faster inference than SOTA GTs.
- Scales to ogbn-papers100M (111M nodes) — an order of magnitude larger than prior scalable GTs.
- Validates "simplify first" philosophy: more layers ≠ better for graph tasks with strong structural priors.
- Complements Exphormer (sparse attention) and NAGphormer (hop tokenization) as a third scalability strategy.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)

## Relation to Other Wiki Pages

- [graph-transformer](graph-transformer.md): extreme simplification end of the scalability spectrum — minimum architecture, maximum scale.
- [alon2020bottleneck](alon2020bottleneck.md): the single global attention layer directly addresses over-squashing with minimal architectural overhead.
- [shirzad2023exphormer](shirzad2023exphormer.md): Exphormer uses structured sparsity; SGFormer uses a single dense global layer — two complementary approaches to O(N) GT attention.
- [graphgps](graphgps.md): GPS uses MPNN + full global attention per layer (multiple layers); SGFormer suggests one layer is enough.
