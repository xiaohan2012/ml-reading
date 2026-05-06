---
title: "GraphiT: Encoding Graph Structure in Transformers"
tags: [source, graph-transformer, positional-encoding]
sources: [mialon2021graphit]
updated: 2026-05-04
---

# GraphiT: Encoding Graph Structure in Transformers

**Source:** https://arxiv.org/abs/2106.05667
**Title:** GraphiT: Encoding Graph Structure in Transformers
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Grégoire Mialon, Dexiong Chen, Margot Selosse, Julien Mairal
**Venue:** arXiv 2021 (Inria)

## Summary

GraphiT proposes two complementary mechanisms for injecting graph structure into a Transformer, both motivated by limitations of absolute LapPE. The first is **relative positional encoding via graph kernels**: instead of adding LapPE vectors to node features, GraphiT modulates the attention scores by elementwise-multiplying with the Gram matrix of a positive-definite kernel on the graph before softmax normalization:

$$\text{PosAttention}(Q, V, K_r) = \text{normalize}\!\left(\exp\!\left(\tfrac{QQ^\top}{\sqrt{d}}\right) \odot K_r\right) V$$

The kernel $K_r$ can be a diffusion kernel ($e^{-\beta L}$, where $\beta$ controls the attention span), a random-walk kernel (visiting up to $p$-hop neighbors), or the adjacency matrix. This is a relative PE strategy: attention between nodes $i$ and $j$ depends on their structural similarity, not an absolute position — so it transfers naturally across graphs and avoids LapPE's sign-ambiguity problem.

The second mechanism is **substructure encoding via GCKN**: a Graph Convolutional Kernel Network layer pre-processes nodes by enumerating short paths and embedding them with a kernel function, producing richer node features that carry local topological information before the Transformer sees them.

GraphiT supports both local (neighborhood) and global (all-pair) attention and is competitive with GNNs on ZINC, MNIST, CIFAR10, MUTAG, PROTEINS. A further advantage is natural interpretability: attention scores reveal which pairs of nodes are mutually important for a prediction, identifying discriminative graph motifs.

## Key Takeaways

- **Relative PE via graph kernels** avoids LapPE's sign ambiguity and cross-graph transferability issues; kernel choice encodes inductive bias (diffusion: global smooth; random-walk: local $p$-hop).
- **GCKN substructure encoding** enriches node features with path-enumeration information before attention — more structural content than raw node attributes alone.
- The kernel-based relative PE is a **multiplication on attention scores** (not addition to features), making it a structurally different approach from LapPE, RWSE, and SignNet.
- Both mechanisms are complementary and can be combined; either alone already improves over GNN baselines on standard benchmarks.
- **Interpretability**: attention score maps localize discriminative substructures, useful for scientific applications.

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-transformer](graph-transformer.md)
- [dwivedi2020benchmarking](dwivedi2020benchmarking.md)

## Relation to Other Wiki Pages

- [positional-encoding](positional-encoding.md): introduces kernel-based relative PE as an alternative to LapPE; pre-dates RWSE/SignNet but shares the motivation of avoiding eigenvector instability.
- [dwivedi2020benchmarking](dwivedi2020benchmarking.md): GraphiT directly builds on and critiques LapPE from Benchmarking-GNNs; relative PE is motivated by LapPE's sign and transferability issues.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GraphGPS later generalizes this kind of relative PE into its PE/SE taxonomy; RWSE is the practical successor to kernel-based approaches.
- [kreuzer2021san](kreuzer2021san.md): SAN (same year, NeurIPS 2021) addresses LapPE issues via full Laplacian spectrum; GraphiT takes the complementary route via kernel modulation of attention.
