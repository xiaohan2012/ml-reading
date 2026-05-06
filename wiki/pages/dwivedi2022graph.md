---
title: "LSPE: Graph Neural Networks with Learnable Structural and Positional Representations"
tags: [source, positional-encoding, graph-transformer, graph-neural-network]
sources: [dwivedi2022graph]
updated: 2026-05-04
---

# LSPE: Graph Neural Networks with Learnable Structural and Positional Representations

**Source:** https://arxiv.org/abs/2110.07875
**Title:** Graph Neural Networks with Learnable Structural and Positional Representations
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, Xavier Bresson
**Venue:** ICLR 2022

## Summary

LSPE proposes to **decouple structural and positional representations** in GNNs and Graph Transformers, arguing that existing approaches conflate the two into a single node vector, making it harder for the network to learn each property independently. The key idea: maintain two separate streams — one for node structural features, one for positional encodings — and allow each to evolve with its own learned dynamics through the network layers.

The architecture processes structural (feature) representations $h_v$ and positional representations $p_v$ in parallel at each layer. The positional stream is updated with its own MLP and aggregation rule, while the structural stream conditions on the current positional state. This decoupled design makes the learned PE adaptive and task-specific, in contrast to fixed LapPE precomputed before training.

Applied to both sparse GNNs (GatedGCN, PNA) and fully-connected Transformers (Transformer, SAN-GT), LSPE achieves performance improvements from 1.79% up to 64.14% across molecular benchmarks (ZINC, AQSOL, OGBL-COLLAB, MNIST, CIFAR10, PATTERN, CLUSTER). The largest gains come on tasks where structural information is most critical.

## Key Takeaways

- **Decoupled PE stream**: positional and structural representations are maintained separately and updated independently through layers — PE becomes learnable and task-adaptive, not just a static input feature.
- Applicable to both **sparse GNNs and Transformers** — a generic architectural pattern, not tied to a specific model.
- Addresses the conflation problem: injecting PE only at the input layer (as in standard LapPE) prevents the network from refining positional information.
- Empirically strongest on structure-sensitive tasks; improvements up to 64% on molecular benchmarks.
- The decoupled design directly influenced GraphGPS's architectural philosophy separating MPNN and global attention streams.

## Entities & Concepts

- [positional-encoding](positional-encoding.md)
- [graph-transformer](graph-transformer.md)
- [graph-neural-network](graph-neural-network.md)

## Relation to Other Wiki Pages

- [positional-encoding](positional-encoding.md): LSPE introduces learnable (rather than fixed) PE that evolves through layers; extends the PE concept beyond static precomputation.
- [dwivedi2020benchmarking](dwivedi2020benchmarking.md): same lead author as Benchmarking-GNNs; LSPE builds directly on LapPE introduced there.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GraphGPS incorporates the decoupled PE philosophy; LSPE's findings on PE separation are validated in GPS experiments.
- [kreuzer2021san](kreuzer2021san.md): LSPE applied to SAN's GT backbone shows the decoupled PE stream improves even spectral-based Transformers.
