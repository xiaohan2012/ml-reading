---
title: "GraphGPS: Recipe for a General, Powerful, Scalable Graph Transformer"
tags: [source, graph-transformer, positional-encoding, gnn]
sources: [rampavsek2022graphgps]
updated: 2026-04-29
---

# GraphGPS: Recipe for a General, Powerful, Scalable Graph Transformer

**Source:** https://arxiv.org/abs/2205.12454
**Title:** Recipe for a General, Powerful, Scalable Graph Transformer
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Rampášek, Galkin, Dwivedi, Lim, Wolf, Beaini
**Venue:** NeurIPS 2022
**Year:** 2022

## Summary

The paper introduces GraphGPS — a modular recipe for building general, powerful, and scalable (GPS) graph Transformers. The core insight: existing Graph Transformers lack a common framework for positional and structural encodings, and suffer from O(N²) complexity that limits them to small graphs. GraphGPS addresses both with a principled 3-ingredient recipe.

![[Pasted image 20260506141717.png]]

- The the green block depicts the GPS layer (explained below)
- There are two streams of features for nodes and edges, respectively (the orange and gray blocks). They form the input to GPS layers

**GPS Layer.**  Each GPS layer combines a local message-passing module and a global attention module in parallel, summing their outputs before an MLP:

$$\mathbf{X}^{\ell+1}_M, \mathbf{E}^{\ell+1} = \texttt{MPNN}(\mathbf{X}^\ell, \mathbf{E}^\ell, \mathbf{A})$$
$$\mathbf{X}^{\ell+1}_T = \texttt{GlobalAttn}(\mathbf{X}^\ell)$$
$$\mathbf{X}^{\ell+1} = \texttt{MLP}(\mathbf{X}^{\ell+1}_M + \mathbf{X}^{\ell+1}_T)$$

Edge features go only to the MPNN (not to global attention), which allows using linear-complexity Transformers (Performer, BigBird) for the global stream — yielding overall O(N+E) complexity. The MPNN implicitly encodes edge information into node representations so global attention can exploit it indirectly.

**PE/SE Taxonomy.** GraphGPS provides the first systematic categorization of positional encodings (PE) and structural encodings (SE) into local/global/relative levels. PE captures position in the graph (close nodes → similar PE); SE captures structural similarity (similar subgraphs → similar SE). They are complementary:
- *Global PE*: Laplacian eigenvectors (LapPE), SignNet — inject global node identifiability
- *Local SE*: Random-walk diagonal (RWSE) — captures local sub-structure membership (e.g., odd-length cycles)
- *Relative PE*: pairwise distances as edge features — used in attention bias

**Ablation findings:**
- MPNN is essential: removing it causes catastrophic performance drops across all datasets; without it, edge features are ignored and the model overfits to PE/SE
- Adding any global attention consistently helps, with minor exceptions (ZINC, which only requires local structure counting)
- RWSE is the most robust local SE; SignNet+DeepSets is the single best encoding but at higher compute cost
- Performer is within ~1% of full Transformer while scaling to thousands of nodes

**Benchmarking.** GraphGPS achieves SOTA on 8/16 and top-3 on 11/16 diverse benchmarks: ZINC, MNIST, CIFAR10, PATTERN, CLUSTER (Benchmarking-GNNs); ogbg-molhiv, molpcba, ppa, code2 (OGB); PCQM4Mv2 (OGB-LSC, beats Graphormer with <50% parameters); MalNet-Tiny (5,000-node graphs); PascalVOC-SP, COCO-SP, Peptides-func, Peptides-struct, PCQM-Contact (LRGB).

## Key Takeaways

- **GPS layer = MPNN ∥ GlobalAttn**: parallel streams (not sequential) prevent early over-smoothing from the MPNN stage; summation preserves both local and global information at every layer.
- **Edge features only go to MPNN**: this is what enables linear Transformers — no full attention matrix, no edge-attention bias; edges still influence global attention implicitly via node representations.
- **MPNN is non-negotiable**: the single biggest ablation finding; pure Transformer degrades severely on every dataset — local neighborhood information is critical.
- **PE and SE are complementary**: PE disambiguates node positions (needed for WL-hard graph pairs); SE captures substructure membership (needed for molecular tasks); RWSE is the most reliable practical choice.
- **RWSE over LapPE for molecular data**: RWSE is cheaper (no eigenvector decomposition) and more robust; LapPE helps more for image superpixel tasks; SignNet is best when compute budget allows.
- **O(N+E) scaling via Performer/BigBird**: first GT to handle MalNet-Tiny graphs (5k nodes) and compete on OGB-LSC (3.8M molecules), far beyond prior GTs limited to ~500-node graphs.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [positional-encoding](positional-encoding.md)

## Relation to Other Wiki Pages

- [graph-transformer](graph-transformer.md): GraphGPS is the foundational blueprint for hybrid MPNN+GT architectures; establishes the PE/SE taxonomy that later work (RelGT, etc.) builds on.
- [relational-graph-transformer](relational-graph-transformer.md): RelGT explicitly builds on GraphGPS for the relational setting — its 5-element tokenization extends the GPS PE/SE framework to heterogeneous temporal graphs; however RelGT uses schema-specific encodings (hop distance, type, time) rather than graph-global Laplacian/RWSE.
- [heterogeneous-graph-transformer](heterogeneous-graph-transformer.md): HGT is a competing GT for heterogeneous graphs; GraphGPS demonstrates that hybrid MPNN+attention is generally superior to pure-attention architectures on most benchmarks.
- [subgraph-gnn-pe](subgraph-gnn-pe.md): RelGT's subgraph GNN PE is the RDL analog of GPS's local SE (RWSE) — both capture subgraph structural information rather than global graph PEs.
