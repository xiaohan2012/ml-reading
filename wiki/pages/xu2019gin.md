---
title: "How Powerful are Graph Neural Networks? (GIN)"
tags: [source, gnn, expressiveness, graph-isomorphism, theory]
sources: [xu2019gin]
updated: 2026-05-07
---

# How Powerful are Graph Neural Networks? (GIN)

**Source:** https://arxiv.org/abs/1810.00826
**Title:** How Powerful are Graph Neural Networks?
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Xu, Hu, Leskovec, Jegelka
**Venue:** ICLR 2019

## Summary

- **What:** It was unclear which design choices (aggregation function, MLP depth, readout) determine how well a GNN can distinguish non-isomorphic graphs — the field lacked a formal expressiveness hierarchy.
- **How:** Prove that the Weisfeiler-Lehman (WL) graph isomorphism test is an upper bound on any aggregation-based GNN; derive GIN with injective sum aggregation + MLP as the unique architecture that achieves this bound.
- **So what:** Establishes a formal expressiveness hierarchy (GIN = WL > mean-GCN/GraphSAGE > max-GraphSAGE) and achieves SOTA on graph classification benchmarks.

## Challenges & Novelty

GCN, GraphSAGE, and GAT all aggregate neighbor features but choose different aggregation functions (normalized mean, mean, attention-weighted mean) without theoretical justification. It was not known whether these choices were equivalent in expressive power or whether any GNN could theoretically distinguish all non-isomorphic graphs.

- **WL as the tight upper bound:** any GNN that aggregates a node's features with its neighborhood via a fixed aggregation function can distinguish at most what the 1-WL color refinement test can — proven by showing that any distinguishable pair by a GNN is also distinguishable by WL, and vice versa for sufficiently powerful GNNs.
- **Mean aggregation is provably broken:** GCN and GraphSAGE use mean, which maps $\{a, a\}$ and $\{a\}$ to the same value — concrete failure case where two different multisets collapse. Max aggregation is even weaker, treating $\{a, a\}$ and $\{a\}$ identically.
- **Sum aggregation + MLP = injective = WL:** sum over a finite set of bounded-norm features is injective (Lemma 3); composing with a universal approximator (MLP with $\geq 1$ hidden layer) gives an injective aggregation. 1-layer perceptrons are insufficient — they can collapse to linear aggregation.

## Relation to Prior Work

| Model | Aggregator | Expressive power | Graph classification |
|---|---|---|---|
| [kipf2017gcn](kipf2017gcn.md) | Normalized mean | $<$ WL | Weak |
| [hamilton2017graphsage](hamilton2017graphsage.md) (mean) | Mean | $<$ WL | Weak |
| [hamilton2017graphsage](hamilton2017graphsage.md) (max) | Max | $\ll$ WL | Weaker |
| [velickovic2018gat](velickovic2018gat.md) | Attention-weighted mean | $<$ WL | Weak |
| **GIN** | Sum + MLP | $=$ WL | SOTA |

- [kipf2017gcn](kipf2017gcn.md): GIN formally proves GCN's mean aggregation cannot distinguish graphs where two nodes have the same mean neighbor embedding but different neighborhood sizes.
- [hamilton2017graphsage](hamilton2017graphsage.md): same analysis applies; GraphSAGE-max is strictly weaker than mean because it ignores multiplicities entirely.
- [velickovic2018gat](velickovic2018gat.md): attention-weighted mean is also not injective — mentioned in GIN as a direction for future analysis, since attention can still collapse two different multisets.
- [gilmer2017mpnn](gilmer2017mpnn.md): GIN's result can be restated as: among all MPNN choices of $M_t$ and $U_t$, only sum aggregation + MLP achieves WL-level power.
- [chen2025relgnn](chen2025relgnn.md): RelGNN's schema-aware route decomposition and FUSE operation move toward injective aggregation in the RDL setting, motivated by this expressiveness analysis.

## Technical Details

**GIN update rule:**

$$h_v^{(k)} = \text{MLP}^{(k)}\!\left((1 + \varepsilon^{(k)}) \cdot h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}\right)$$

where $\varepsilon$ is either learned or fixed to 0. The $(1+\varepsilon)h_v$ term ensures the central node's self-feature is not absorbed into the sum — this disambiguates the node from its neighbors.

**Why MLP, not linear?** A single linear layer $W$ is a linear map on the summed features — it cannot represent all possible injective functions on multisets. Universal approximation requires at least one hidden layer.

**Theorem (Xu et al., Thm. 3).** Let $\mathcal{A}$ be a class of GNNs where each layer aggregates neighbor features with a fixed aggregation function. $\mathcal{A}$ is at most as powerful as the WL test in distinguishing graph structures. Moreover, GINs with sum aggregation and universal approximator readout are as powerful as WL.

**Graph-level readout** (concatenation over layers captures multiscale structure):

$$h_G = \text{CONCAT}\!\left(\text{READOUT}\!\left(\{h_v^{(k)} : v \in G\}\right) \mid k = 0, 1, \ldots, K\right)$$

READOUT is sum per layer. Concatenating across layers preserves features from all neighborhood scales (local $k=1$ to global $k=K$).

**GIN-$\varepsilon$** (learnable $\varepsilon$) vs. **GIN-0** ($\varepsilon = 0$): empirically similar; the paper reports both. When node features already distinguish nodes, $\varepsilon = 0$ is sufficient since the sum already captures multiplicity.

## Experiments

- SOTA on 5/9 graph classification benchmarks at ICLR 2019: MUTAG, COLLAB, IMDB-B, IMDB-M, RDT-B — outperforming GCN, GraphSAGE, and the classical WL subtree kernel.
- GIN-$\varepsilon$ and GIN-0 comparable in practice; both outperform GCN by 3–8% on social network datasets where structural (not feature-based) discrimination matters most.
- Ablation confirms: mean aggregation (GCN-style) underperforms sum (GIN-style) on all benchmarks with non-trivial graph structure.
- MLP depth $\geq 2$ consistently outperforms depth-1 (linear) — empirically validating the theoretical requirement for universal approximation.

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md)
