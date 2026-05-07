---
title: "k-GNN: Weisfeiler and Leman Go Neural — Higher-order Graph Neural Networks"
tags: [source, gnn, expressiveness, wl-test, higher-order]
sources: [morris2019kgnn]
updated: 2026-05-07
---

# k-GNN: Weisfeiler and Leman Go Neural

**Source:** https://arxiv.org/abs/1810.02244
**Title:** Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Morris, Ritzert, Fey, Hamilton, Lenssen, Rattan, Grohe
**Venue:** AAAI 2019

## Summary

- **What:** Standard GNNs (1-GNNs) cannot distinguish all non-isomorphic graphs — they are bounded by the 1-WL test — but no neural architecture had been proposed to provably exceed this bound.
- **How:** Define $k$-GNNs that perform message passing over $k$-tuples of nodes (not individual nodes), implementing the higher-order $k$-WL algorithm neurally.
- **So what:** $k$-GNNs for $k \geq 2$ are provably strictly more expressive than all MPNNs; 54.45% average MAE reduction on QM9 vs. 1-GNNs.

## Challenges & Novelty

The 1-WL algorithm iteratively hashes (node_color, multiset_of_neighbor_colors) — it fails on regular graphs (e.g., two 3-regular graphs with different cycle structures). GIN (Xu et al., ICLR 2019, concurrent work) proves 1-GNN ≡ 1-WL but doesn't go beyond it. k-GNN provides the construction to break through the 1-WL ceiling.

- **$k$-WL as a principled target:** the $k$-WL hierarchy is well-studied in graph theory and logic — it is known that $k$-WL is strictly more expressive than $(k-1)$-WL for $k \geq 2$ and that 3-WL can distinguish all non-isomorphic graphs (with few exceptions). k-GNN neurally implements this hierarchy.
- **Hierarchical 1-$k$-GNNs bootstrap from 1-GNN:** instead of random initialization, the $k$-tuple representations are initialized from 1-GNN node embeddings, enabling multi-granularity representations that combine node-level and tuple-level structure.
- **Computational tradeoff is explicit:** $k$-tuples create $O(N^k)$ tokens — $k=2$ gives $O(N^2)$ (edges as tokens), $k=3$ gives $O(N^3)$ (triangles as tokens). Practical variants restrict to $k=2$ or $k=3$.

## Relation to Prior Work

| Method | Unit | Expressive power | Complexity |
|---|---|---|---|
| [kipf2017gcn](kipf2017gcn.md) | Node | $<$ 1-WL | $O(|\mathcal{E}|)$ |
| [xu2019gin](xu2019gin.md) | Node | $=$ 1-WL | $O(|\mathcal{E}|)$ |
| **2-GNN** | Node pair | $\geq$ 1-WL, $\leq$ 3-WL | $O(N^2)$ |
| **3-GNN** | Node triple | $\geq$ 2-WL, $\leq$ 4-WL | $O(N^3)$ |
| [ying2021graphormer](ying2021graphormer.md) | Node (global attn) | $>$ 1-WL (via SPD bias) | $O(N^2)$ |

- [xu2019gin](xu2019gin.md): GIN and k-GNN both independently prove 1-GNN ≡ 1-WL (ICLR 2019 and AAAI 2019, concurrent). GIN focuses on the sum vs. mean distinction to reach the 1-WL ceiling; k-GNN focuses on going beyond it with higher-order tuples.
- [ying2021graphormer](ying2021graphormer.md): Graphormer exceeds 1-WL through spatial attention biases (all-pairs distances) — an orthogonal approach to k-GNN's tuple-based message passing, at the same $O(N^2)$ cost as 2-GNN.
- [gilmer2017mpnn](gilmer2017mpnn.md): k-GNNs generalize MPNN — message passing between $k$-tuples rather than nodes; the formalism is the same but the token space is $V^k$.

## Technical Details

**$k$-WL coloring.** For a $k$-tuple $\mathbf{s} = (s_1, \ldots, s_k) \in V^k$, define $j$-th neighborhood: $\delta_j(\mathbf{s}) = \{(s_1, \ldots, s_{j-1}, w, s_{j+1}, \ldots, s_k) : w \in V\}$ (replace $j$-th component with every node).

Color refinement:
$$c_k^{(t)}(\mathbf{s}) = \text{HASH}\!\left(c_k^{(t-1)}(\mathbf{s}),\, \left\{\!\!\left\{c_k^{(t-1)}(\mathbf{s}') : \mathbf{s}' \in \delta_j(\mathbf{s})\right\}\!\!\right\}_{j=1}^k\right)$$

**$k$-GNN layer.** Neural implementation of $k$-WL: each $k$-tuple $\mathbf{s}$ maintains a hidden state $h_\mathbf{s}^{(l)}$, updated by aggregating over its $j$-th neighborhoods:

$$h_\mathbf{s}^{(l+1)} = \sigma\!\left(W_1^{(l)} h_\mathbf{s}^{(l)} + \sum_{j=1}^k W_{2j}^{(l)} \sum_{\mathbf{s}' \in \delta_j(\mathbf{s})} h_{\mathbf{s}'}^{(l)}\right)$$

**Hierarchical 1-$k$-GNN.** Run a 1-GNN first to get node embeddings $\{h_v^{(L)}\}$. Initialize $k$-tuple: $h_{(s_1,\ldots,s_k)}^{(0)} = h_{s_1}^{(L)} \| \cdots \| h_{s_k}^{(L)}$. Then run $k$-GNN layers on top. This combines local node features (from 1-GNN) with higher-order structural features (from $k$-GNN).

**Local $k$-GNNs (practical variant).** Restrict $k$-tuples to connected subgraphs of size $k$ — e.g., edges for $k=2$, triangles for $k=3$. This reduces $O(N^k)$ to $O(N \cdot \bar{d}^{k-1})$ where $\bar{d}$ is average degree, making it practical for sparse graphs.

## Experiments

- Average 54.45% reduction in MAE across 12 QM9 molecular property tasks vs. 1-GNN baselines.
- Hierarchical 1-3-GNN outperforms 3-GNN without hierarchical initialization — bootstrapping from 1-GNN features is important.
- Local 3-GNN competitive with global 3-GNN at a fraction of the cost for molecular graphs with small average degree.
- On social networks (COLLAB, IMDB), 2-GNN shows smaller improvements over 1-GNN — higher-order structure matters more for molecular than social graph tasks.

## Entities & Concepts

- [graph-neural-network](graph-neural-network.md)
