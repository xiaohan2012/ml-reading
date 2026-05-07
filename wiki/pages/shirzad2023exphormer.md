---
title: "Exphormer: Sparse Transformers for Graphs"
tags: [source, graph-transformer, scalability]
sources: [shirzad2023exphormer]
updated: 2026-05-07
---

# Exphormer: Sparse Transformers for Graphs

**Source:** https://arxiv.org/abs/2303.06147
**Title:** Exphormer: Sparse Transformers for Graphs
**Date ingested:** 2026-05-04
**Type:** paper
**Authors:** Hamed Shirzad, Ameya Velingker, Balaji Venkatachalam, Danica J. Sutherland, Ali Kemal Sinop
**Venue:** ICML 2023

## Summary

- **What:** Full $O(N^2)$ global attention (as in SAN, Graphormer) doesn't scale to large graphs; naive sparse approximations (Performer, BigBird) sacrifice structure without principled justification.
- **How:** Combine two complementary sparse components: (1) **expander graph attention** — sparse edges drawn from a mathematically-optimal expander graph with provable spectral mixing; (2) **virtual global nodes** — a small set of nodes that attend to all real nodes at $O(N)$ cost.
- **So what:** $O(N + |\mathcal{E}|)$ complexity; drop-in replacement for the global attention module in GraphGPS; better accuracy than Performer/BigBird approximations while scaling to million-node graphs.

## Challenges & Novelty

Sparse attention methods (BigBird, Performer) replace $O(N^2)$ attention with $O(N)$ or $O(N\sqrt{N})$ approximations — but these are agnostic to graph structure and don't guarantee good information mixing. Standard graph sparsity (neighborhood attention) handles local structure but misses long-range dependencies. Exphormer provides a principled sparse design that guarantees mixing.

- **Expander graphs have optimal mixing properties:** an expander is a sparse graph where every node subset $S$ has many neighbors outside $S$ — it is formally characterized by a large spectral gap $\lambda_1 - \lambda_2$. Information propagates across any expander in $O(\log N)$ hops despite $O(N)$ edges, ensuring that sparse attention over expander edges captures global structure.
- **Virtual nodes provide guaranteed global reach:** $m$ virtual nodes (with $m \ll N$), each attending to all real nodes, add $O(mN)$ attention operations — with $m = O(1)$, this is $O(N)$. Any node can send information to any other node in two hops: node $\to$ virtual node $\to$ any other node.
- **Drop-in for GraphGPS global attention:** Exphormer replaces the Performer/BigBird module in GraphGPS without changing the MPNN component — demonstrating modularity and showing that better sparse global attention directly improves GPS.

## Relation to Prior Work

| Attention type | Complexity | Mixing guarantee | Structure-aware |
|---|---|---|---|
| Full attention (SAN, Graphormer) | $O(N^2)$ | Perfect | No |
| Performer / BigBird | $O(N)$ / $O(N\sqrt{N})$ | None | No |
| Neighborhood attention (GT) | $O(|\mathcal{E}|)$ | Local only | Yes |
| **Exphormer** | $O(N + |\mathcal{E}|)$ | Spectral mixing | Yes |

- [rampavsek2022graphgps](rampavsek2022graphgps.md): Exphormer is a drop-in for GPS's global attention module; improves over Performer/BigBird on accuracy while maintaining same complexity class.
- [alon2020bottleneck](alon2020bottleneck.md): Exphormer's virtual global nodes directly address over-squashing — they provide shortcuts for long-range signals at $O(N)$ cost, analogous to the FA layer.
- [wu2023sgformer](wu2023sgformer.md): SGFormer takes the complementary approach — single dense global attention layer with linear kernel approximation; Exphormer uses structured sparse expander attention.

## Technical Details

**Expander graph construction.** An $(N, d, \lambda)$-expander is a $d$-regular graph on $N$ nodes with second eigenvalue $\lambda_2 \leq \lambda$. The expander property guarantees: for any two disjoint sets $S, T \subseteq V$ with $|S||T| = \Omega(N^2)$, the number of edges between $S$ and $T$ is $\Omega(|S||T|d/N)$ — nearly as many as in a random graph.

Practical expander construction: random $d$-regular graphs (Erdős–Rényi with fixed degree) for $d = O(\log N)$ are expanders with high probability. Alternatively: use Ramanujan graphs (explicit constructions achieving optimal spectral gap).

**Expander attention.** Compute standard dot-product attention but only between node pairs $(i, j)$ where edge $(i, j)$ exists in the expander graph:

$$a_{ij} = \frac{\exp(Q_i K_j^T / \sqrt{d})}{\sum_{k \in \mathcal{E}_\text{exp}(i)} \exp(Q_i K_k^T / \sqrt{d})}, \quad (i,j) \in \mathcal{E}_\text{exp}$$

**Virtual nodes.** Add $m = O(1)$ virtual nodes $\{v_1^\star, \ldots, v_m^\star\}$ connected to all real nodes. In each attention layer, virtual nodes:
1. Attend to all real nodes (global context aggregation)
2. Real nodes attend to all virtual nodes (global context broadcast)

Total cost: $O(mN)$ per layer for virtual node interactions.

**Full Exphormer attention:** expander attention ($O(|\mathcal{E}_\text{exp}|)$) + virtual node attention ($O(mN)$) + original graph neighborhood attention ($O(|\mathcal{E}|)$).

## Experiments

- GraphGPS + Exphormer outperforms GraphGPS + Performer on ZINC (MAE: 0.049 vs. 0.070), PCQM4Mv2, and large-scale OGB benchmarks.
- Scales to ogbn-products (2.4M nodes) with 3× faster training than full-graph attention methods.
- Expander sparsity ($d = 4, \ldots, 16$) is sufficient; adding more expander edges beyond $d = 8$ shows diminishing returns.
- Virtual nodes contribute most on tasks requiring global reasoning (graph-level prediction); expander attention is most important for node-level tasks.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [positional-encoding](positional-encoding.md)
