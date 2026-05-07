---
title: "GraphMixer: Do We Really Need Complicated Model Architectures For Temporal Networks?"
tags: [source, temporal-graph, dynamic-graph, ctdg, mlp-mixer]
sources: [cong2023graphmixer]
updated: 2026-05-07
---

# GraphMixer: Do We Really Need Complicated Model Architectures For Temporal Networks?

**Source:** https://arxiv.org/abs/2302.11636
**Title:** Do We Really Need Complicated Model Architectures For Temporal Networks?
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Cong, Zhang, Kang, Yuan, Wu, Zhou, Tong, Mahabadi
**Venue:** ICLR 2023

## Summary

- **What:** CTDG models have grown increasingly complex (memory modules, multi-head attention, Transformer patching) but it is unclear whether this complexity is necessary.
- **How:** Replace all recurrent and attention components with fixed time encoding + MLP-Mixer over sorted link history + mean-pooled node neighbors — a fully MLP-based three-component architecture.
- **So what:** GraphMixer matches or outperforms TGAT, TGN, and DyGFormer on 6/7 benchmarks while training 2–5× faster; challenges the necessity of architectural complexity for temporal link prediction.

## Challenges & Novelty

TGAT uses temporal graph attention over multi-hop neighborhoods; TGN adds a GRU memory module; DyGFormer applies Transformer with patching over first-hop histories. Each addition was motivated as necessary, yet direct comparisons were done under inconsistent implementations, making it unclear whether complexity or better engineering drove results.

- **MLP-Mixer over sorted link history is sufficient:** the link encoder retrieves the $L$ most recent links for each endpoint, encodes each with a fixed Bochner time feature, then applies channel-mixing and node-mixing MLPs (no pairwise attention) — matching Transformer-based history encoders on most benchmarks.
- **Mean pooling beats attention for node encoding:** attention over temporal neighbors overfits to high-degree hub nodes; element-wise mean over the $k$ most recent neighbors' features is more robust.
- **Link history dominates node structure:** ablation shows the link encoder contributes far more than the node encoder — recent temporal interactions encode more predictive signal than spatial neighborhood structure for link prediction.

## Relation to Prior Work

| Model | History encoder | Node encoder | Complexity | Memory |
|---|---|---|---|---|
| [xu2020tgat](xu2020tgat.md) | Temporal graph attention (2-hop) | — | $O(N \cdot k^2)$ | No |
| [rossi2020tgn](rossi2020tgn.md) | GRU memory | Temporal attention | $O(N)$ | Yes (GRU) |
| [yu2023dygformer](yu2023dygformer.md) | Transformer + patching | Neighbor co-occurrence | $O(L/P \cdot d)$ | No |
| **GraphMixer** | MLP-Mixer (fixed time enc.) | Mean pool | $O(L \cdot d)$ | No |

- [xu2020tgat](xu2020tgat.md): GraphMixer reuses TGAT's Bochner time encoding but applies it via MLP-Mixer rather than multi-head temporal attention.
- [rossi2020tgn](rossi2020tgn.md): TGN's memory module is replaced by GraphMixer's link encoder with comparable results — challenges the necessity of explicit memory.
- [yu2023dygformer](yu2023dygformer.md): both encode first-hop link history; DyGFormer uses Transformer + patching, GraphMixer uses MLP-Mixer — simpler, faster, comparably accurate.
- [huang2023tgb](huang2023tgb.md): TGB's finding that simple baselines outperform complex models on node property prediction converges with GraphMixer's result on link prediction.

## Technical Details

**Three-component architecture.**

1. **Link encoder** (MLP-Mixer over time-sorted links): retrieve $L$ most recent historical links for each of $u$ and $v$. Encode each link as a time-feature vector $\phi(\Delta t)$ (Bochner cosine/sine encoding). Apply a channel-mixing MLP across feature dimensions and a node-mixing MLP across the $L$ time steps. Output: fixed-size summary $\mathbf{z}_u^\text{link}, \mathbf{z}_v^\text{link}$.

2. **Node encoder** (mean pooling): for node $u$ at time $t$, average the raw features of its $k$ most recent neighbors:
$$\mathbf{z}_u^\text{node} = W \cdot \text{mean}(\{x_{w} : w \in \text{recent}_k(u, t)\})$$

3. **Link classifier** (MLP): $\hat{y} = \text{MLP}([\mathbf{z}_u^\text{link}\|\mathbf{z}_u^\text{node}\|\mathbf{z}_v^\text{link}\|\mathbf{z}_v^\text{node}])$

**MLP-Mixer pattern.** For $L$ link tokens each of dimension $d$: channel-mixing MLP operates independently across tokens (mixes features within each time step); node-mixing MLP operates independently across features (mixes across time steps). This is the temporal graph analogue of the spatial MLP-Mixer.

**Time encoding.** Fixed sinusoidal (Bochner-style) encoding of $\Delta t = t_\text{query} - t_\text{link}$ — same formula as TGAT but not learned.

## Experiments

- Matches or outperforms TGAT, TGN, DyGFormer on 6/7 datasets (Wikipedia, Reddit, MOOC, LastFM, Enron, UCI, Social Evo) in Average Precision.
- 2–5× faster training than DyGFormer due to elimination of pairwise attention; competitive with TGN speed.
- Ablation: removing the link encoder causes the largest accuracy drop; removing the node encoder has minimal impact — confirming link history > neighborhood structure for temporal link prediction.
- Mean node encoder outperforms attention-based temporal node encoder on 5/7 datasets — attention overfits to hub nodes.

## Entities & Concepts

- [temporal-graph](temporal-graph.md)
