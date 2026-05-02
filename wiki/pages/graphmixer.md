---
title: "GraphMixer: Do We Really Need Complicated Model Architectures For Temporal Networks?"
tags: [source, temporal-graph, dynamic-graph, ctdg, mlp-mixer]
sources: [graphmixer]
updated: 2026-04-29
---

# GraphMixer: Do We Really Need Complicated Model Architectures For Temporal Networks?

**Source:** https://arxiv.org/abs/2302.11636
**Title:** Do We Really Need Complicated Model Architectures For Temporal Networks?
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Cong, Zhang, Kang, Yuan, Wu, Zhou, Tong, Mahabadi
**Venue:** ICLR 2023
**Year:** 2023

## Summary

Cong, Zhang, Kang, Yuan, Wu, Zhou, Tong, and Mahabadi (Penn State, Meta AI, UIUC) ask whether RNN- and self-attention-based components are truly necessary for temporal graph learning. The answer is no: a simple MLP-only architecture — GraphMixer — consistently matches or outperforms TGAT, TGN, and DyGFormer across seven standard benchmarks with faster convergence.

**Three-component architecture.** GraphMixer decomposes temporal link prediction into three simple modules:

1. **Link Encoder** (MLP-Mixer over time-sorted temporal links):
   - For a query edge $(u, v, t)$, retrieve the $L$ most recent historical links for each of $u$ and $v$
   - Each link is encoded as a time-feature vector via a fixed time encoding $\phi(t)$ (Bochner/sinusoidal)
   - A channel-mixing MLP mixes across the $L$ time steps; a node-mixing MLP mixes across node features
   - Output: a fixed-size summary of recent link history for each endpoint

2. **Node Encoder** (mean-pooling over temporal neighborhood):
   - For node $u$ at query time $t$, average the features of its $k$ most recent neighbors' node features
   - No attention, no GRU — just mean pooling with a learned linear projection
   - Surprisingly, this simple aggregator matches attention-based temporal aggregation

3. **Link Classifier** (MLP):
   - Concatenates link encoder output + node encoder output for both $u$ and $v$
   - Single MLP predicts link probability

**No RNN, no self-attention.** The entire model uses only MLPs and mean-pooling. The time encoding is fixed (not learned). The link encoder is the closest thing to a sequence model — it applies MLP-Mixer's channel/node mixing independently rather than computing pairwise attention.

**Results.** On 7 temporal graph benchmarks (Wikipedia, Reddit, MOOC, LastFM, Enron, UCI, Social Evo):
- GraphMixer matches or outperforms TGAT, TGN, and DyGFormer on 6/7 datasets for Average Precision
- Trains 2–5× faster than DyGFormer; competitive with TGN speed
- Node encoder ablation: mean > attention for temporal graphs (attention over-fits to hub nodes)

**Key finding.** Recent link history (link encoder) is more informative than structural neighborhood (node encoder) for temporal link prediction. The node encoder's contribution is secondary — its simplicity (mean pooling) is a feature, not a limitation.

## Key Takeaways

- **MLP-only beats RNN+attention**: fixed time encoding + MLP-Mixer over sorted links + mean-pooled neighbors is sufficient to match or beat TGAT/TGN/DyGFormer.
- **Link history > neighborhood structure** for temporal link prediction: the link encoder (temporal history) dominates; node encoder (spatial structure) is complementary but secondary.
- **Mean pooling beats attention for node encoding**: attention over temporal neighbors overfits to high-degree hub nodes; mean pooling is more robust.
- **Architecture complexity is not the bottleneck**: the CTDG community's move toward more complex architectures (memory → Transformer → patching) may be over-engineering the link prediction problem.
- **Speed advantage**: no recurrent computation means GraphMixer parallelizes fully; 2–5× faster training than DyGFormer.

## Entities & Concepts

- [temporal-graph](temporal-graph.md)

## Relation to Other Wiki Pages

- [temporal-graph](temporal-graph.md): GraphMixer is a counter-argument to architectural complexity in CTDG learning; calibrates the field similarly to TGB's empirical finding that simple baselines often match complex models.
- [tgb](tgb.md): TGB's finding (simple baselines competitive with TGAT/TGN/DyGFormer) is echoed by GraphMixer's result — both suggest the community may be over-indexing on architectural complexity.
- [tgn](tgn.md): TGN's memory + GRU is replaced by GraphMixer's MLP link encoder with comparable or better results; challenges TGN's memory module as necessary.
- [tgat](tgat.md): GraphMixer replaces TGAT's Bochner attention with the same Bochner encoding but applies it via MLP-Mixer rather than temporal graph attention.
- [dygformer](dygformer.md): DyGFormer uses Transformer + patching over first-hop history; GraphMixer uses MLP-Mixer over the same history — simpler, faster, comparably accurate.
