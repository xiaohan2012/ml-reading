---
title: "DyGFormer: Towards Better Dynamic Graph Learning"
tags: [source, temporal-graph, transformer, dynamic-graph, ctdg, patching]
sources: [yu2023dygformer]
updated: 2026-04-29
---

# DyGFormer: Towards Better Dynamic Graph Learning

**Source:** https://arxiv.org/abs/2303.13047
**Title:** Towards Better Dynamic Graph Learning with History Distillation
**Date ingested:** 2026-04-29
**Type:** paper
**Authors:** Yu, Cao, Petzold
**Venue:** NeurIPS 2023
**Year:** 2023

## Summary

Yu et al. make two contributions: (1) **DyGFormer**, a Transformer-based CTDG architecture that addresses two failures of prior work — inability to model source/destination node correlations and inability to handle long interaction histories without truncation; and (2) **DyGLib**, a unified library for reproducible CTDG evaluation.

**Problem with prior methods.** TGAT and TGN compute node representations independently for source $u$ and destination $v$, combining them only at the decoder. This ignores structural correlations (shared neighborhoods). Also, for nodes with long histories, fixed-size sampling truncates useful context; RNN-based memory (TGN) suffers from vanishing/exploding gradients.

**DyGFormer architecture.** Given an interaction $(u, v, t)$:

1. **Extract first-hop histories**: $\mathcal{S}_u^t$ and $\mathcal{S}_v^t$ — all prior interactions of $u$ and $v$ before time $t$ (first-hop only, no multi-hop).

2. **Four encodings per interaction**:
   - Neighbor features $\mathbf{X}_{*,N}^t$ — node features of encountered neighbors
   - Edge features $\mathbf{X}_{*,E}^t$ — features on the interaction edge
   - Time interval encoding $\mathbf{X}_{*,T}^t$ — Bochner cosine/sine encoding of $\Delta t = t - t'$ (same as TGAT)
   - **Neighbor co-occurrence** $\mathbf{X}_{*,C}^t$ — a 2D vector per neighbor counting its frequency in $\mathcal{S}_u^t$ and $\mathcal{S}_v^t$ respectively; captures shared-neighborhood signal between $u$ and $v$

3. **Patching**: divide each sequence into non-overlapping patches of size $P$ (adaptive: larger $P$ for longer sequences, keeping patch count constant). Each patch flattens $P$ consecutive interactions → reduces sequence length from $|\mathcal{S}|$ to $\lceil |\mathcal{S}|/P \rceil$.

4. **Joint Transformer**: encode the *concatenated* patch sequences of both $u$ and $v$ together — $\mathbf{Z}^t = [\mathbf{Z}_u^t; \mathbf{Z}_v^t]$ — so attention flows both within and across the two nodes' histories. Standard pre-LN Transformer (ViT-style: GELU, LN before each block).

5. **Readout**: average Transformer output rows belonging to $u$ (resp. $v$) through a linear output layer → $\mathbf{h}_u^t$, $\mathbf{h}_v^t$.

**DyGLib.** Standardized training pipeline for CTDG: unified data format, same trainer for all methods, 13 datasets, 9 methods, three negative sampling strategies (random / historical / inductive). Identifies previously unreported bugs in prior implementations and corrects them.

**Results.** DyGFormer achieves SOTA on most datasets across all three negative sampling strategies and both transductive/inductive settings. Under the harder historical and inductive negative sampling (where random-NS methods saturate), performance gaps between methods become more meaningful.

## Key Takeaways

- **First-hop only is enough**: prior work assumed multi-hop aggregation is necessary; DyGFormer shows that long first-hop histories processed by Transformer outperform expensive multi-hop GNNs — sequence length > hop depth.
- **Neighbor co-occurrence explicitly models link propensity**: if $u$ and $v$ share many historical neighbors, they're more likely to interact — encoding shared frequency gives the model a direct signal about structural similarity.
- **Patching solves the length bottleneck**: instead of fixed-size sampling (TGAT/TGN) or RNN gradients (Jodie/DyRep), patching keeps computation constant and preserves all history via local temporal proximity within patches.
- **Joint encoding of u+v is architecturally novel**: previous CTDGs compute $\mathbf{h}_u$ and $\mathbf{h}_v$ separately; joint Transformer lets each attend to the other's history, effectively computing a joint embedding.
- **Evaluation methodology matters**: random negative sampling is too easy — historical and inductive NS reveal model differences that random NS masks. DyGLib's standardized eval exposes this.

## Entities & Concepts

- [temporal-graph](temporal-graph.md)

## Relation to Other Wiki Pages

- [temporal-graph](temporal-graph.md): DyGFormer is the current SOTA CTDG architecture; Transformer on first-hop sequences outperforms GNN multi-hop aggregation.
- [xu2020tgat](xu2020tgat.md): DyGFormer uses the same Bochner time encoding; unlike TGAT, replaces graph attention over 2-hop neighborhoods with Transformer over long first-hop sequences.
- [rossi2020tgn](rossi2020tgn.md): DyGFormer eliminates TGN's memory module (and its RNN gradient issues) by relying on long first-hop history + Transformer; patching handles the long-context problem memory was solving.
- [chmura2026tgm](chmura2026tgm.md): DyGLib is a complementary standardization effort; TGM unifies CTDG+DTDG, DyGLib focuses on CTDG reproducibility and evaluation rigor.
