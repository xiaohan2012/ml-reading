---
title: "RDL Reading Plan"
tags: [query, analysis, relational-deep-learning, reading-plan]
sources: [fey2024rdlposition, robinson2024relbench, chen2025relgnn, dwivedi2025relgt, ranjan2025relationaltr, fey2025kumorfm, fey2025kumorfm2, kipf2017gcn, hamilton2017graphsage, velickovic2018gat, xu2019gin, hu2020hgt, rampavsek2022graphgps, rossi2020tgn, yu2023dygformer]
updated: 2026-04-29
---

# RDL Reading Plan

## Critical Path (4 papers — covers the full landscape)

1. **[fey2024rdlposition](fey2024rdlposition.md)** — Fey et al., ICML 2024
   Blueprint. Defines REG, Training Table, temporal neighbor sampling. GNN = differentiable SQL JOIN+AGG. Start here.

2. **[robinson2024relbench](robinson2024relbench.md)** — Robinson et al., NeurIPS 2024
   Empirical companion. Validates RDL beats data scientist baseline on 30 tasks with 96% fewer human hours.

3. **[chen2025relgnn](chen2025relgnn.md)** — Chen et al., ICML 2025
   Best supervised method. Bridge/hub topology analysis; composite message passing FUSE op. Concise and principled.

4. **[ranjan2025relationaltr](ranjan2025relationaltr.md)** — Ranjan et al., ICLR 2026
   Best zero-shot model. Cell-level tokens + Relational Attention; 93% of supervised AUROC at 22M params; beats 27B LLMs.

## Core Architectures (add after critical path)

5. **[dwivedi2025relgt](dwivedi2025relgt.md)** — Dwivedi et al., 2025
   Best supervised GT. 5-element tokenization + subgraph GNN PE. Understand before reading KumoRFM.

6. **[wang2025griffin](wang2025griffin.md)** — Wang et al., ICML 2025
   First pretrained RFM. Cross-attention within rows + GNN across tables; strongest in low-data regimes.

## Foundation Models (add after core architectures)

7. **[fey2025kumorfm](fey2025kumorfm.md)** — Fey et al., 2025 whitepaper
   First end-to-end RFM system. PQL interface, ICL, explainability (saliency-per-cell), fine-tuning path.

8. **[fey2025kumorfm2](fey2025kumorfm2.md)** — Fey et al., 2025 (arxiv 2604.12596)
   Scales to 500B+ rows. Hierarchical 4-scale attention, task-conditioned ICL. First few-shot model to beat supervised SOTA.

## GNN Background (read if unfamiliar with GNNs)

- **[xu2019gin](xu2019gin.md)** — ICLR 2019. Most important for expressiveness theory: sum+MLP = WL-level; proves GCN/GraphSAGE are not injective.
- **[hamilton2017graphsage](hamilton2017graphsage.md)** — NeurIPS 2017. Inductive GNN via neighborhood sampling; HeteroGraphSAGE = RDL baseline.
- **[kipf2017gcn](kipf2017gcn.md)** — ICLR 2017. Spectral GNN; renormalization trick. Foundational but transductive.
- **[velickovic2018gat](velickovic2018gat.md)** — ICLR 2018. Learned attention weights per neighbor; multi-head; inductive.
- **[hu2020hgt](hu2020hgt.md)** — WWW 2020. Heterogeneous graph transformer; the standard baseline RDL papers compare against.
- **[rampavsek2022graphgps](rampavsek2022graphgps.md)** — NeurIPS 2022. GPS recipe (MPNN ∥ GlobalAttn per layer); parent design to RelGT.

## Temporal Graph Background (read if focusing on temporal RDL)

- **[rossi2020tgn](rossi2020tgn.md)** — 2020. Memory module + temporal graph attention; 30× faster than TGAT; canonical CTDG model.
- **[yu2023dygformer](yu2023dygformer.md)** — NeurIPS 2023. First-hop Transformer + patching + neighbor co-occurrence encoding; current CTDG SOTA.

## Reading Order by Goal

| Goal | Papers |
|---|---|
| Understand RDL from scratch | 1 → 2 → 3 → 4 |
| Build a supervised RDL model | 1 → 2 → 3 → 5 |
| Build a foundation model | 1 → 2 → 4 → 6 → 7 → 8 |
| Benchmark/reproduce results | 2 → 3 → 5 (all use RelBench v1) |
| Temporal graph focus | tgn → dygformer → 1 → 2 |
| GNN theory background | gcn → gin → graphsage → 1 |
