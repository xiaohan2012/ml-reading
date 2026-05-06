---
title: "Attending to Graph Transformers (Survey)"
tags: [source, graph-transformer, survey, positional-encoding]
sources: [muller2023attending]
updated: 2026-05-04
---

# Attending to Graph Transformers (Survey)

**Source:** https://arxiv.org/abs/2302.04181
**Title:** Attending to Graph Transformers
**Date ingested:** 2026-05-04
**Type:** paper (survey)
**Authors:** Luis Müller, Mikhail Galkin, Christopher Morris, Ladislav Rampášek
**Venue:** arXiv 2023

## Summary

This survey provides the first systematic taxonomy of Graph Transformer architectures, bringing order to a rapidly growing and heterogeneous literature. The authors classify GT designs along several axes: attention mechanism (local neighborhood / global / hybrid), positional/structural encodings (Laplacian, random-walk, spectral, SPD, none), tokenization granularity, and expressiveness guarantees.

The paper also provides empirical probing: how well can Graph Transformers recover various graph properties (degree, clustering coefficient, shortest path distances, etc.)? How do they handle heterophily vs. homophily? Key findings: (1) GTs don't uniformly outperform GNNs — structure matters for which succeeds; (2) PE choice has outsized impact on expressiveness and benchmark performance; (3) the "global attention = better" assumption doesn't always hold — sometimes local attention is superior.

The survey serves as a reference guide for the GT landscape and is cited extensively when papers need to characterize the GT design space.

## Key Takeaways

- **Taxonomy of GTs**: local, global, hybrid; PE taxonomy; tokenization options — first systematic classification.
- **PE choice dominates**: which PE is used often matters more than other architectural choices.
- **Global attention is not always better**: on some graph tasks, local (neighborhood) attention outperforms full-graph attention.
- Empirical probing reveals which GT designs can recover which structural graph properties.
- Essential reference for understanding the GT design space before choosing a specific architecture.

## Entities & Concepts

- [graph-transformer](graph-transformer.md)
- [positional-encoding](positional-encoding.md)

## Relation to Other Wiki Pages

- [graph-transformer](graph-transformer.md): this is the primary survey of the GT landscape; provides the taxonomy used implicitly in the GT page.
- [positional-encoding](positional-encoding.md): the survey's PE analysis informs which PE choices are most impactful.
- [rampavsek2022graphgps](rampavsek2022graphgps.md): GPS is cited as a key hybrid local+global architecture; this survey contextualizes it within the broader landscape.
- [dwivedi2025relgt](dwivedi2025relgt.md): RelGT cites this survey to motivate that existing GTs fail on REGs — none of the surveyed architectures handles heterogeneity + temporality + scale simultaneously.
