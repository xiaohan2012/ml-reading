---
title: Relational Entity Graph (REG)
tags: [concept, relational-deep-learning, heterogeneous-graph, temporal-graph]
sources: [relational-deep-learning-position, relational-graph-transformer, relgnn, griffin]
updated: 2026-04-29
---

# Relational Entity Graph (REG)

## Description

A Relational Entity Graph is the graph representation of a relational database used in [Relational Deep Learning](relational-deep-learning.md). Formally, a REG is a heterogeneous temporal graph $G = (V, E, \phi, \psi, \tau)$ where:

- $V$: nodes representing database entities (table rows)
- $E$: edges representing primary-foreign key relationships
- $\phi$: node type mapping (which table the entity belongs to)
- $\psi$: edge relation type mapping
- $\tau$: timestamp associated with each entity

REGs have three distinctive properties that make them hard for standard GNNs and GTs:
1. **Schema-defined structure**: topology is determined by primary-foreign key constraints, not arbitrary connections
2. **Temporal dynamics**: entities carry timestamps; future information must be excluded during training (temporal leakage prevention)
3. **Multi-type heterogeneity**: different tables have different attribute schemas and modalities

These properties rule out most standard positional encodings (Laplacian, random walk, node2vec), which are expensive, assume homogeneity, or don't handle temporal constraints.

## Appearances in Sources

- [relational-deep-learning-position](relational-deep-learning-position.md) — formally defined (Sec. 3.2); introduces $\phi$, $\psi$, $\tau$ mappings and the time-consistent computation graph algorithm
- [relational-graph-transformer](relational-graph-transformer.md) — defined as the problem domain; RelGT's tokenization is designed around REG's three properties
- [relational-transformer](relational-transformer.md) — RT operates at cell level on the same relational structure; primary-foreign key links are encoded as attention masks
- [relgnn](relgnn.md) — first structural characterization of REG topology: bridge nodes (2 FKs) and hub nodes (3+ FKs); atomic routes as schema-derived paths
- [griffin](griffin.md) — converts RDB to heterogeneous temporal graph; converts rows to nodes with cross-attention over cell features

## Related Concepts

- [relational-deep-learning](relational-deep-learning.md) — framework that uses REGs
- [multi-element-tokenization](multi-element-tokenization.md) — RelGT's tokenization strategy tailored to REGs
- [relational-attention](relational-attention.md) — RT's attention mechanism over F→P and P→F links in REGs
- [temporal-graph](temporal-graph.md) — REGs are a special case: temporal heterogeneous graphs with schema-defined structure
- [graph-transformer](graph-transformer.md) — architecture extended to handle REGs
