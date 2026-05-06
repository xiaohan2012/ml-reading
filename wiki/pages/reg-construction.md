---
title: Building a Relational Entity Graph
tags: [query, analysis, relational-deep-learning, temporal-graph]
sources: [fey2024rdlposition, relational-entity-graph, training-table, xu2020tgat, rossi2020tgn, trivedi2019dyrep, chen2025relgnn]
updated: 2026-04-29
---

# Building a Relational Entity Graph

## From a Relational Database to a Graph

A relational database is a collection of tables linked by **primary key (PK) / foreign key (FK) relationships**. The [fey2024rdlposition](fey2024rdlposition.md) paper defines a mechanical, schema-driven conversion:

| Database concept | REG concept |
|---|---|
| Table row | Node |
| Table name | Node type $\phi(v)$ |
| FK relationship between tables | Edge |
| FK relation type (which tables are linked) | Edge type $\psi(e)$ |
| Row's timestamp column (if any) | Node timestamp $\tau(v)$ |

Formally: $G = (\mathcal{V}, \mathcal{E}, \phi, \psi, \tau)$

This is fully automatic from the schema — no domain knowledge is needed to decide which rows become nodes or which FK columns become edges. If Table A has a foreign key pointing to Table B, every row in A gets a directed edge to the corresponding row in B.

**Example**: an e-commerce database with tables `Users`, `Orders`, `Products`:
- Each user row → a `User`-type node
- Each order row → an `Order`-type node
- Each product row → a `Product`-type node
- FK `Orders.user_id → Users.id` → edges of type `placed_by` from each Order node to its User node
- FK `Orders.product_id → Products.id` → edges of type `contains` from each Order node to its Product node

The result is a **heterogeneous graph**: multiple node types (User, Order, Product), multiple edge types (placed\_by, contains) — which is why standard homogeneous GNNs don't apply directly.

## Why It Is a Temporal Graph

Most real-world relational databases have timestamps — orders have an order date, reviews have a posted date, events have an event time. The [relational-entity-graph](relational-entity-graph.md) preserves these as $\tau(v)$ on each node. The graph is temporal for two distinct reasons:

### 1. The data itself is an event stream

Each row in a transactional table (an Order, a Click, a Review) represents an event that happened at a specific time. When converted to a REG, the nodes carry those timestamps. The graph is not a static snapshot — it has an implicit time axis, structurally identical to the Continuous-Time Dynamic Graphs studied by [xu2020tgat](xu2020tgat.md), [rossi2020tgn](rossi2020tgn.md), and [trivedi2019dyrep](trivedi2019dyrep.md): a stream of timestamped events defining a graph.

### 2. Temporal leakage must be prevented

Consider predicting whether a user will churn *next month*. The label $y_v$ is computed from events *after* the seed time $t_v$. If the model saw those same future events during message passing, it would trivially memorize the label — **data leakage**.

The [training-table](training-table.md) abstraction handles this via the `seed_time` field: when computing the embedding for entity $v$ at seed time $t_v$, **only neighbors $w$ with $\tau(w) \leq t_v$ are included in the computation graph**. This is Algorithm 1 in [fey2024rdlposition](fey2024rdlposition.md) — the *time-consistent computation graph*.

This means the REG is queried differently for every training example: the same graph node has a different neighborhood depending on the seed time of the prediction task.

**Concretely**: for a churn prediction task with seed time March 1, the model sees all orders and reviews a user placed *before* March 1. For another example with seed time February 1, the same user's computation graph is smaller — February orders are excluded.

## The Three REG Properties That Make It Hard

Per [relational-entity-graph](relational-entity-graph.md), these three properties together rule out most off-the-shelf GNN/GT techniques:

1. **Schema-defined structure**: topology is fixed by the database schema (not arbitrary). Standard PEs (Laplacian, RWSE) are expensive, assume homogeneous graphs, and don't respect this structure.
2. **Temporal dynamics**: time-gated computation graphs mean you can't precompute a single adjacency matrix. Embeddings are functions of `(entity, seed_time)` pairs.
3. **Multi-type heterogeneity**: different tables have entirely different attribute schemas and modalities (text columns in Products, numeric in Orders, etc.), requiring separate encoders per node type.

[chen2025relgnn](chen2025relgnn.md) further characterizes the specific topological patterns that emerge from PK-FK graphs: **bridge nodes** (exactly 2 FK relationships, appear on paths between entity types) and **hub nodes** (3+ FK relationships, aggregate multiple entity types). Its FUSE operation is designed around these patterns.

## Related Concepts

- [relational-entity-graph](relational-entity-graph.md) — formal definition of the REG
- [training-table](training-table.md) — the seed\_time abstraction that makes the graph temporal
- [relational-deep-learning](relational-deep-learning.md) — the full RDL pipeline
- [temporal-graph](temporal-graph.md) — the broader CTDG literature that REGs specialize
- [chen2025relgnn](chen2025relgnn.md) — exploits bridge/hub topology specific to PK-FK graphs
