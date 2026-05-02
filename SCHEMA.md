# Wiki Schema

## Identity
- **Path:** /Users/hanxiao/docs/ml-reading
- **Domain:** ML reading and research, with a focus on GNN and interpretability
- **Source types:** papers, URLs
- **Created:** 2026-04-28

## Page Frontmatter
Every wiki page must start with:
---
title: <page title>
tags: [tag1, tag2]
sources: [source-slug1]
updated: YYYY-MM-DD
---

## Cross-References
Use standard markdown links with paths relative to the linking file.
- From `wiki/pages/*.md` to another page: `[transformer-architecture](transformer-architecture.md)`
- From `wiki/index.md` or `wiki/overview.md` to a page: `[transformer-architecture](pages/transformer-architecture.md)`

## Log Entry Format
## [YYYY-MM-DD] <operation> | <title>
Operations: init, ingest, query, update, lint

## Index Categories
- Sources
- Entities
- Concepts
- Analyses

## Conventions
- raw/ is immutable — skills never modify it
- log.md is append-only — never rewritten, only appended
- index.md is updated on every operation that adds or changes pages
- All pages live flat in wiki/pages/ — no subdirectories
- overview.md reflects the current synthesis across all sources
