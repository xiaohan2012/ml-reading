---
name: rewrite
description: Rewrite a wiki page (or section) for conciseness, readability, and structure — with explicit passes for redundancy and paragraph-to-list conversion.
---

# Rewrite

Tighten a wiki page so it is faster to scan and easier to navigate, without dropping content.

## Principles

- **Conciseness:** cut hedging, filler, and repetition; every sentence must earn its place, and any fact removed from one section must survive in another.
- **Readability:** lead with framing and motivation before mechanics, and use bolded lead-ins so a reader skimming only the bold text reconstructs the argument.
- **Structure:** prefer nested bullets and comparison tables over prose paragraphs, with one idea per bullet and consistent depth across siblings.

## Process

1. **Read the full target** (whole page if no section specified) so cross-section redundancy is visible.
2. **Redundancy scan** — list every duplicated fact, restated definition, or overlapping section. For each, propose where it should live (keep one canonical home, drop or cross-reference the rest). Confirm with the user before deleting.
3. **Paragraph scan** — list every prose paragraph longer than ~3 sentences or that contains multiple parallel ideas. For each, propose either:
    - a nested bullet list (when the paragraph enumerates items, steps, or trade-offs), or
    - a comparison table (when the content is parallel across categories), or
    - a tightened paragraph with a bolded lead-in (when it's genuinely connective tissue).
4. **Diagnose remaining issues** — verbose sentences, oversized summaries, inconsistent bullet depth, missing lead-ins. Confirm any non-trivial restructuring with the user.
5. **Apply targeted edits.** Use nesting (indentation) to encode "this elaborates the parent" rather than flat sibling bullets when the relationship is hierarchical.
6. **Re-read to verify no fact was lost.** If a fact was moved, note where it now lives.

## Summary section

The `## Summary` is **high-level only** — what / how / so-what at one bullet each, nested only when an aspect has multiple parallel sub-ideas. Rules:

- **One idea per top-level bullet**; nest only if the aspect has 2+ parallel sub-ideas worth naming.
- **Name the move, not its mechanism** — mechanism lives in Technical Details.
- **No numbers, metrics, or benchmark deltas** — those live in Experiments.
- **No mechanism-rationale restatements** — if a sub-bullet explains *why* the parent works, cut it (the parent is the claim).
- When in doubt, prefer the shorter version and confirm with the user before adding back.

## Reporting

After edits, briefly report:

- Sections touched.
- Redundancies removed (and where the surviving copy lives).
- Paragraphs converted to lists/tables.
- Anything intentionally kept as prose (and why).
