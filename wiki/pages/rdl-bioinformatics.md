---
title: "RDL for Bioinformatics and Transcriptomics"
tags: [query, analysis, relational-deep-learning, bioinformatics, transcriptomics]
sources: [fey2024rdlposition, chen2025relgnn, fey2025kumorfm2, gu2026relbenchv2]
updated: 2026-04-29
---

# RDL for Bioinformatics and Transcriptomics

Bioinformatics data is inherently relational: most interesting prediction tasks require joining across multiple linked tables. This page maps bioinformatics data types to relational schemas and identifies where RDL methods apply.

## Flat / Single-Table (standard tabular ML sufficient)

| Data type | Structure | Notes |
|---|---|---|
| Bulk RNA-seq | samples × genes expression matrix | Flat; RDL adds value only when joining across donors, cohorts, pathways |
| GWAS summary stats | one row per SNP: position, p-value, effect size, MAF | Flat per study; relational when joining to gene annotations, pathways |
| VCF / variant calls | one row per variant: chrom, pos, ref, alt, genotype | Flat per sample; relational across individuals and phenotypes |
| Proteomics (mass spec) | samples × proteins; mirrors expression | Same as bulk RNA-seq |
| Phenotype / clinical metadata | samples × traits | Always joins to omics tables in practice |

---

## Relational Schemas

### EHR / Clinical Genomics (MIMIC-style — already in RelBench v2)

```
patients ──< admissions ──< diagnoses
                  │        ──< procedures
                  │        ──< lab_events (patient_id, admission_id, itemid, value, time)
                  │        ──< prescriptions
                  │
lab_items ──< lab_events   (itemid FK)
```

Temporal: `admissions.admittime`, `lab_events.charttime`. Seed entity = patient or admission. Already in [gu2026relbenchv2](gu2026relbenchv2.md) (MIMIC-IV, length-of-stay prediction).

---

### Bulk RNA-seq (relational extension)

```
donors ──< samples ──< expressions (sample_id, gene_id, count/TPM) ──> genes
               │
               └── metadata (tissue, condition, batch)

genes ──< gene_annotations (chromosome, biotype, length)
genes ──< pathway_members ──> pathways    (bridge: 2 FKs)
```

Tasks: disease status, drug response, tissue type — sample-level prediction.

---

### Single-Cell RNA-seq (scRNA-seq)

```
studies ──< donors ──< samples ──< cells
                                     │
                    cell_type ───────┘  (FK — assigned by clustering)
                                     │
                    cells ──< expressions (cell_id, gene_id, count)
                                               │
                                      genes ──┘

genes ──< pathway_members  ──> pathways       (bridge: 2 FKs — MSigDB, GO)
genes ──< gene_sets        ──> gene_set_library
```

Temporal: if time-course experiment (drug treatment at 0h, 6h, 24h), `samples.timepoint` = seed_time. Scale: Human Cell Atlas = ~50M cells × 30K genes → billion-row expressions table, matching [KumoRFM-2](fey2025kumorfm2.md)'s 500B-row regime.

Tasks:
- Cell-level: cell type classification, perturbation response
- Sample-level: patient treatment response
- Gene-level: differential expression

---

### Spatial Transcriptomics

```
samples ──< spots (spot_id, x, y, tissue_region)
                │
spots ──< expressions (spot_id, gene_id, count)  ──> genes

spots ──< spatial_neighbors (spot_id, neighbor_spot_id, distance)
          (self-referential — physical adjacency graph)
```

Spots are nodes; physical adjacency = edges; expression profiles = node features. Standard GNNs (GCN, GAT) already applied here. Multi-table value: connecting spots → samples → donors → clinical outcomes.

---

### Perturbation Transcriptomics (Perturb-seq / CRISPR screens)

```
cells ──< expressions   (cell_id, gene_id, count)     ──> genes
  │
  └── perturbations (gene_knocked_out, guide_RNA, plate, well)
  └── cell_metadata  (cell_cycle_phase, size, viability)

genes ──< protein_interactions ──> genes    (STRING DB — self-referential hub)
genes ──< regulatory_links     ──> genes    (GRN: TF → target gene)
```

Task: given a gene knockout, predict resulting expression profile across all other genes — link prediction + regression over a gene regulatory network.

---

### Multi-Omics (RNA + ATAC + Protein, e.g., 10x Multiome)

```
cells ──< rna_counts     (cell_id, gene_id, count)    ──> genes
     ──< atac_peaks      (cell_id, peak_id, count)    ──> peaks
     ──< protein_counts  (cell_id, protein_id, count) ──> proteins

peaks ──< peak_gene_links (peak_id, gene_id, correlation)  ──> genes
          (bridge: cis-regulatory elements linking ATAC → RNA)

proteins ──< protein_gene_map ──> genes   (bridge: coding genes)
```

Task: predict gene expression (RNA) from chromatin accessibility (ATAC) — cross-modal regression.

---

### Population Genomics / GWAS

```
individuals ──< genotypes (individual_id, variant_id, allele)  ──> variants
     │
     └──< phenotypes (individual_id, trait, value)
     └── cohorts (ancestry, study)

variants ──< variant_annotations (variant_id, pathway_id, consequence) ──> pathways
variants ──< gene_map ──> genes   (positional: variant within gene body)
```

The `genotypes` junction table is potentially billions of rows (individuals × variants). Seed entity = individual; task = disease risk prediction.

---

## RDL Structural Patterns Found in Bioinformatics

| Pattern | Bioinformatics example | RDL concept |
|---|---|---|
| One-to-many, temporal | patient → lab events over time | CTDG edges with timestamps |
| Bridge node (2 FKs) | gene–pathway membership; ATAC peak–gene link | [RelGNN](chen2025relgnn.md) bridge topology |
| Hub node (3+ FKs) | drug–protein–cell assay conditions | [RelGNN](chen2025relgnn.md) hub topology |
| Self-referential edges | gene regulatory network (TF → target) | Homogeneous edges on same node type |
| Billion-row junction | individual × variant genotypes; cell × gene expressions | [KumoRFM-2](fey2025kumorfm2.md) scale regime |
| Heterogeneous node types | cells, genes, pathways, peaks, proteins | Heterogeneous GNN / RelGT |

---

## Where RDL Adds the Most Value

| Setting | Why RDL helps |
|---|---|
| **Perturbation-seq** | Gene regulatory network = natural REG; GNN propagates perturbation effects through graph |
| **Multi-omics** | Cross-modal tables (RNA + ATAC + protein) linked by gene/peak bridge tables |
| **Clinical + scRNA-seq** | cells → samples → donors → clinical outcomes; multi-hop prediction unavailable to flat models |
| **Spatial transcriptomics** | Physical neighbor graph + expression → clinical; connecting to outcomes needs multi-table joins |
| **Population genomics** | Individuals → variants → genes → pathways; relational structure encodes biological hierarchy |

---

## Key Engineering Challenge: High-Dimensional Sparse Node Features

Standard RDL node encoders ([fey2024rdlposition](fey2024rdlposition.md), PyTorch Frame) handle tens of columns per row. Gene expression vectors are **30K-dimensional and sparse** — a fundamentally different regime.

Practical solution: use a domain-specific encoder (scVI, scBERT, Geneformer) as the initial embedding layer to compress expression profiles into dense vectors, then feed those into the RDL GNN. This is analogous to using sentence-BERT for text columns in standard RDL.

This two-stage design (expression encoder → relational GNN) has not yet appeared in the RDL literature as of the wiki's ingested papers — it is an open research direction.

---

## Related Pages

- [relational-deep-learning](relational-deep-learning.md) — RDL framework
- [relational-entity-graph](relational-entity-graph.md) — graph abstraction all RDL methods use
- [chen2025relgnn](chen2025relgnn.md) — bridge/hub topology analysis directly applicable to pathway/regulatory structures
- [fey2025kumorfm2](fey2025kumorfm2.md) — billion-scale support relevant to large single-cell atlases
- [gu2026relbenchv2](gu2026relbenchv2.md) — MIMIC-IV clinical data already in the benchmark
- [rdl-approaches](rdl-approaches.md) — which RDL models to use for which tasks
