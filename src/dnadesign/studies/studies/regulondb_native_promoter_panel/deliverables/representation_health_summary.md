# Representation Health Summary

## Purpose

Screen native source-record and TSS-upstream core60 RegulonDB promoter representation views before interpreting sigma-factor structure. This is the first review gate because a collapsed or near-degenerate candidate X should not be rescued by a visually appealing UMAP.

## Inputs

- `usr_regulondb_native_promoters`: native source records and RegulonDB metadata.
- `usr_regulondb_native_promoter_core60`: TSS-upstream core60 windows derived from native promoter records.
- Evo 2 7B intermediate embedding and output-layer mean sidecars for native sequence-mean and core60 mean pooling.

## Outputs

- `representation_health_summary`
- `latent_geometry_browser`

### representation_health_summary | Representation Health Summary

#### Plot details

Square metric panels compare native and TSS-upstream core60 intermediate-embedding and output-layer mean views on retained-PCA effective rank, PC1 variance fraction, and sampled pairwise cosine-distance spread. Reference controls are not promoted to independent representation-health cohorts here.

## Interpretation

Read the metric panels as a collapse and richness check. Higher effective rank and broader pairwise cosine-distance spread indicate that a representation preserves more usable variation in the retained PCA diagnostic. This plot does not prove sigma-factor separation; it decides which representation views are worth carrying into native/core60 shift and sigma-structure summaries.
