# Sigma-factor UMAP panel

## Purpose

Review native RegulonDB promoter records and TSS-upstream core60 windows in the same LatentDNA plot-review surface. The panel is exploratory: it checks whether categorical sigma-factor metadata carried from USR records remains visible after Construct, Infer sidecars, LatentDNA materialization, and UMAP projection.

## Inputs

- `usr_regulondb_native_promoters`: native source records and RegulonDB metadata.
- `usr_regulondb_native_promoter_core60`: TSS-upstream core60 windows derived from native promoter records.
- Evo 2 7B intermediate embedding sidecars for the native sequence-mean view and core60 mean view.

## Outputs

- `sigma_umap_intermediate_embedding_7b_native_source_record_seq_mean`
- `sigma_umap_intermediate_embedding_7b_core60_tss_upstream`
- `latent_geometry_browser`

## Interpretation

Use these plots to inspect whether sigma-factor categories are plausibly organized in latent space. Do not treat UMAP separation as a decision metric. Any apparent category structure should be followed by high-dimensional distance, neighbor, or enrichment summaries before it is used operationally.
