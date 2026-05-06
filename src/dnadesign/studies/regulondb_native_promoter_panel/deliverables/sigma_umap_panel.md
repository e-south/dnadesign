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

### sigma_umap_intermediate_embedding_7b_native_source_record_seq_mean | Native Sigma UMAP

#### Plot details

Exploratory UMAP over the native source-record intermediate embedding. The notebook hue dropdown exposes configured RegulonDB metadata overlays so the same fixed geometry can be audited by sigma-factor set, confidence level, metadata completeness, and emitted length when those columns are present.

### sigma_umap_intermediate_embedding_7b_core60_tss_upstream | Core60 Sigma UMAP

#### Plot details

Exploratory UMAP over the TSS-upstream core60 intermediate embedding. The panel uses the same hue contract as the native UMAP so reviewers can compare metadata organization without treating UMAP separation as the decision metric.

## Interpretation

Use these plots to inspect whether sigma-factor categories are plausibly organized in latent space. Do not treat UMAP separation as a decision metric. Any apparent category structure should be followed by high-dimensional distance, neighbor, or enrichment summaries before it is used operationally.
