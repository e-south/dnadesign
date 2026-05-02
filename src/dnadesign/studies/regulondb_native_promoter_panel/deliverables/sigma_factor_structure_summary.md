# Sigma-Factor Structure Summary

## Purpose

Check whether RegulonDB sigma-factor set metadata has projection-independent structure in each candidate representation. This is a high-dimensional cohort check and should be read before the appendix UMAP panels.

## Inputs

- Native and TSS-upstream core60 Evo 2 7B intermediate embedding views.
- Native and TSS-upstream core60 Evo 2 7B output-layer mean views.
- `regulondb__sigma_factor_set` metadata.

## Outputs

- `sigma_factor_structure_summary`

## Interpretation

The separation ratio compares mean between-cohort centroid distance to mean within-cohort centroid distance. Higher values indicate that sigma-factor set labels are more separated under the configured representation. Rare sigma-factor sets below the configured minimum group size are excluded from the metric rather than forced into noisy centroids.
