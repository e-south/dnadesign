<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md) / [Workflows](README.md)

# Export matrices and tables

Export aligned numerical features for analysis in another tool. LatentDNA writes the feature matrix and its row and column ledgers; the consumer decides how to use those features for prediction, selection or visualization.

## Declare the handoff

An export in `config.yaml` names a `row_basis` and ordered `blocks`. Supported blocks are reduced views and selected numeric table columns. Specify alignment explicitly when a block uses a different row basis. The [local walkthrough](../getting-started/README.md) includes a complete reduced-view export.

For an export declared as `reduced_features`:

```bash
# Inspect the configured export before writing it.
latentdna explain export reduced_features --workspace /path/to/workspace
# Write the configured numeric bundle after reviewing its definition.
latentdna export matrix reduced_features --workspace /path/to/workspace
```

The bundle under `outputs/exports/reduced_features/` contains `matrix.npy`, `rows.parquet`, `features.parquet` and `manifest.json`. Keep these files together: row order and feature order are part of the handoff. Use `export table` for a tabular bundle or `export anndata` for an AnnData file with declared metadata and optional projections or neighbor graphs.

## Connect a consumer

Match observations to the exported row IDs rather than assuming two tables share an order. For [OPAL](../../../opal/), follow its [data configuration](../../../opal/docs/reference/configuration.md) to prepare records with a supported feature column and measured labels. An exported matrix alone is not an initialized campaign.

See the [export contract](../reference/export-contract.md) for bundle fields and [artifact manifests](../reference/artifact-manifests.md) for provenance and freshness checks.
