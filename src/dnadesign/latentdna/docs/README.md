<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA](../README.md)

# LatentDNA guide

Use LatentDNA to examine how sequence representations vary, which sequences are close to chosen references, and whether different representations preserve similar neighborhoods. It consumes precomputed vectors and records each analysis with its inputs and outputs.

**Start with [Run a small local analysis](getting-started/README.md).** Create an example vector table, fit PCA, render a variance plot and export the reduced matrix.

## The objects you work with

- A **source** identifies input data, such as a Parquet table or Infer feature sidecars.
- A **view** selects a vector representation and its row identifiers. One source can provide several views.
- A **workspace** holds `config.yaml`, input references and generated files under `outputs/`.
- An **artifact** is a saved result with a manifest recording parameters and provenance. A **recipe** orders computations; a **deliverable** groups their expected outputs for review.

## Choose a comparison

| Analysis goal | Workflow |
| --- | --- |
| Measure change between paired representations of the same sequences | [Context comparisons](workflows/context-shift.md) |
| Compare neighborhood or cluster structure across representations | [Cross-view agreement](workflows/cross-view-agreement.md) |
| Measure resemblance to declared reference sequences or groups | [Reference distances](workflows/control-distances.md) |
| Examine metadata composition around references | [Landmark neighborhoods](workflows/landmark-neighborhoods.md) |
| Hand aligned features to another analysis tool | [Export matrices and tables](workflows/export-opal-x.md) |

A distance is meaningful only in its declared [coordinate space](concepts/coordinate-spaces.md). Metadata labels describe the comparison; geometric proximity alone does not establish sequence function.

## Configure and inspect

Use the [workspace guide](../workspaces/README.md) to organize an analysis and the [reference index](reference/README.md) for fields and commands. [Operations](operations/README.md) covers validation, missing or stale results, and notebook review. The [workspace snapshot](reference/workspace-snapshot-contract.md) summarizes artifact status, while [artifact naming](reference/artifact-naming.md) keeps larger analyses readable.

For integrations, start with [SeqLedger](../../seqledger/), [Infer](../../infer/) or the [external-workspace contract](../../../../docs/integrations/study-workspaces.md). Contributors can use the [architecture guide](architecture/README.md) and [analysis implementation contracts](dev/README.md).
