<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md) / [Reference](README.md)

# CLI reference

Run `latentdna --help` to list command groups and `latentdna <group> <command> --help` for exact arguments. Commands that act on an analysis take `--workspace <id|path>`. Use an explicit path when running outside the workspace directory.

## Choose a command

| Task | Commands |
| --- | --- |
| Create or inspect configuration | `workspace init`, `workspace where`, `workspace list`, `workspace show`, `validate workspace`, `explain workspace` |
| Inspect inputs and outputs | `inspect source`, `inspect views`, `inspect alignment`, `inspect landmarks`, `inspect missingness`, `inspect artifacts` |
| Save row metadata or match views | `snapshot build`, `alignment build` |
| Build and transform vectors | `view materialize`, `view derive`, `view reduce`, `view stats` |
| Select rows and derive scalars | `sample build`, `scalar derive` |
| Compare geometry | `neighbors fit`, `cluster fit`, `projection fit`, `distance score`, `enrich score`, `agreement compare` |
| Export or plot | `export matrix`, `export table`, `export anndata`, `plot render` |
| Run declared analyses | `recipe validate`, `recipe run`, `deliverable list`, `deliverable status`, `deliverable run` |
| Review notebooks | `notebook generate`, `notebook smoke`, `inspect notebook-health` |
| Inspect or clear generated state | `workspace snapshot`, `workspace refresh`, `runs list`, `runs show`, `runs prune` |

`explain deliverable`, `explain plot` and `explain export` describe named configuration entries. `snapshot build` saves source rows and metadata; `workspace snapshot` reports analysis status. They serve different purposes.

## Output and execution flags

- `--format text|json|yaml` chooses output formatting; `--json` selects JSON directly.
- `--quiet` reduces text output to a concise success line.
- `--dry-run`, where supported, validates the target and reports expected outputs without writing them.
- `--force`, where supported, replaces an existing artifact. Check the intended target before using it.
- Stochastic analyses expose `--seed`; recipe and deliverable runs also expose `--progress none|text|json`.

Options differ by command. For example, `workspace refresh` supports a preview but does not take `--force`, and `notebook smoke` checks the workspace without a notebook ID argument.

## Analysis constraints

`view reduce` fits PCA. `projection fit` fits UMAP on a declared sample; the configured distance metric determines neighborhood construction. A projection is a visualization of that geometry, not a measurement of sequence function.

`neighbors fit` and `cluster fit` require exactly one of `--view` or `--reduced-view`. Reduced views already have fixed rows and cannot also take `--sample` or `--alignment`. Neighbor metrics include Euclidean and cosine; clustering supports k-means and Leiden.

Source-backed views select a vector column or matrix bundle. Derived views support differences, normalization, aggregation, reducer application and concatenation. See [view contracts](view-contract.md) before combining representations.

## Plot rendering

Use either a named `plots.<id>` entry or an inline plot specification, not both. Named plots provide a reusable configuration. Inline plots take `--kind` plus the relevant artifact arguments.

For example, after the [local walkthrough](../getting-started/README.md) has fitted `example_pca`:

```bash
# Render explained variance from the saved reducer in the example workspace.
latentdna plot render another_variance_plot --workspace /path/to/latentdna-demo \
  --kind curve --reducer example_pca
```

Supported plot families include projection scatters and grids, heatmaps, distance and scalar scatters, distributions, curves, cluster correspondence and agreement summaries. Plotters read persisted artifacts; they do not run missing analyses implicitly. Layout, labels and hues come from [plot configuration](workspace-schema.md#plot-presentation).

## Machine-readable results

| Surface | Schema or behavior |
| --- | --- |
| Artifact-producing command result | `latentdna.command_result.v1` |
| Artifact manifest | `latentdna.manifest.v1` |
| Deliverable status | `latentdna.deliverable_status.v1` |
| `latentdna workspace snapshot` | `latentdna.workspace_snapshot.v1` |

A workspace snapshot writes `outputs/status/workspace_snapshot.json` unless `--dry-run` is set. Generated artifacts live directly under `outputs/`. Nested output roots are rejected.

Notebook generation can return `attention` when a default plot is missing. It rejects stale default deliverables until they are rebuilt. `notebook smoke` and `inspect notebook-health` return a nonzero exit when health is `error`. See [operations](../operations/README.md) for the check-and-rebuild workflow.

Use [manifests](artifact-manifests.md) to inspect input and implementation provenance, [exports](export-contract.md) for data handoffs, and [workspace schema](workspace-schema.md) for configuration details.
