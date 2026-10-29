<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md)

# Check and rebuild analysis outputs

Validation checks whether a workspace can use its declared inputs. Artifact status checks whether saved results still match those inputs, parameters and implementation. Neither check establishes that a scientific interpretation is correct.

## Inspect before computing

In an activated LatentDNA environment, replace the workspace path below:

```bash
# Select an existing analysis workspace.
WORKSPACE="/absolute/path/to/my-analysis"
# Check declared source schemas without materializing embedding matrices.
latentdna validate workspace --workspace "$WORKSPACE" --deep
# Inspect configured views and persisted outputs.
latentdna inspect views --workspace "$WORKSPACE"
# List artifacts already present in the workspace.
latentdna inspect artifacts --workspace "$WORKSPACE"
# Read status without publishing a snapshot file.
latentdna workspace snapshot --workspace "$WORKSPACE" --json --dry-run
```

Deep validation can identify missing columns or stale view declarations. Planned sources may be empty; an alias referring to a missing feature payload is an error. A valid configuration does not imply that all outputs have been computed.

## Run a configured analysis

A workspace can declare recipes and deliverables. List the available deliverables before choosing one:

```bash
# Discover the output bundles declared by this workspace.
latentdna deliverable list --workspace "$WORKSPACE"
```

For a chosen ID, use `latentdna deliverable status <id>` and `latentdna deliverable run <id>`, each with `--workspace "$WORKSPACE"`. The run command invokes its recipe and checks the resulting outputs. Use `--dry-run` to preview a supported computation before writing.

For individual artifacts, rerun the producing command. Use `--force` when intentionally replacing an existing artifact after changing its configuration. Do not edit generated matrices or manifests by hand. The [manifest contract](../reference/artifact-manifests.md) explains integrity and freshness checks.

## Review in a notebook

Notebook review requires the `latentdna-notebook` installation profile and a notebook declaration in the workspace. Generate the notebook with `latentdna notebook generate <id> --workspace "$WORKSPACE"`, then run `latentdna notebook smoke --workspace "$WORKSPACE"`. The generated app reads saved artifacts; it does not compute missing prerequisites.

If generation reports `attention`, inspect missing outputs. If the default deliverable is stale, rebuild it before regenerating the notebook. See [CLI behavior](../reference/cli-contracts.md) for status details.

## Publish status or clear generated outputs

Omit `--dry-run` from `workspace snapshot` to write `outputs/status/workspace_snapshot.json`. Integrations can consume its [documented schema](../reference/workspace-snapshot-contract.md); `ops/status.registry.yaml` registers the status interface.

`workspace refresh` removes selected generated artifacts beneath `outputs/`. Inspect `latentdna workspace refresh --help` and preview the intended target with `--dry-run` before clearing it. Source datasets are outside that cleanup scope.
