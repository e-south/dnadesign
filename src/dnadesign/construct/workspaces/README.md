<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Construct](../README.md) / [Construct docs](../docs/README.md)

# Choose a Construct workspace

A workspace holds related construction jobs, their input references and output
locations. Each project entry in `construct.workspace.yaml` names one YAML
configuration. Use separate entries for alternative templates or window
lengths; one template job can place several named parts from an input row.

## Choose a starting point

| Starting point | What it demonstrates |
| --- | --- |
| [Local demo](demo_anchor_template_local/README.md) | Place a part at either of two template slots; emit a window or the full construct. |
| [Shared-dataset demo](demo_anchor_template_shared_dataset/README.md) | Write two placement projects into one output dataset. |
| [Custom workspace](#create-a-workspace-for-your-own-inputs) | Import your own sequences and declare their placement. |

Start with [installation and the complete demo](../docs/getting-started.md) if
you have not used Construct before. With the environment active, inspect local
workspaces and available templates:

```bash
# List workspaces under the active root and the packaged examples.
construct workspace list
```

New workspaces are created at `<root>/<id>`. The root is the current directory
unless you set `--root` or `CONSTRUCT_WORKSPACE_ROOT`; the ID is a directory
name, not a path. Construct refuses to overwrite an existing workspace.

## Create a workspace for your own inputs

Run these commands from the directory where you want to keep the project:

```bash
# Create a blank configuration and input manifest template.
construct workspace init --id my_construct
# Enter the new workspace.
cd my_construct
# Inspect its generated project entry.
construct workspace show --workspace .
```

Edit `inputs/import_manifest.template.yaml` with your input and template
sequences, following the [import manifest reference](../docs/reference/seed-manifest.md).
Then import them:

```bash
# Import the edited manifest and print the resulting record IDs.
construct seed import-manifest \
  --manifest inputs/import_manifest.template.yaml --root outputs/usr_datasets
```

Use those IDs in `config.yaml` to choose the template, part placement and
output dataset. Update the matching project entry in `construct.workspace.yaml`
with the same dataset and template identities. The
[configuration reference](../docs/reference/config.md) describes placement and
window rules; the [registry reference](../docs/reference/workspace-registry.md)
explains the fields checked against that configuration.

```bash
# Check the edited registry and configuration together.
construct workspace doctor --workspace .
# Validate actual inputs and preview the new project.
construct workspace validate-project --workspace . --project my_construct --runtime
# Preview the output before saving sequences.
construct workspace run-project --workspace . --project my_construct --dry-run
```

Replace `my_construct` if you renamed the project. After reviewing the preview,
repeat `run-project` without `--dry-run` to write the results.

## Files to edit and inspect

| File | Purpose |
| --- | --- |
| `construct.workspace.yaml` | Project names, configuration paths and expected dataset identities. |
| `config.yaml` or `config.*.yaml` | Input, template, placement, window and output choices. |
| `inputs/import_manifest.template.yaml` | Your input sequences for a blank workspace. |
| `inputs/seed_manifest.yaml` | Demo seed receipt with record IDs and coordinates. |
| `outputs/` | Generated data and run artifacts; change configuration and rerun to update them. |

## Choose where data are stored

Packaged and blank workspaces use `outputs/usr_datasets` for their SeqLedger
root. To use another root, set the relevant input, template and output `root`
fields in the job configuration, or use the explicit
[`--seqledger-root` override](../docs/reference/cli.md#command-map).
Registry root fields describe intended locations; runtime validation reports
the paths actually used.

For downstream reuse, see the
[Construct → SeqLedger → Infer guide](../../seqledger/docs/operations/assembly/construct-infer-shared-dataset-runbook.md).
