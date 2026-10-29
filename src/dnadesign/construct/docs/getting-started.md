<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Construct docs](README.md)

# Build your first construct

This demo replaces one interval in a template with each of the supplied DNA
parts, then extracts a 1,000-base window around the inserted part. You will
preview the outputs, save them to SeqLedger and inspect their coordinates.
All demo data stay in a new workspace.

## Install and activate

From the DNADesign checkout:

```bash
# Install Construct and its data-storage dependencies.
uv sync --locked --no-default-groups --extra construct
# Keep the commands available when moving outside the checkout.
source .venv/bin/activate
```

See [installation options](../../../../docs/setup/installation.md) for other
DNADesign tools. Keep this environment active for the commands below.

## Create the demo workspace

```bash
# Use a fresh temporary directory so the demo cannot overwrite an existing project.
construct_demo_root=$(mktemp -d)
construct workspace init --id construct_demo \
  --root "$construct_demo_root" --profile anchor-template-demo
cd "$construct_demo_root/construct_demo"

# Import the packaged parts and template into this workspace's SeqLedger root.
construct seed anchor-template-demo \
  --root outputs/usr_datasets --manifest inputs/seed_manifest.yaml
# Check that the workspace's project entries match their configuration files.
construct workspace doctor --workspace .
```

`workspace init` prints the workspace path. The seed command creates the input
and template datasets and records their IDs in `inputs/seed_manifest.yaml`.
A *project* is one named configuration in `construct.workspace.yaml`. Here,
`slot_a_window` means placement at slot A followed by a 1,000-base window.

## Preview the result

```bash
# Resolve input records, template coordinates and planned output lengths.
construct workspace validate-project \
  --workspace . --project slot_a_window --runtime
# Plan the same job without writing output sequences or metadata.
construct workspace run-project \
  --workspace . --project slot_a_window --dry-run
```

Runtime validation prints the resolved input and output roots, the replacement
interval and each planned sequence length. Inspect those before writing. The
seed step above writes demo inputs; the Construct dry-run does not write the
assembled outputs.

The dry-run summary reports four planned outputs and no collisions:

```text
Config validated (dry run): job=anchor_template_slot_a_window_1kb rows=4 output_collisions=0
```

## Save and inspect sequences

```bash
# Write the planned sequences and their lineage to SeqLedger.
construct workspace run-project --workspace . --project slot_a_window
# Check the resulting dataset and inspect the inserted-part coordinates.
seqledger --root outputs/usr_datasets validate \
  anchor_template_slot_a_window_1kb_demo --strict
seqledger --root outputs/usr_datasets head \
  anchor_template_slot_a_window_1kb_demo -n 3 \
  --columns id,usr_label__primary,construct__anchor_start,construct__anchor_end
```

The output dataset is under
`outputs/usr_datasets/anchor_template_slot_a_window_1kb_demo/`. The `sequence`
field contains the assembled window; `construct__anchor_start` and
`construct__anchor_end` locate the inserted part using zero-based, half-open
coordinates. Add `sequence` to `--columns` to print the DNA.

The run summary should report `rows_planned=4`, `rows_written=4` and
`rows_skipped_existing=0`. SeqLedger's strict validation then confirms:

```text
OK: validation passed.
```

The demo rejects a second write of the same output IDs. For a deliberate retry,
use the [conflict policy](reference/outputs.md#dataset-write-behavior) rather
than deleting or editing dataset files by hand.

## Choose the next task

- [Try another slot or the full template](../workspaces/demo_anchor_template_local/runbook.md).
- [Combine placements in one dataset](../workspaces/demo_anchor_template_shared_dataset/runbook.md).
- [Use your own sequences](../workspaces/README.md#create-a-workspace-for-your-own-inputs).
- [Pass the results to Infer](../../seqledger/docs/operations/assembly/construct-infer-shared-dataset-runbook.md).
