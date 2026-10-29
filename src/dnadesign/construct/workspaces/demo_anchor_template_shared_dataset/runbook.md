[Shared-dataset demo](README.md)

# Save two placements to one dataset

Run these commands from a copied shared-dataset workspace with the Construct
environment active. Follow the [setup and copy instructions](README.md) first.

## Prepare the inputs

```bash
# Seed the same four DNA parts and template used by the local demo.
construct seed anchor-template-demo \
  --root outputs/usr_datasets --manifest inputs/seed_manifest.yaml
# Inspect the two projects and check their configuration files.
construct workspace show --workspace .
# Check the project entries against their configuration files.
construct workspace doctor --workspace .
```

The projects replace slot A at `[2300, 2335)` or slot B at `[3621, 3656)` in
the template. Both extract a 1,000-base window centered on the inserted part.
Coordinates are zero-based and half-open.

## Preview both placements

```bash
# Resolve both sets of inputs and check the shared output destination.
construct workspace validate-project \
  --workspace . --project slot_a_window --runtime
# Resolve the slot B inputs and output destination.
construct workspace validate-project \
  --workspace . --project slot_b_window --runtime
# Preview each write before adding sequences to the output dataset.
construct workspace run-project \
  --workspace . --project slot_a_window --dry-run
# Preview the slot B write.
construct workspace run-project \
  --workspace . --project slot_b_window --dry-run
```

Check that both previews name `anchor_template_shared_dataset_demo` as the
output dataset under `outputs/usr_datasets`.

## Save and inspect the combined dataset

```bash
# Append the windows from each placement.
construct workspace run-project --workspace . --project slot_a_window
# Append the slot B windows to the same dataset.
construct workspace run-project --workspace . --project slot_b_window
# Verify the dataset, then inspect source labels and input lineage.
seqledger --root outputs/usr_datasets validate \
  anchor_template_shared_dataset_demo --strict
# Inspect source labels and input lineage for all eight windows.
seqledger --root outputs/usr_datasets head \
  anchor_template_shared_dataset_demo -n 8 \
  --columns id,usr_label__primary,construct__input_id,construct__template_id
```

Each run writes four rows, giving eight windows in the combined dataset.
`usr_label__primary` preserves the input label; `construct__*` columns record
which input, template and placement produced each output. Labels can repeat
across placements, so use record IDs and lineage to distinguish them.

## Adapt or continue

Add a project by creating its config and registering it in
`construct.workspace.yaml`. Keep `output.on_conflict: error` to detect output
ID collisions. For interrupted writes, see the
[retry policy](../../docs/reference/outputs.md#dataset-write-behavior).
To change storage roots, configure them explicitly and revalidate both
projects before writing.

The optional [shell wrapper](runbook.sh) provides `seed`, `validate-all`,
`dry-run-all` and `run-all` modes. Every mode seeds inputs first. It uses the
DNADesign checkout path saved during workspace creation; set
`CONSTRUCT_RUNBOOK_PROJECT_ROOT` if that checkout moves.

To calculate sequence-model outputs from these windows, continue with the
[Construct → SeqLedger → Infer guide](../../../seqledger/docs/operations/assembly/construct-infer-shared-dataset-runbook.md).
