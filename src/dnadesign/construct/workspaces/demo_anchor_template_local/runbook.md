[Local demo](README.md)

# Run template placements

Run these commands from a copied demo workspace, with the Construct environment
active. See [setup](../../docs/getting-started.md#install-and-activate) and
[workspace creation](README.md) if you have not prepared one yet.

## Seed and inspect inputs

```bash
# Import the four parts and the template into the local SeqLedger root.
construct seed anchor-template-demo \
  --root outputs/usr_datasets --manifest inputs/seed_manifest.yaml
# Inspect project names, then check them against their configuration files.
construct workspace show --workspace .
# Check that project entries agree with their configuration files.
construct workspace doctor --workspace .
# Print input labels and DNA sequences.
seqledger --root outputs/usr_datasets head anchor_parts_demo -n 4 \
  --columns id,usr_label__primary,sequence
```

`inputs/seed_manifest.yaml` records dataset IDs, record IDs and coordinates.
The template contains two replaceable intervals: slot A at `[2300, 2335)` and
slot B at `[3621, 3656)`, using zero-based, half-open coordinates.

## Preview and save a window

```bash
# Resolve the input records, placement and planned sequence lengths.
construct workspace validate-project \
  --workspace . --project slot_a_window --runtime
# Preview output writes without saving assembled sequences.
construct workspace run-project \
  --workspace . --project slot_a_window --dry-run
# Save the four assembled windows and their lineage.
construct workspace run-project --workspace . --project slot_a_window
# Validate the output dataset.
seqledger --root outputs/usr_datasets validate \
  anchor_template_slot_a_window_1kb_demo --strict
```

The run reports four planned and four written rows. Validation ends with
`OK: validation passed.` Each output is 1,000 bases, including the complete
inserted part. Inspect its span using `construct__anchor_start` and
`construct__anchor_end`, as shown in the [first-run guide](../../docs/getting-started.md#save-and-inspect-sequences).

## Try the full construct or another slot

```bash
# Keep the full template after placing a part at slot A.
construct workspace validate-project \
  --workspace . --project slot_a_full --runtime
# Preview the full-length output without saving it.
construct workspace run-project \
  --workspace . --project slot_a_full --dry-run
```

Remove `--dry-run` to save this output. Choose `slot_b_window` or `slot_b_full`
to place the same parts at slot B. Each project has its own output dataset;
see the [project table](README.md#choose-an-output).

## Retries and data locations

The demo uses `output.on_conflict: error`, so repeating a saved job fails on
existing IDs. Use the [documented retry policy](../../docs/reference/outputs.md#dataset-write-behavior)
when resuming an interrupted write. To store data elsewhere, set explicit
input, template and output roots in the job config, or use
[`--seqledger-root`](../../docs/reference/cli.md#command-map) consistently for
validation and execution after importing the inputs there.

The optional [shell wrapper](runbook.sh) combines seeding and execution, for
example `./runbook.sh --mode dry-run --config config.slot_a.window.yaml`.
It seeds inputs even in dry-run mode. It runs through `uv` using a DNADesign
checkout path saved when the workspace was created; if that checkout moves,
set `CONSTRUCT_RUNBOOK_PROJECT_ROOT` to its new path or use the commands above.
