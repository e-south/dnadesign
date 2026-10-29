<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Infer guide](../README.md) / [Operations](README.md)

# Save extraction results in SeqLedger

This procedure creates a small dataset, extracts pooled logits and mean log-likelihoods, and stores them alongside the sequence records. You can inspect and validate the job without a GPU; executing Evo2 requires the [GPU runtime](../../../../../docs/setup/installation.md#gpu-extra-infer-evo2).

Use an activated Infer environment. Local extraction needs Infer and SeqLedger; the optional scheduler section also needs OPS. The [workflow walkthrough](../tutorials/demo_pressure_test_usr_ops_notify.md) explains validation, persistence and resuming.

## 1. Create a workspace and input

```bash
# Create a workspace configured for dataset write-back.
infer workspace init --id demo_usr_pressure --profile usr-pressure
# Locate the workspace created in this directory.
export WORKSPACE_ROOT="$PWD/workspaces/demo_usr_pressure"
# Reuse its configuration for every validation and run.
export INFER_CONFIG="$WORKSPACE_ROOT/config.yaml"
# Keep the example dataset inside the workspace.
export USR_ROOT="$WORKSPACE_ROOT/outputs/usr_datasets"
# Match the dataset name declared in the generated configuration.
export DATASET_ID="infer_pressure_demo"

# Supply a small input without relying on files in the source checkout.
cat > "$WORKSPACE_ROOT/inputs/sequences.csv" <<'CSV'
sequence
ACGTACGT
ACGTTCGT
TGCATGCA
CSV
```

The workspace contains the configuration and directories, but no dataset yet. Its `source: usr` selects SeqLedger, and `ingest.root` resolves relative to `config.yaml`.

## 2. Register outputs and import sequences

```bash
# Print the registration command derived from this job's output types.
infer validate seqledger-registry --config "$INFER_CONFIG"
```

Use the printed `seqledger` command in your activated environment. Omit its `uv run` prefix when working outside a checkout. It registers pooled logits as `list<float64>` and mean log-likelihood as `float64`, under the pattern `infer__<model_id>__<job_id>__<out_id>`. Then create and populate the dataset:

```bash
# Create the dataset under the registered root.
seqledger --root "$USR_ROOT" init "$DATASET_ID" --source "infer demo"
# Import the three DNA records.
seqledger --root "$USR_ROOT" import "$DATASET_ID" --from csv \
  --path "$WORKSPACE_ROOT/inputs/sequences.csv" --bio-type dna --alphabet dna_4
```

## 3. Validate before execution

```bash
# Check the model and job configuration.
infer validate config --config "$INFER_CONFIG"
# Verify the output registration against the configuration.
infer validate seqledger-registry --config "$INFER_CONFIG"
# Check the resolved dataset and required field without running the model.
infer run --config "$INFER_CONFIG" --dry-run
```

If output definitions change, update their namespace registration before inference. The job named `pressure_evo2_logits_llr` requests `evo2.log_likelihood` with `reduction: mean`; its `llr_mean` output ID is a label, not a likelihood-ratio calculation.

## 4. Extract and inspect results

Once Evo2 dependencies and a suitable GPU are available:

```bash
# Execute the configured extraction job and write its values to SeqLedger.
infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr

# Inspect attached values and recorded dataset events.
seqledger --root "$USR_ROOT" head "$DATASET_ID" -n 5
# Confirm that write-back produced dataset events.
seqledger --root "$USR_ROOT" events tail "$DATASET_ID" -n 20
```

A successful run attaches the two requested output columns to the input record IDs. The CLI summary reports the output names and shapes. Inspect those columns before passing them to another tool.

## 5. Resume or reset

Rerunning the same job resumes processing and skips completed outputs:

```bash
# Run the job, skipping already completed rows.
infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr
```

To recompute this example from scratch, first archive its Infer overlay:

```bash
# Optional reset: preserve base records and archive only the Infer overlay.
infer prune --seqledger "$DATASET_ID" --seqledger-root "$USR_ROOT"
# Run the job, skipping already completed rows.
infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr
```

`prune` does not remove sequence-view sidecars. Feature-bundle reuse also checks request metadata; see the [resume contract](../reference/evo2-provider.md#resume-and-digest-behavior).

## Optional: run through a scheduler

Local extraction does not require OPS. On an SGE cluster with OPS installed, generate a runbook for the same workspace:

```bash
# Keep the runbook beside this workspace's logs.
export OPS_RUNBOOK="$WORKSPACE_ROOT/outputs/logs/ops/runbooks/infer-pressure.runbook.yaml"
# Generate a scheduler plan without notifications.
ops runbook init \
  --runbook "$OPS_RUNBOOK" \
  --workflow infer \
  --workspace-root "$WORKSPACE_ROOT" \
  --id infer_pressure_test \
  --no-notify
# Inspect paths, commands and requested resources.
ops runbook plan --runbook "$OPS_RUNBOOK"
```

Review the generated resources and paths, then run the preflight without submitting:

```bash
# Execute the runbook with the submission mode specified below.
ops runbook execute \
  --runbook "$OPS_RUNBOOK" \
  --audit-json "$WORKSPACE_ROOT/outputs/logs/ops/audit/infer-pressure.audit.json" \
  --no-submit
```

When the plan and environment are ready, submit explicitly:

```bash
# Execute the runbook with the submission mode specified below.
ops runbook execute \
  --runbook "$OPS_RUNBOOK" \
  --audit-json "$WORKSPACE_ROOT/outputs/logs/ops/audit/infer-pressure-submit.audit.json" \
  --submit
```

For notification setup, see [Notify](../../../notify/). The runbook initializer accepts `--with-notify` when delivery is configured. See the [OPS runbook guide](../../../ops/docs/) for orchestration and the [SCC setup guide](scc-evo2-gpu-uv-runbook.md) for BU-specific GPU preparation.

## Choose embedding outputs

To request intermediate embeddings instead of the supplied outputs, replace the job's `outputs` list with:

```yaml
# Average the selected layer over sequence positions for each record.
outputs:
  - id: emb_mid
    fn: evo2.embedding
    params:
      layer: mid
      pool: { method: mean, dim: 1 }
    format: list
  - id: emb_final
    fn: evo2.embedding
    params:
      layer: final
      pool: { method: mean, dim: 1 }
    format: list
```

`mid` and `final` are semantic layer aliases. `dim: 1` averages over sequence positions. Rerun output registration and validation after changing the configuration. For anchor-aware pooling or bundled likelihoods and vectors, use [feature bundles](evo2-sequence-features.md).
