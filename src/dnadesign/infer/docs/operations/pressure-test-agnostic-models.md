## Infer Pressure Test: Agnostic Model Namespaces + SeqLedger Write-Back

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28

This guide pressure-tests infer as a model-agnostic extraction engine with explicit namespace contracts and SeqLedger write-back.

For a full walkthrough, use the [end-to-end demo tutorial](../tutorials/demo_pressure_test_usr_ops_notify.md).

For deterministic SCC GPU environment setup before pressure runs, use the [SCC Evo2 GPU environment runbook](scc-evo2-gpu-uv-runbook.md).

### Objective

- run extract jobs that produce multiple outputs (for example logits + log-likelihood ratio variants),
- ensure outputs attach to SeqLedger with infer-prefixed namespaced columns:
  - `infer__<model_id>__<job_id>__<out_id>`
- support local CLI execution and ops runbook orchestration.

### Safety Posture

- Use `--dry-run` and `--no-submit` first.
- Use read-only scheduler checks before submit (`qstat -u "$USER"`).
- Keep `ingest.root` explicit for workspace and cluster runs.
- For fresh SeqLedger test datasets, register exact infer output types before write-back:
  - pooled Evo2 likelihoods are `float64`
  - pooled Evo2 logits are `list<float64>`
- Use `infer validate seqledger-registry --config ...` to derive the exact namespace registration command from the active infer config.

## Ordered procedure

### 1) Prepare workspace and variables

```bash
uv run infer workspace init --id demo_usr_pressure --profile usr-pressure # Create the pressure-test workspace scaffold.
export WORKSPACE_ROOT="$PWD/workspaces/demo_usr_pressure" # Pin workspace root for subsequent commands.
export INFER_CONFIG="$WORKSPACE_ROOT/config.yaml" # Reuse one config path across checks and runs.
export USR_ROOT="$WORKSPACE_ROOT/outputs/usr_datasets" # Point SeqLedger tooling at workspace-local datasets.
export DATASET_ID="infer_pressure_demo" # Set target dataset id once for repeatable CLI calls.
```


The scaffold creates directories and a config, but no input dataset. First
prepare the SeqLedger registry from the configured output types:

```bash
# Print the namespace registration command for this config.
uv run infer validate seqledger-registry --config "$INFER_CONFIG"
```

Run the command printed after `register:` to create the registry. Then create
the dataset and import the included sequences:

```bash
# Create the dataset under the registered SeqLedger root.
uv run seqledger --root "$USR_ROOT" init "$DATASET_ID" --source "infer demo"
# Import the included DNA sequences.
uv run seqledger --root "$USR_ROOT" import "$DATASET_ID" --from csv \
  --path src/dnadesign/seqledger/assets/demo_material/demo_sequences.csv \
  --bio-type dna --alphabet dna_4
```

### 2) Contract preflight

```bash
uv run infer validate config --config "$INFER_CONFIG" # Check config schema and runtime contracts.
uv run infer validate seqledger-registry --config "$INFER_CONFIG" # Derive/verify namespace registration requirements.
uv run infer run --config "$INFER_CONFIG" --dry-run # Exercise execution-path contracts without mutating datasets; SeqLedger jobs still preflight the resolved dataset.
```

If output definitions change, rerun namespace registration before inference.

### 3) Execute local pressure test

```bash
uv run infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr # Run the pressure job locally.
```

### 4) Verify SeqLedger state and events

```bash
uv run seqledger --root "$USR_ROOT" head "$DATASET_ID" -n 5 # Inspect resulting records after write-back.
uv run seqledger --root "$USR_ROOT" events tail "$DATASET_ID" -n 20 # Confirm infer event emission for the dataset.
```

### 5) Resume and prune the infer namespace when needed

Second runs on the same dataset should resume and skip completed infer rows:

```bash
uv run infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr # Confirm resume behavior on rerun.
```

To reset only infer outputs for the dataset, archive the infer overlay and rerun:

```bash
uv run infer prune --seqledger "$DATASET_ID" --seqledger-root "$USR_ROOT" # Archive infer overlay columns for a clean rerun.
uv run infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr # Re-run pressure job after infer namespace reset.
```

### 6) Initialize infer ops runbook

```bash
# Keep Ops runbooks workspace-scoped under the workspace logs tree.
export OPS_RUNBOOK="$WORKSPACE_ROOT/outputs/logs/ops/runbooks/infer-pressure.runbook.yaml" # Reuse one stable workspace-scoped Ops runbook path.
# Generate the infer workflow runbook at that workspace-scoped path.
uv run ops runbook init \
  --runbook "$OPS_RUNBOOK" \
  --workflow infer \
  --workspace-root "$WORKSPACE_ROOT" \
  --id infer_pressure_test \
  --no-notify
```

### 7) Plan and execute no-submit preflight

```bash
uv run ops runbook presets # Review workflow presets before execution.
uv run ops runbook plan --runbook "$OPS_RUNBOOK" # Produce a deterministic execution plan.
# Execute planned phases in dry scheduler mode.
uv run ops runbook execute \
  --runbook "$OPS_RUNBOOK" \
  --audit-json "$WORKSPACE_ROOT/outputs/logs/ops/audit/infer-pressure.audit.json" \
  --no-submit
```

### 8) Submit after preflight passes

```bash
qstat -u "$USER" # Verify queue state before submitting jobs.
# Execute submit path after preflight checks pass.
uv run ops runbook execute \
  --runbook "$OPS_RUNBOOK" \
  --audit-json "$WORKSPACE_ROOT/outputs/logs/ops/audit/infer-pressure-submit.audit.json" \
  --submit
```

### 9) Enable notify in the same runbook when needed

First configure webhook secret wiring (for example by setting a readable `NOTIFY_WEBHOOK_FILE` path), then re-initialize the same runbook:

```bash
# Rebuild runbook with notify phases enabled.
uv run ops runbook init \
  --runbook "$OPS_RUNBOOK" \
  --workflow infer \
  --workspace-root "$WORKSPACE_ROOT" \
  --id infer_pressure_test \
  --with-notify \
  --force
```

### 10) Run focused ad-hoc extract checks when isolating issues

```bash
# Run a focused single-output extraction check.
uv run infer extract \
  --model-id evo2_7b \
  --fn evo2.log_likelihood \
  --format float \
  --seqledger "$DATASET_ID" \
  --seqledger-root "$USR_ROOT" \
  --field sequence \
  --write-back
```

For embedding pressure checks in config-driven runs, use semantic layer names instead of raw Evo2 block names when possible:

```yaml
outputs: # Declare embedding outputs for pressure checks.
  - id: emb_mid # Stable output id for mid-layer pooled embeddings.
    fn: evo2.embedding # Invoke Evo2 embedding extractor.
    params: # Configure semantic layer and pooling behavior.
      layer: mid # Use semantic mid-layer alias.
      pool: { method: mean, dim: 1 } # Mean-pool over sequence axis.
    format: list # Emit vector output as a list.
  - id: emb_final # Stable output id for final-layer pooled embeddings.
    fn: evo2.embedding # Invoke Evo2 embedding extractor.
    params: # Configure semantic layer and pooling behavior.
      layer: final # Use semantic final-layer alias.
      pool: { method: mean, dim: 1 } # Mean-pool over sequence axis.
    format: list # Emit vector output as a list.
```
