## infer end-to-end pressure-test demo (usr + ops + notify)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-13

This demo executes infer in an end-to-end pressure-test loop that can run standalone or via ops orchestration.

### Objective

1. Validate infer config and dry-run contracts.
2. Execute infer extraction pressure path.
3. Verify USR dataset state after write-back.
4. Execute ops runbook in no-submit and submit modes.
5. Optionally enable notify phases for delivery smoke checks.

### 1) Prepare workspace and config

```bash
uv run infer workspace init --id demo_usr_pressure --profile usr-pressure # Create pressure-test workspace and config.
export WORKSPACE_ROOT="$PWD/workspaces/demo_usr_pressure" # Reuse workspace root across steps.
export INFER_CONFIG="$WORKSPACE_ROOT/config.yaml" # Pin infer config path.
export USR_ROOT="$WORKSPACE_ROOT/outputs/usr_datasets" # Point USR tools at workspace-local datasets.
export DATASET_ID="infer_pressure_demo" # Set dataset id for repeatable commands.
```

### 2) Validate infer contract surface

```bash
uv run infer validate config --config "$INFER_CONFIG" # Validate infer config contract.
uv run infer validate usr-registry --config "$INFER_CONFIG" # Derive the exact infer namespace registration command for this config.
uv run infer run --config "$INFER_CONFIG" --dry-run # Exercise run path without mutation.
```

If the dataset or `infer` namespace is fresh, run the rendered `uv run usr --root ... namespace register infer --columns '...'` command before the first non-dry infer write-back.

### 3) Execute infer pressure job locally

```bash
uv run infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr # Execute local pressure job.
```

### 4) Verify USR write-back and events

```bash
uv run usr --root "$USR_ROOT" head "$DATASET_ID" -n 5 # Inspect top rows after write-back.
uv run usr --root "$USR_ROOT" events tail "$DATASET_ID" -n 20 # Confirm infer event stream updates.
```

### 5) Resume and prune infer output state

```bash
uv run infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr # Confirm resume behavior.
uv run infer prune --usr "$DATASET_ID" --usr-root "$USR_ROOT" # Archive infer namespace for reset.
uv run infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr # Re-run after prune reset.
```

### 6) Build ops runbook and inspect plan

```bash
# Initialize infer ops runbook for this workspace.
# Keep the runbook path under the workspace-scoped Ops logs tree.
export OPS_RUNBOOK="$WORKSPACE_ROOT/outputs/logs/ops/runbooks/infer-pressure.runbook.yaml" # Reuse one stable workspace-scoped Ops runbook path.
# Generate the infer workflow runbook at that workspace-scoped path.
uv run ops runbook init \
  --runbook "$OPS_RUNBOOK" \
  --workflow infer \
  --workspace-root "$WORKSPACE_ROOT" \
  --id infer_pressure_test \
  --no-notify

uv run ops runbook presets # Review workflow presets before execution.
uv run ops runbook plan --runbook "$OPS_RUNBOOK" # Generate execution plan.
```

### 7) Execute no-submit and submit paths

```bash
# Execute runbook in no-submit mode first.
uv run ops runbook execute \
  --runbook "$OPS_RUNBOOK" \
  --audit-json "$WORKSPACE_ROOT/outputs/logs/ops/audit/infer-pressure.audit.json" \
  --no-submit

qstat -u "$USER" # Inspect queue state before submit path.

# Execute runbook submit path after checks.
uv run ops runbook execute \
  --runbook "$OPS_RUNBOOK" \
  --audit-json "$WORKSPACE_ROOT/outputs/logs/ops/audit/infer-pressure-submit.audit.json" \
  --submit
```

### 8) Optional notify-enabled runbook variant

```bash
# Re-initialize runbook with notify stages enabled.
uv run ops runbook init \
  --runbook "$OPS_RUNBOOK" \
  --workflow infer \
  --workspace-root "$WORKSPACE_ROOT" \
  --id infer_pressure_test \
  --with-notify \
  --force
```

### 9) Contract reminder

- USR write-back column pattern is `infer__<model_id>__<job_id>__<out_id>`.
- Invalid or unreadable USR `records.parquet` fails fast during resume scan.
- Fresh USR test datasets must register exact infer output types before write-back.
- `evo2.embedding` accepts semantic `layer` values `mid` and `final` for the common pooled embedding checks.
