## infer end-to-end pressure-test demo (usr + ops + notify)

This demo executes infer in an end-to-end pressure-test loop that can run standalone or via ops orchestration.

### Objective

1. Validate infer config and dry-run contracts.
2. Execute infer extraction pressure path.
3. Verify USR dataset state after write-back.
4. Execute ops runbook in no-submit and submit modes.
5. Optionally enable notify phases for delivery smoke checks.

### 1) Prepare workspace and config

```bash
export WORKSPACE_ROOT="$PWD/src/dnadesign/infer/workspaces/test_stress_ethanol"
export INFER_CONFIG="$WORKSPACE_ROOT/config.yaml"
export USR_ROOT="/projectnb/dunlop/esouth/outputs/usr_datasets"
export DATASET_ID="test_stress_ethanol"
mkdir -p "$WORKSPACE_ROOT/outputs/logs/ops/audit"
cp src/dnadesign/infer/docs/operations/examples/pressure_test_infer_config.yaml "$INFER_CONFIG"
```

### 2) Validate infer contract surface

```bash
uv run infer validate config --config "$INFER_CONFIG"
uv run infer run --config "$INFER_CONFIG" --dry-run
```

### 3) Execute infer pressure job locally

```bash
uv run infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr
```

### 4) Verify USR write-back and events

```bash
uv run usr --root "$USR_ROOT" head "$DATASET_ID" -n 5
uv run usr --root "$USR_ROOT" events tail "$DATASET_ID" -n 20
```

### 5) Build ops runbook and inspect plan

```bash
uv run ops runbook init \
  --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml" \
  --workflow infer \
  --workspace-root "$WORKSPACE_ROOT" \
  --id infer_pressure_test \
  --no-notify

uv run ops runbook precedents
uv run ops runbook plan --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml"
```

### 6) Execute no-submit and submit paths

```bash
uv run ops runbook execute \
  --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml" \
  --audit-json "$WORKSPACE_ROOT/outputs/logs/ops/audit/infer-pressure.audit.json" \
  --no-submit

qstat -u "$USER"

uv run ops runbook execute \
  --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml" \
  --audit-json "$WORKSPACE_ROOT/outputs/logs/ops/audit/infer-pressure-submit.audit.json" \
  --submit
```

### 7) Optional notify-enabled runbook variant

```bash
uv run ops runbook init \
  --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml" \
  --workflow infer \
  --workspace-root "$WORKSPACE_ROOT" \
  --id infer_pressure_test \
  --with-notify \
  --force
```

### 8) Contract reminder

- USR write-back column pattern is `infer__<model_id>__<job_id>__<out_id>`.
- Invalid or unreadable USR `records.parquet` fails fast during resume scan.
