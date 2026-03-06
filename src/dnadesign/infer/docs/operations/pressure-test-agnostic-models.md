## Infer Pressure Test: Agnostic Model Namespaces + USR Write-Back

This guide pressure-tests infer as a model-agnostic extraction engine with explicit namespace contracts and USR write-back.

### Objective

- run extract jobs that produce multiple outputs (for example logits + log-likelihood ratio variants),
- ensure outputs attach to USR with infer-prefixed namespaced columns:
  - `infer__<model_id>__<job_id>__<out_id>`
- support both standalone CLI usage and ops runbook orchestration paths.

### Safety Posture

- Use `--dry-run` and `--no-submit` first.
- Use read-only scheduler checks before submit (`qstat -u "$USER"`).
- Keep `ingest.root` explicit for workspace and cluster runs.

## Path A: Standalone CLI Pressure Test

### 1) Prepare variables

```bash
export WORKSPACE_ROOT="$PWD/src/dnadesign/infer/workspaces/test_stress_ethanol"
export INFER_CONFIG="$WORKSPACE_ROOT/config.yaml"
export USR_ROOT="/projectnb/dunlop/esouth/outputs/usr_datasets"
export DATASET_ID="test_stress_ethanol"
mkdir -p "$WORKSPACE_ROOT/outputs/logs/ops/audit"
```

### 2) Seed config from example

```bash
cp src/dnadesign/infer/docs/operations/examples/pressure_test_infer_config.yaml "$INFER_CONFIG"
```

### 3) Contract preflight

```bash
uv run infer validate config --config "$INFER_CONFIG"
uv run infer run --config "$INFER_CONFIG" --dry-run
```

### 4) Execute pressure test (local CLI)

```bash
uv run infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr
```

### 5) Verify USR state and events

```bash
uv run usr --root "$USR_ROOT" head "$DATASET_ID" -n 5
uv run usr --root "$USR_ROOT" events tail "$DATASET_ID" -n 20
```

## Path B: Ops Runbook Pressure Test (Cluster-Friendly)

### 1) Scaffold infer runbook

```bash
uv run ops runbook init \
  --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml" \
  --workflow infer \
  --workspace-root "$WORKSPACE_ROOT" \
  --id infer_pressure_test \
  --no-notify
```

### 2) Plan and inspect command graph

```bash
uv run ops runbook precedents
uv run ops runbook plan --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml"
```

### 3) Execute no-submit pressure pass

```bash
uv run ops runbook execute \
  --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml" \
  --audit-json "$WORKSPACE_ROOT/outputs/logs/ops/audit/infer-pressure.audit.json" \
  --no-submit
```

### 4) Submit only after preflight passes

```bash
qstat -u "$USER"
uv run ops runbook execute \
  --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml" \
  --audit-json "$WORKSPACE_ROOT/outputs/logs/ops/audit/infer-pressure-submit.audit.json" \
  --submit
```

### 5) Optional notify-enabled variant

If you need notify smoke/watcher phases, first configure webhook secret wiring (for example by setting a readable `NOTIFY_WEBHOOK_FILE` path), then re-init with notify enabled:

```bash
uv run ops runbook init \
  --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml" \
  --workflow infer \
  --workspace-root "$WORKSPACE_ROOT" \
  --id infer_pressure_test \
  --with-notify \
  --force
```

## Path C: Independent Ad-Hoc CLI Extract

```bash
uv run infer extract \
  --model-id evo2_7b \
  --fn evo2.log_likelihood \
  --format float \
  --usr "$DATASET_ID" \
  --usr-root "$USR_ROOT" \
  --field sequence \
  --write-back
```

Use this path for focused troubleshooting when runbook orchestration is unnecessary.
