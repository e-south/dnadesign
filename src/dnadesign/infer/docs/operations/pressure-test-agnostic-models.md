## Infer Pressure Test: Agnostic Model Namespaces + USR Write-Back

This guide pressure-tests infer as a model-agnostic extraction engine with explicit namespace contracts and USR write-back.

For a full walkthrough, use the [end-to-end demo tutorial](../tutorials/demo_pressure_test_usr_ops_notify.md).

For deterministic SCC GPU environment setup before pressure runs, use the [SCC Evo2 GPU environment runbook](scc-evo2-gpu-uv-runbook.md).

### Objective

- run extract jobs that produce multiple outputs (for example logits + log-likelihood ratio variants),
- ensure outputs attach to USR with infer-prefixed namespaced columns:
  - `infer__<model_id>__<job_id>__<out_id>`
- support local CLI execution and ops runbook orchestration.

### Safety Posture

- Use `--dry-run` and `--no-submit` first.
- Use read-only scheduler checks before submit (`qstat -u "$USER"`).
- Keep `ingest.root` explicit for workspace and cluster runs.
- For fresh USR test datasets, register exact infer output types before write-back:
  - pooled Evo2 likelihoods are `float64`
  - pooled Evo2 logits are `list<float64>`
- Use `infer validate usr-registry --config ...` to derive the exact namespace registration command from the active infer config.

## Ordered procedure

### 1) Prepare workspace and variables

```bash
uv run infer workspace init --id test_stress_ethanol --profile usr-pressure
export WORKSPACE_ROOT="$PWD/src/dnadesign/infer/workspaces/test_stress_ethanol"
export INFER_CONFIG="$WORKSPACE_ROOT/config.yaml"
export USR_ROOT="/projectnb/dunlop/esouth/outputs/usr_datasets"
export DATASET_ID="test_stress_ethanol"
```

### 2) Contract preflight

```bash
uv run infer validate config --config "$INFER_CONFIG"
uv run infer validate usr-registry --config "$INFER_CONFIG"
uv run infer run --config "$INFER_CONFIG" --dry-run
```

### 3) Execute local pressure test

```bash
uv run infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr
```

### 4) Verify USR state and events

```bash
uv run usr --root "$USR_ROOT" head "$DATASET_ID" -n 5
uv run usr --root "$USR_ROOT" events tail "$DATASET_ID" -n 20
```

### 5) Resume and prune the infer namespace when needed

Second runs on the same dataset should resume and skip completed infer rows:

```bash
uv run infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr
```

To reset only infer outputs for the dataset, archive the infer overlay and rerun:

```bash
uv run infer prune --usr "$DATASET_ID" --usr-root "$USR_ROOT"
uv run infer run --config "$INFER_CONFIG" --job pressure_evo2_logits_llr
```

### 6) Initialize infer ops runbook

```bash
uv run ops runbook init \
  --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml" \
  --workflow infer \
  --workspace-root "$WORKSPACE_ROOT" \
  --id infer_pressure_test \
  --no-notify
```

### 7) Plan and execute no-submit preflight

```bash
uv run ops runbook precedents
uv run ops runbook plan --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml"
uv run ops runbook execute \
  --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml" \
  --audit-json "$WORKSPACE_ROOT/outputs/logs/ops/audit/infer-pressure.audit.json" \
  --no-submit
```

### 8) Submit after preflight passes

```bash
qstat -u "$USER"
uv run ops runbook execute \
  --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml" \
  --audit-json "$WORKSPACE_ROOT/outputs/logs/ops/audit/infer-pressure-submit.audit.json" \
  --submit
```

### 9) Enable notify in the same runbook when needed

First configure webhook secret wiring (for example by setting a readable `NOTIFY_WEBHOOK_FILE` path), then re-initialize the same runbook:

```bash
uv run ops runbook init \
  --runbook "$WORKSPACE_ROOT/infer-pressure.runbook.yaml" \
  --workflow infer \
  --workspace-root "$WORKSPACE_ROOT" \
  --id infer_pressure_test \
  --with-notify \
  --force
```

### 10) Run focused ad-hoc extract checks when isolating issues

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

For embedding pressure checks in config-driven runs, use semantic layer names instead of raw Evo2 block names when possible:

```yaml
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
