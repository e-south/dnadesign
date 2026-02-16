# BU SCC End-to-End: DenseGen -> USR -> Notify

This runbook is a DenseGen-focused overlay for BU SCC.

Use it when you want to:
- submit DenseGen with SGE (`qsub`)
- write results into USR datasets
- watch USR `.events.log` with Notify
- sync datasets between SCC and local machines

Canonical platform references:
- [docs/bu-scc/quickstart.md](../../../../../docs/bu-scc/quickstart.md)
- [docs/bu-scc/batch-notify.md](../../../../../docs/bu-scc/batch-notify.md)
- [docs/bu-scc/jobs/README.md](../../../../../docs/bu-scc/jobs/README.md)

## Boundary contract

- Notify input: USR `<dataset>/.events.log`
- DenseGen `outputs/meta/events.jsonl`: diagnostics only

---

## 1) Prepare workspace config

Keep USR output inside workspace `outputs/` and set solver/runtime caps explicitly.

Create a workspace first:

```bash
uv run dense workspace init --id <workspace_id> --from-workspace demo_sampling_baseline --copy-inputs --output-mode usr
```

```yaml
densegen:
  run:
    id: <workspace_id>
    root: "."

  solver:
    backend: GUROBI
    strategy: iterate
    threads: 16
    time_limit_seconds: 60

  runtime:
    max_seconds_per_plan: 21600

  output:
    targets: [usr]
    usr:
      root: outputs/usr_datasets
      dataset: densegen/<workspace_id>
      chunk_size: 128
      health_event_interval_seconds: 60
```

---

## 2) Validate config and solver before submit

```bash
CONFIG="<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace_id>/config.yaml"
uv run dense validate-config --probe-solver -c "$CONFIG"
uv run dense inspect config --probe-solver -c "$CONFIG"
```

---

## 3) Submit DenseGen batch job

Submit via the canonical template and pass project/resources at submit time:

```bash
qsub -P <project> \
  -pe omp 16 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG="$CONFIG" \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

Keep `densegen.solver.threads <= pe omp slots`.

---

## 4) Inspect run and resolve USR event path

```bash
uv run dense inspect run --events --library -c "$CONFIG"
uv run dense inspect run --usr-events-path -c "$CONFIG"
```

Use the printed USR path for Notify.

---

## 5) Create Notify profile from workspace config

```bash
NOTIFY_DIR="<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace_id>/outputs/notify/densegen"

uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source auto \
  --policy densegen
```

Flag expectations:
- `--config` is the workspace/run config path, not a repo-root config
- profile schema is v2 and stores `events_source` (`tool`, `config`) to prevent drift
- setup can run before `.events.log` exists
- flag-by-flag command rationale: [Notify command anatomy](../../../../../docs/notify/usr-events.md#command-anatomy-notify-setup-slack)

---

## 6) Deploy Notify watcher

Preferred pattern: dedicated watcher batch job (durable cursor/spool state).

Profile mode (recommended):

```bash
qsub -P <project> \
  -v NOTIFY_PROFILE="$NOTIFY_DIR/profile.json" \
  docs/bu-scc/jobs/notify-watch.qsub
```

Env mode (if you intentionally do not use a profile):

```bash
qsub -P <project> \
  -v NOTIFY_TOOL=densegen,NOTIFY_CONFIG="$CONFIG",WEBHOOK_ENV=NOTIFY_WEBHOOK \
  docs/bu-scc/jobs/notify-watch.qsub
```

If `EVENTS_PATH` is explicit in env mode, set `NOTIFY_POLICY` (`densegen`, `infer_evo2`, or `generic`).

For profile setup and diagnostics:
- `uv run notify setup slack ...`
- `uv run notify profile doctor ...`
- `uv run notify usr-events watch --dry-run ...`

Reference:
- [docs/bu-scc/batch-notify.md](../../../../../docs/bu-scc/batch-notify.md)

---

## 7) Sync USR dataset between SCC and local

Run on the destination machine:

```bash
export USR_REMOTES_PATH="$HOME/.config/dnadesign/usr-remotes.yaml"

uv run usr remotes wizard --preset bu-scc --name bu-scc --user <BU_USERNAME> --base-dir <dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/outputs/usr_datasets
uv run usr remotes doctor --remote bu-scc

uv run usr diff densegen/<workspace_id> bu-scc
uv run usr pull densegen/<workspace_id> bu-scc -y
```

For deeper sync operations:
- [../../../usr/docs/operations/sync.md](../../../usr/docs/operations/sync.md)

---

@e-south
