# BU SCC End-to-End: DenseGen -> USR -> Notify

This runbook is a DenseGen-focused overlay for BU SCC.

Use it when you want to:
- submit DenseGen with SGE (`qsub`)
- write results into USR datasets
- watch USR `.events.log` with Notify
- sync datasets between SCC and local machines

Canonical platform references:
- [docs/hpc/bu_scc_quickstart.md](../../../../../docs/hpc/bu_scc_quickstart.md)
- [docs/hpc/bu_scc_batch_notify.md](../../../../../docs/hpc/bu_scc_batch_notify.md)
- [docs/hpc/jobs/README.md](../../../../../docs/hpc/jobs/README.md)

## Boundary contract

- Notify input: USR `<dataset>/.events.log`
- DenseGen `outputs/meta/events.jsonl`: diagnostics only

---

## 1) Prepare workspace config

Keep USR output inside workspace `outputs/` and set solver/runtime caps explicitly.

```yaml
densegen:
  run:
    id: bu_scc_demo
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
      dataset: densegen/bu_scc_demo
      chunk_size: 128
      health_event_interval_seconds: 60
```

---

## 2) Validate config and solver before submit

```bash
CONFIG="<dnadesign_repo>/src/dnadesign/densegen/workspaces/bu_scc_demo/config.yaml"
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
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/bu_scc_demo/config.yaml \
  docs/hpc/jobs/bu_scc_densegen_cpu.qsub
```

Keep `densegen.solver.threads <= pe omp slots`.

---

## 4) Inspect run and resolve USR event path

```bash
uv run dense inspect run --events --library -c <dnadesign_repo>/src/dnadesign/densegen/workspaces/bu_scc_demo/config.yaml
uv run dense inspect run --usr-events-path -c <dnadesign_repo>/src/dnadesign/densegen/workspaces/bu_scc_demo/config.yaml
```

Use the printed USR path for Notify.

---

## 5) Create Notify profile from workspace config

```bash
CONFIG="<dnadesign_repo>/src/dnadesign/densegen/workspaces/bu_scc_demo/config.yaml"
NOTIFY_DIR="<dnadesign_repo>/src/dnadesign/densegen/workspaces/bu_scc_demo/outputs/notify/densegen"

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
- flag-by-flag command rationale: [Notify command anatomy](../../../../../docs/notify/usr_events.md#command-anatomy-notify-setup-slack)

---

## 6) Deploy Notify watcher

Preferred pattern: dedicated watcher batch job (durable cursor/spool state).

Profile mode (recommended):

```bash
qsub -P <project> \
  -v NOTIFY_PROFILE="$NOTIFY_DIR/profile.json" \
  docs/hpc/jobs/bu_scc_notify_watch.qsub
```

Env mode (if you intentionally do not use a profile):

```bash
qsub -P <project> \
  -v NOTIFY_TOOL=densegen,NOTIFY_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/bu_scc_demo/config.yaml,WEBHOOK_ENV=NOTIFY_WEBHOOK \
  docs/hpc/jobs/bu_scc_notify_watch.qsub
```

If `EVENTS_PATH` is explicit in env mode, set `NOTIFY_POLICY` (`densegen`, `infer_evo2`, or `generic`).

For profile setup and diagnostics:
- `uv run notify setup slack ...`
- `uv run notify profile doctor ...`
- `uv run notify usr-events watch --dry-run ...`

Reference:
- [docs/hpc/bu_scc_batch_notify.md](../../../../../docs/hpc/bu_scc_batch_notify.md)

---

## 7) Sync USR dataset between SCC and local

Run on the destination machine:

```bash
export USR_REMOTES_PATH="$HOME/.config/dnadesign/usr-remotes.yaml"

uv run usr remotes wizard --preset bu-scc --name bu-scc --user <BU_USERNAME> --base-dir <dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/outputs/usr_datasets
uv run usr remotes doctor --remote bu-scc

uv run usr diff densegen/bu_scc_demo bu-scc
uv run usr pull densegen/bu_scc_demo bu-scc -y
```

For deeper sync operations:
- [../../../usr/docs/operations/sync.md](../../../usr/docs/operations/sync.md)

---

@e-south
