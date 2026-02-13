# DenseGen -> USR -> Notify on HPC

This runbook explains the operational stack on HPC:

DenseGen (generator) -> USR (canonical dataset + event log) -> Notify (webhook delivery)

For BU SCC policy and template authority, use:
- [docs/bu-scc/quickstart.md](../../../../../docs/bu-scc/quickstart.md)
- [docs/bu-scc/batch-notify.md](../../../../../docs/bu-scc/batch-notify.md)
- [docs/bu-scc/jobs/README.md](../../../../../docs/bu-scc/jobs/README.md)

## Boundary contract

- DenseGen runtime telemetry: `outputs/meta/events.jsonl` (DenseGen diagnostics)
- USR mutation stream: `<usr_root>/<dataset>/.events.log` (Notify input)

Notify consumes USR `.events.log` only.

---

## 1) Prerequisites

- Scheduler: SGE (`qsub` + `#$` directives)
- Job scripts that use modules: `#!/bin/bash -l`
- Batch jobs should declare project/runtime/resources explicitly (`-P`, `h_rt`, `pe omp`, memory)

References:
- [BU SCC submitting jobs](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/)
- [BU SCC interactive jobs](https://www.bu.edu/tech/support/research/system-usage/running-jobs/interactive-jobs/)

---

## 2) Single-run submission (DenseGen)

Use the canonical template and pass per-run values at submit time:

```bash
qsub -P <project> \
  -pe omp 16 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/demo_hpc/config.yaml \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

Before submission:

```bash
uv run dense validate-config --probe-solver -c <dnadesign_repo>/src/dnadesign/densegen/workspaces/demo_hpc/config.yaml
```

---

## 3) Array submission pattern

If each task has a separate config, submit an array and consume `SGE_TASK_ID`:

```bash
#!/bin/bash -l
#$ -P <project>
#$ -N densegen_array
#$ -t 1-16
#$ -l h_rt=04:00:00
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -j y
#$ -o outputs/logs/$JOB_NAME.$JOB_ID.$TASK_ID.out

set -euo pipefail

CONFIG="<dnadesign_repo>/src/dnadesign/densegen/workspaces/task_${SGE_TASK_ID}/config.yaml"
uv run dense validate-config --probe-solver -c "$CONFIG"
uv run dense run --no-plot -c "$CONFIG"
```

Submit with:

```bash
qsub run_densegen_array.sh
```

---

## 4) Resolve USR event path for Notify

After DenseGen completes:

```bash
uv run dense inspect run --usr-events-path -c <dnadesign_repo>/src/dnadesign/densegen/workspaces/demo_hpc/config.yaml
```

Use this path in Notify. Do not use `outputs/meta/events.jsonl`.

---

## 5) Create Notify profile from workspace config

Use resolver mode to avoid manual event-path copying:

```bash
CONFIG="<dnadesign_repo>/src/dnadesign/densegen/workspaces/demo_hpc/config.yaml"
NOTIFY_DIR="<dnadesign_repo>/src/dnadesign/densegen/workspaces/demo_hpc/outputs/notify/densegen"

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
- `--config` must point to the run/workspace `config.yaml`, not a repo-root config
- profile schema is v2 and stores `events_source` (`tool`, `config`) for re-resolution
- setup may point at a future `.events.log`; watcher should start with `--wait-for-events`
- flag-by-flag command rationale: [Notify command anatomy](../../../../../docs/notify/usr-events.md#command-anatomy-notify-setup-slack)

---

## 6) Deploy Notify watcher

Recommended: dedicated watcher batch job for durable delivery.

Profile mode (recommended):

```bash
qsub -P <project> \
  -v NOTIFY_PROFILE="$NOTIFY_DIR/profile.json" \
  docs/bu-scc/jobs/notify-watch.qsub
```

Env mode (if you intentionally do not use a profile):

```bash
qsub -P <project> \
  -v NOTIFY_TOOL=densegen,NOTIFY_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/demo_hpc/config.yaml,WEBHOOK_ENV=NOTIFY_WEBHOOK \
  docs/bu-scc/jobs/notify-watch.qsub
```

If `EVENTS_PATH` is explicit in env mode, set `NOTIFY_POLICY` (`densegen`, `infer_evo2`, or `generic`).

For setup checks, run `profile doctor` and watcher `--dry-run` before live follow mode.

---

## 7) Spool and drain recovery

If webhook delivery fails, keep spool enabled and drain later:

```bash
uv run notify spool drain --profile "$NOTIFY_DIR/profile.json"
```

---

@e-south
