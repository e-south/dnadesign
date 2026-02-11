# BU SCC job templates

These scripts are submit-ready templates for BU SCC SGE jobs:

- `bu_scc_densegen_cpu.qsub`: DenseGen CPU batch run
- `bu_scc_evo2_gpu_infer.qsub`: Evo2 GPU smoke/inference job shell
- `bu_scc_notify_watch.qsub`: Notify watcher for USR `.events.log`

## Quick start

Use project (`-P`) and runtime/config overrides at submit time.

```bash
qsub -P <project> docs/hpc/jobs/bu_scc_densegen_cpu.qsub
qsub -P <project> docs/hpc/jobs/bu_scc_evo2_gpu_infer.qsub
qsub -P <project> docs/hpc/jobs/bu_scc_notify_watch.qsub
```

## DenseGen CPU submissions

Default template run:

```bash
qsub -P <project> \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml \
  docs/hpc/jobs/bu_scc_densegen_cpu.qsub
```

DenseGen + GUROBI with explicit 16-slot cap:

```bash
qsub -P <project> \
  -pe omp 16 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml \
  docs/hpc/jobs/bu_scc_densegen_cpu.qsub
```

When using GUROBI, keep config aligned with scheduler slots:
- `densegen.solver.threads <= pe omp slots`
- set `densegen.solver.time_limit_seconds` for per-solve limits
- set `densegen.runtime.max_seconds_per_plan` for per-plan runtime limits

## Evo2 GPU submissions

```bash
qsub -P <project> \
  -v CUDA_MODULE=cuda/<version>,GCC_MODULE=gcc/<version> \
  docs/hpc/jobs/bu_scc_evo2_gpu_infer.qsub
```

## Notify watcher submissions

Preferred mode (profile-driven, secure by default):

```bash
CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml
NOTIFY_DIR="<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/outputs/notify/densegen"

uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source auto \
  --policy densegen

qsub -P <project> \
  -v NOTIFY_PROFILE="$NOTIFY_DIR/profile.json" \
  docs/hpc/jobs/bu_scc_notify_watch.qsub
```

Explicit env mode (no profile):

```bash
qsub -P <project> \
  -v NOTIFY_TOOL=densegen,NOTIFY_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,WEBHOOK_ENV=NOTIFY_WEBHOOK \
  docs/hpc/jobs/bu_scc_notify_watch.qsub
```

`bu_scc_notify_watch.qsub` mode selection:
- if `NOTIFY_PROFILE` is set, it runs `notify usr-events watch --profile ... --follow`
- otherwise it requires `EVENTS_PATH` or auto-resolves from `NOTIFY_TOOL` + `NOTIFY_CONFIG`
- it accepts future `.events.log` paths and uses `--wait-for-events` for run-before-events startup
- env mode still requires the webhook variable named by `WEBHOOK_ENV`
- watcher polling cadence is configurable via `NOTIFY_POLL_INTERVAL_SECONDS` (default `1.0`)
- env mode requires a policy (`NOTIFY_POLICY`) unless resolver mode (`NOTIFY_TOOL` + `NOTIFY_CONFIG`) sets one
- env mode requires a namespace (`NOTIFY_NAMESPACE`) unless resolver mode (`NOTIFY_TOOL` + `NOTIFY_CONFIG`) sets one
- you can override policy defaults with explicit `NOTIFY_ACTIONS` and `NOTIFY_TOOLS`
- env-mode default state paths are namespaced: `outputs/notify/<namespace>/cursor` and `outputs/notify/<namespace>/spool`

## Logs

Each script writes logs to:

- `outputs/logs/$JOB_NAME.$JOB_ID.out`

Tail logs:

```bash
tail -f outputs/logs/<job_name>.<job_id>.out
```

## Arrays

For arrays, use `qsub -t ...` and consume `SGE_TASK_ID` inside scripts.

Reference: [BU SCC Batch + Notify runbook: Job arrays](../bu_scc_batch_notify.md#5-job-arrays-parameter-sweeps)

## Edit vs submit-time overrides

- Prefer `qsub -P ... -v ... -l ... -pe ...` for run-specific values.
- Keep template scripts stable and versioned.
- Avoid one-off manual edits in production submissions.

## References

- Runbook: [BU SCC Batch + Notify runbook](../bu_scc_batch_notify.md)
- BU scheduler docs: <https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/>
