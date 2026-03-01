## BU SCC job templates

These scripts are submit-ready templates for BU SCC SGE jobs:

- `densegen-cpu.qsub`: DenseGen CPU batch run
- `evo2-gpu-infer.qsub`: Evo2 GPU smoke/inference job shell
- `notify-watch.qsub`: Notify watcher for USR `.events.log`

### Quick start

Use project (`-P`) and runtime/config overrides at submit time.

```bash
qsub -P <project> docs/bu-scc/jobs/densegen-cpu.qsub
qsub -P <project> docs/bu-scc/jobs/evo2-gpu-infer.qsub
qsub -P <project> docs/bu-scc/jobs/notify-watch.qsub
```

### DenseGen CPU submissions

Default template run:

```bash
qsub -P <project> \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

`densegen-cpu.qsub` command defaults:
- validation: `uv run dense validate-config --probe-solver -c "$DENSEGEN_CONFIG"`
- run: `uv run dense run --no-plot -c "$DENSEGEN_CONFIG"`
- actor tags: `USR_ACTOR_TOOL=densegen`, `USR_ACTOR_RUN_ID=$JOB_ID.$SGE_TASK_ID`

Override command args at submit time when needed:
- `DENSEGEN_VALIDATE_ARGS` (example: `--probe-solver`)
- `DENSEGEN_RUN_ARGS` (example: `--resume --extend-quota 8 --no-plot`)

Resume + quota extension submission:

```bash
qsub -P <project> \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,DENSEGEN_RUN_ARGS='--resume --extend-quota 8 --no-plot' \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

DenseGen + GUROBI with explicit 16-slot cap:

```bash
qsub -P <project> \
  -pe omp 16 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

When using GUROBI, keep config aligned with scheduler slots:
- `densegen.solver.threads <= pe omp slots`
- set `densegen.solver.solver_attempt_timeout_seconds` for per-solve limits
- set `densegen.runtime.checkpoint_every` for flush/checkpoint cadence
- keep overall job runtime bounded via scheduler `-l h_rt=...`

### Evo2 GPU submissions

```bash
qsub -P <project> \
  -v CUDA_MODULE=cuda/<version>,GCC_MODULE=gcc/<version> \
  docs/bu-scc/jobs/evo2-gpu-infer.qsub
```

### Notify watcher submissions

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

# SCC TLS trust chain for HTTPS webhook delivery.
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

qsub -P <project> \
  -v NOTIFY_PROFILE="$NOTIFY_DIR/profile.json" \
  docs/bu-scc/jobs/notify-watch.qsub
```

Explicit env mode (no profile):

```bash
qsub -P <project> \
  -v NOTIFY_TOOL=densegen,NOTIFY_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,WEBHOOK_ENV=NOTIFY_WEBHOOK,NOTIFY_TLS_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem \
  docs/bu-scc/jobs/notify-watch.qsub
```

`notify-watch.qsub` mode selection:
- if `NOTIFY_PROFILE` is set, it runs `notify usr-events watch --profile ... --follow`
- otherwise it requires `EVENTS_PATH` or auto-resolves from `NOTIFY_TOOL` + `NOTIFY_CONFIG`
- it accepts future `.events.log` paths and uses `--wait-for-events` for run-before-events startup
- env mode still requires the webhook variable named by `WEBHOOK_ENV`
- set `NOTIFY_TLS_CA_BUNDLE` (or `SSL_CERT_FILE`) for HTTPS webhook delivery
- watcher polling cadence is configurable via `NOTIFY_POLL_INTERVAL_SECONDS` (default `1.0`)
- env mode requires a policy (`NOTIFY_POLICY`) unless resolver mode (`NOTIFY_TOOL` + `NOTIFY_CONFIG`) sets one
- env mode requires a namespace (`NOTIFY_NAMESPACE`) unless resolver mode (`NOTIFY_TOOL` + `NOTIFY_CONFIG`) sets one
- you can override policy defaults with explicit `NOTIFY_ACTIONS` and `NOTIFY_TOOLS`
- env-mode default state paths are namespaced: `outputs/notify/<namespace>/cursor` and `outputs/notify/<namespace>/spool`

### Local qsub-like smoke harness

Run a real watcher flow locally without SCC scheduler access:

```bash
# Env mode: explicit events/policy/namespace wiring.
docs/bu-scc/jobs/notify-watch-local-smoke.sh \
  --mode env \
  --repo-root "$(pwd)" \
  --workdir /tmp/notify-smoke-env

# Profile mode: setup profile + watcher profile flow.
docs/bu-scc/jobs/notify-watch-local-smoke.sh \
  --mode profile \
  --repo-root "$(pwd)" \
  --workdir /tmp/notify-smoke-profile
```

What it does:
- writes a terminal USR event into a local test events file
- starts a local HTTP capture endpoint (no external Slack posting)
- runs `docs/bu-scc/jobs/notify-watch.qsub` with real `uv run notify ...` commands
- verifies at least one success payload was delivered
- keeps all runtime artifacts under the specified `--workdir`

### Logs

Each script writes logs to:

- `outputs/logs/$JOB_NAME.$JOB_ID.out`

Tail logs:

```bash
tail -f outputs/logs/<job_name>.<job_id>.out
```

### Arrays

For arrays, use `qsub -t ...` and consume `SGE_TASK_ID` inside scripts.

Reference: [BU SCC Batch + Notify runbook: Job arrays](../batch-notify.md#5-job-arrays-parameter-sweeps)

### Edit vs submit-time overrides

- Prefer `qsub -P ... -v ... -l ... -pe ...` for run-specific values.
- Keep template scripts stable and versioned.
- Avoid one-off manual edits in production submissions.

### References

- Runbook: [BU SCC Batch + Notify runbook](../batch-notify.md)
- BU scheduler docs: <https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/>
