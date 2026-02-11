# DenseGen -> USR -> Notify on HPC

This runbook covers the operational stack on HPC:

DenseGen (generator) -> USR (canonical store + event log) -> Notify (webhook delivery)

For BU SCC platform specifics, also read:
- [docs/hpc/bu_scc_install.md](../../../../../docs/hpc/bu_scc_install.md)
- [docs/hpc/bu_scc_batch_notify.md](../../../../../docs/hpc/bu_scc_batch_notify.md)

## Boundary contract

- DenseGen runtime telemetry: `outputs/meta/events.jsonl` (DenseGen diagnostics)
- USR mutation stream: `<usr_root>/<dataset>/.events.log` (Notify input)

Notify consumes USR `.events.log` only.

---

## 1) HPC prerequisites (BU SCC quick notes)

- BU SCC batch system: SGE (`qsub` + `#$` directives)
- Use `#!/bin/bash -l` for module-aware job scripts
- Run compute jobs with `qsub`; run Notify watchers in login/OnDemand shells

References:
- [BU SCC submitting jobs](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/)
- [BU SCC SSH access](https://www.bu.edu/tech/support/research/system-usage/connect-scc/ssh/)
- [SCC OnDemand](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/)

---

## 2) Example SGE job script (single run)

`run_densegen.sh`:

```bash
#!/bin/bash -l
#$ -N densegen_demo
#$ -cwd
#$ -j y
#$ -o densegen_demo.$JOB_ID.out
#$ -l h_rt=04:00:00
#$ -pe omp 4

set -euo pipefail

# Point to your workspace config.
CONFIG="/project/$USER/densegen_runs/demo_hpc/config.yaml"

# Validate config + solver before running.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Run generation (no plotting in batch job).
uv run dense run --no-plot -c "$CONFIG"
```

Submit:

```bash
# Submit the batch job.
qsub run_densegen.sh
```

---

## 3) Example SGE array pattern

`run_densegen_array.sh`:

```bash
#!/bin/bash -l
#$ -N densegen_array
#$ -cwd
#$ -j y
#$ -o densegen_array.$JOB_ID.$TASK_ID.out
#$ -t 1-16

set -euo pipefail

# Template workspace config; task index can select per-task config if needed.
CONFIG_TEMPLATE="/project/$USER/densegen_runs/task_${SGE_TASK_ID}/config.yaml"

# Validate each task config.
uv run dense validate-config --probe-solver -c "$CONFIG_TEMPLATE"

# Run each task.
uv run dense run --no-plot -c "$CONFIG_TEMPLATE"
```

Submit:

```bash
# Submit array job.
qsub run_densegen_array.sh
```

---

## 4) Resolve USR event path for Notify

After DenseGen run completes:

```bash
# Print exact USR .events.log path for this run.
uv run dense inspect run --usr-events-path -c /project/$USER/densegen_runs/demo_hpc/config.yaml
```

Use this path in Notify. Do not use `outputs/meta/events.jsonl`.

---

## 5) Configure and run Notify watcher

```bash
# Export webhook URL through env var (example name).
export DENSEGEN_WEBHOOK="https://example.com/webhook"

# Build a Notify profile against the USR event stream.
uv run notify profile wizard \
  --profile /project/$USER/densegen_runs/demo_hpc/outputs/notify.profile.json \
  --provider slack \
  --events /project/$USER/densegen_runs/demo_hpc/outputs/usr_datasets/densegen/demo_hpc/.events.log \
  --cursor /project/$USER/densegen_runs/demo_hpc/outputs/notify.cursor \
  --spool-dir /project/$USER/densegen_runs/demo_hpc/outputs/notify_spool \
  --secret-source env \
  --url-env DENSEGEN_WEBHOOK \
  --only-tools densegen \
  --only-actions densegen_health,densegen_flush_failed,materialize

# Validate profile wiring.
uv run notify profile doctor --profile /project/$USER/densegen_runs/demo_hpc/outputs/notify.profile.json

# Dry-run first.
uv run notify usr-events watch --profile /project/$USER/densegen_runs/demo_hpc/outputs/notify.profile.json --dry-run

# Run live watcher.
uv run notify usr-events watch --profile /project/$USER/densegen_runs/demo_hpc/outputs/notify.profile.json --follow
```

---

## 6) Spool and drain (network-safe pattern)

If webhook delivery fails, keep `--spool-dir` enabled and drain later from a stable network host.

```bash
# Drain previously spooled payload files.
uv run notify spool drain --profile /project/$USER/densegen_runs/demo_hpc/outputs/notify.profile.json
```

---

@e-south
