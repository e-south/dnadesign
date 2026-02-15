# BU SCC Batch + Notify Runbook (`dnadesign` platform workflows)

## At a glance

**Intent:** Run `dnadesign` workflows as SGE batch jobs on BU SCC with restart-safe logs and optional Notify webhooks.

**When to use:**
- Any run that should not execute in an interactive shell.
- Any run that needs explicit runtime/resource control.
- Any run that needs event-driven webhook notifications.

**Key boundary:** Notify consumes **USR `<dataset>/.events.log`** (JSONL), not DenseGen `outputs/meta/events.jsonl`.

Related docs:
- Install/bootstrap: [BU SCC Install bootstrap](bu_scc_install.md)
- End-to-end quick path: [BU SCC Quickstart](bu_scc_quickstart.md)
- Job templates: [HPC jobs README](jobs/README.md)
- Notify operator manual: [Notify USR events operator manual](../notify/usr_events.md)

---

## 1) Scheduler rules that matter on BU SCC

BU SCC uses SGE (`qsub` + `#$` directives).

Operational rules to keep in mind:
- Always request runtime and resources explicitly for reproducible scheduling.
- If `h_rt` is omitted, BU defaults jobs to a 12-hour walltime.
- Runtime ceilings differ by job class (for example: single/OMP up to 720h, MPI up to 120h, GPU up to 48h).
- `-pe omp N` requests CPU slots for OpenMP-style jobs; keep solver thread counts aligned with `N`.

BU references:
- Submitting jobs: <https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/>
- Interactive jobs: <https://www.bu.edu/tech/support/research/system-usage/running-jobs/interactive-jobs/>
- Advanced batch: <https://www.bu.edu/tech/support/research/system-usage/running-jobs/advanced-batch/>

---

## 2) Minimal `qsub` template (SGE)

```bash
#!/bin/bash -l
#$ -P <project>
#$ -N <job_name>
#$ -l h_rt=04:00:00
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -j y
#$ -o outputs/logs/$JOB_NAME.$JOB_ID.out

set -euo pipefail
```

---

## 3) Canonical job scripts

Use versioned templates from [HPC jobs README](jobs/README.md):
- [DenseGen CPU batch template](jobs/bu_scc_densegen_cpu.qsub)
- [Evo2 GPU template](jobs/bu_scc_evo2_gpu_infer.qsub)
- [Notify watcher template](jobs/bu_scc_notify_watch.qsub)

Submit examples:

```bash
qsub -P <project> docs/hpc/jobs/bu_scc_densegen_cpu.qsub
qsub -P <project> docs/hpc/jobs/bu_scc_evo2_gpu_infer.qsub
qsub -P <project> docs/hpc/jobs/bu_scc_notify_watch.qsub
```

---

## 4) DenseGen + GUROBI: resource and time caps

For DenseGen GUROBI runs, cap runtime at three layers:

1. Scheduler-level cap (SGE):
- `-pe omp <N>` for CPU slots
- `-l h_rt=<HH:MM:SS>` for job walltime
- `-l mem_per_core=<size>` for per-slot memory

2. Solver-level cap (DenseGen config):
- `densegen.solver.threads` (set `<= N`)
- `densegen.solver.time_limit_seconds` (per-solve cap)

3. Runtime policy cap (DenseGen config):
- `densegen.runtime.max_seconds_per_plan` (per-plan run cap)

Example config fragment:

```yaml
densegen:
  solver:
    backend: GUROBI
    strategy: iterate
    threads: 16
    time_limit_seconds: 60
  runtime:
    max_seconds_per_plan: 21600
```

Example submit command:

```bash
qsub -P <project> \
  -pe omp 16 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml \
  docs/hpc/jobs/bu_scc_densegen_cpu.qsub
```

Before long runs:

```bash
uv run dense validate-config --probe-solver -c <dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml
uv run dense inspect config --probe-solver -c <dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml
```

DenseGen config references:
- Solver schema and constraints: `src/dnadesign/densegen/docs/reference/config.md`

---

## 5) Job arrays (parameter sweeps)

```bash
#!/bin/bash -l
#$ -P <project>
#$ -N <array_job>
#$ -t 1-32
#$ -l h_rt=04:00:00
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -j y
#$ -o outputs/logs/$JOB_NAME.$JOB_ID.$TASK_ID.out

set -euo pipefail

TASK_ID="${SGE_TASK_ID}"
CONFIG="/project/<project>/<user>/runs/config_${TASK_ID}.yaml"
uv run dense validate-config --probe-solver -c "$CONFIG"
uv run dense run --no-plot -c "$CONFIG"
```

---

## 6) Notify deployment patterns (BU SCC-safe)

### Mode A (recommended): dedicated watcher batch job

Use a lightweight watcher job with profile-driven wiring.
This keeps webhook source, filters, cursor, and spool settings decoupled from submit commands.

Template submit example:

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

### Mode B: explicit env wiring (no profile)

Use this when you intentionally do not want a profile file.
If `EVENTS_PATH` is omitted, the script resolves it from `NOTIFY_TOOL` and `NOTIFY_CONFIG`.
If `EVENTS_PATH` is explicit, set `NOTIFY_POLICY` (`densegen`, `infer_evo2`, or `generic`)
and `NOTIFY_NAMESPACE` (for example, `densegen`).

```bash
qsub -P <project> \
  -v NOTIFY_TOOL=densegen,NOTIFY_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,WEBHOOK_ENV=NOTIFY_WEBHOOK \
  docs/hpc/jobs/bu_scc_notify_watch.qsub
```

### Mode C: short-lived validation in login/OnDemand shell

Use this only for setup checks (`profile doctor`, `--dry-run`) or troubleshooting.
For durable delivery, keep watchers in batch.

---

## 7) Large downloads and dataset/model transfer

Recommended pattern:
- Use BU Data Transfer Node for large model prefetch and large USR dataset push/pull.
- Keep compute-heavy work off transfer-only nodes.

Example transfer-node request:

```bash
qsub -l download <<'QSUB'
#!/bin/bash -l
#$ -P <project>
#$ -N transfer_job
#$ -l h_rt=24:00:00
#$ -pe omp 1
#$ -j y
#$ -o outputs/logs/transfer.$JOB_ID.out

set -euo pipefail

# Example: transfer-only operations
# uv run python scripts/prefetch_models.py
# uv run usr pull densegen/demo_hpc --remote bu-scc -y
QSUB
```

Transfer reference:
- <https://www.bu.edu/tech/support/research/system-usage/transferring-files/cloud-applications/>

---

## 8) Package boundary reminders

- DenseGen runtime diagnostics: `outputs/meta/events.jsonl` (DenseGen-only diagnostics)
- Notify input: USR `<dataset>/.events.log`

Do not configure Notify against DenseGen runtime events.

---

Back: [HPC index](README.md)

Next: [BU SCC Quickstart](bu_scc_quickstart.md)

Next: [HPC job templates](jobs/README.md)
