# BU SCC Batch + Notify Runbook (`dnadesign` platform workflows)

## At a glance

**Intent:** Run `dnadesign` workflows as SGE batch jobs on BU SCC with optional Notify webhooks and durable operator state.

**When to use:**
- Any run that should not execute in an interactive shell.
- Any run requiring restart safety, durable logs, and scheduler-managed resources.
- Any run requiring external status updates via Notify.

**Key boundary:** Notify consumes **USR `<dataset>/.events.log`** (JSONL), not DenseGen `outputs/meta/events.jsonl`.

Related docs:
- Install/bootstrap: [BU SCC Install bootstrap](bu_scc_install.md)
- Notify operator manual: [Notify USR events operator manual](../notify/usr_events.md)

---

## 1) Minimal `qsub` template (SGE)

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

BU SCC scheduling reference:
- <https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/>

---

## 2) Canonical job scripts

Use the templates in [HPC jobs README](jobs/README.md):
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

## 3) Job arrays (parameter sweeps)

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
uv run dense run -c "$CONFIG" --no-plot
```

BU SCC advanced batch reference:
- <https://www.bu.edu/tech/support/research/system-usage/running-jobs/advanced-batch/>

---

## 4) Notify patterns (BU SCC-safe)

### Mode A (recommended): dedicated Notify watcher batch job

Run Notify as a lightweight long-lived watcher with persistent cursor and spool paths.

```bash
uv run notify usr-events watch \
  --events /path/to/<usr_root>/<dataset>/.events.log \
  --cursor /projectnb/<project>/$USER/notify/<dataset>.cursor \
  --spool-dir /projectnb/<project>/$USER/notify/spool \
  --provider slack \
  --url-env DENSEGEN_WEBHOOK \
  --only-actions densegen_health,densegen_flush_failed,materialize \
  --follow
```

### Mode B: run Notify from transfer/login side for network-heavy delivery

BU provides a data transfer node (`scc-globus.bu.edu`) intended for high-bandwidth transfer workflows and download-node jobs (`qsub -l download`). Use this path for network-heavy tasks rather than compute nodes.

BU transfer reference:
- <https://www.bu.edu/tech/support/research/system-usage/transferring-files/cloud-applications/>

---

## 5) Large downloads and dataset/model transfer

Recommended pattern:
- Use BU Data Transfer Node for large HuggingFace model prefetch and large USR dataset push/pull.
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

# Example: model prefetch or dataset sync only (no compute-heavy work)
# uv run python scripts/prefetch_models.py
# uv run usr pull densegen/demo_hpc --remote bu-scc -y
QSUB
```

BU transfer-node constraints and intended usage:
- <https://www.bu.edu/tech/support/research/system-usage/transferring-files/cloud-applications/>

---

## 6) Package boundary reminders

- DenseGen runtime diagnostics: `outputs/meta/events.jsonl` (DenseGen-only diagnostics)
- Notify input: USR `<dataset>/.events.log`

Do not configure Notify against DenseGen runtime events.

---

## Planned improvements (TODO)

A future `dnadesign hpc wizard` / expanded `usr remotes wizard` could:
- generate and validate `USR_REMOTES_PATH`
- validate SSH connectivity and `rsync` availability
- preconfigure BU SCC hosts and base directories
- validate remote write/read paths and watcher cursor/spool locations

Status: planned; not implemented in this runbook.

---

Back: [HPC index](README.md)

Next: [BU SCC Quickstart](bu_scc_quickstart.md)

Next: [HPC job templates](jobs/README.md)
