# DenseGen -> USR -> Notify on HPC

This runbook is for the operational stack:

DenseGen (generator) -> USR (canonical store plus `.events.log`) -> Notify (webhook delivery).

Boundary contract:
- DenseGen runtime diagnostics: `outputs/meta/events.jsonl` (DenseGen-only)
- USR mutation events: `<usr_root>/<dataset>/.events.log` (Notify input)

Notify reads USR `.events.log` only.

---

## 1) BU SCC quick prerequisites

BU SCC uses SGE (`qsub`) for batch jobs and `#$` directives in job scripts.
Reference: [BU SCC submitting jobs](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/)

Use `#!/bin/bash -l` in batch scripts when you need `module` commands.
Reference: [BU SCC submitting jobs](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/)

Connection and auth:
- SSH login host example: `scc1.bu.edu`
- Some clients need `-o PasswordAuthentication=no` for Duo workflows
Reference: [BU SCC SSH access](https://www.bu.edu/tech/support/research/system-usage/connect-scc/ssh/)

Operational placement:
- Run compute workloads with `qsub`
- Run Notify watcher on a login-node shell or SCC OnDemand shell
Reference: [SCC OnDemand](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/)

---

## 2) SGE job script (single run)

```bash
#!/bin/bash -l
#$ -N densegen_demo
#$ -cwd
#$ -j y
#$ -o densegen_demo.$JOB_ID.out
#$ -l h_rt=04:00:00
#$ -pe omp 4

set -euo pipefail

CONFIG="/project/$USER/densegen_runs/demo_hpc/config.yaml"

export USR_ACTOR_TOOL=densegen
export USR_ACTOR_RUN_ID="${JOB_ID}_${SGE_TASK_ID:-0}"

uv run dense validate-config --probe-solver -c "$CONFIG"
uv run dense run --no-plot -c "$CONFIG"
```

Common SGE env vars available in jobs: `JOB_ID`, `JOB_NAME`, `NSLOTS`, `SGE_TASK_ID`.
Reference: [BU SCC submitting jobs](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/)

Submit:

```bash
qsub run_densegen.sh
```

---

## 3) SGE array job pattern

```bash
#!/bin/bash -l
#$ -N densegen_array
#$ -cwd
#$ -j y
#$ -o densegen_array.$JOB_ID.$TASK_ID.out
#$ -t 1-16
#$ -l h_rt=04:00:00
#$ -pe omp 4

set -euo pipefail

CONFIG_DIR="/project/$USER/densegen_runs"
CONFIG="$CONFIG_DIR/run_${SGE_TASK_ID}/config.yaml"

export USR_ACTOR_TOOL=densegen
export USR_ACTOR_RUN_ID="${JOB_ID}_${SGE_TASK_ID}"

uv run dense run --no-plot -c "$CONFIG"
```

Reference: [BU SCC batch script examples](https://www.bu.edu/tech/support/research/system-usage/running-jobs/batch-script-examples/)

---

## 4) Where Notify runs

Run Notify separately from the batch job, tailing the USR event log produced by DenseGen:

```bash
uv run notify usr-events watch \
  --events /project/$USER/densegen_runs/demo_hpc/outputs/usr_datasets/densegen/demo_hpc/.events.log \
  --cursor /project/$USER/densegen_runs/notify/demo_hpc.cursor \
  --provider slack \
  --url-env DENSEGEN_WEBHOOK \
  --only-actions densegen_health,densegen_flush_failed,materialize \
  --follow
```

If login-shell lifecycle is unstable, run in SCC OnDemand shell.
Reference: [SCC OnDemand](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/)

---

## 5) Dataset transfer back to local

Datasets are not Git-tracked; transfer them with USR remotes sync:

```bash
export USR_REMOTES_PATH="$HOME/.config/dnadesign/usr-remotes.yaml"

uv run usr remotes wizard \
  --preset bu-scc \
  --name bu-scc \
  --user "$USER" \
  --host scc1.bu.edu \
  --base-dir /project/$USER/densegen_runs/demo_hpc/outputs/usr_datasets

uv run usr remotes doctor --remote bu-scc
uv run usr diff densegen/demo_hpc --remote bu-scc
uv run usr pull densegen/demo_hpc --remote bu-scc -y
```

For transfer-heavy workflows, BU provides a data transfer node (`scc-globus.bu.edu`) and `qsub -l download` workflows.
Reference: [BU SCC file transfer guide](https://www.bu.edu/tech/support/research/system-usage/transferring-files/cloud-applications/)

---

## Common pitfalls

- Watching the wrong event log: Notify consumes USR `.events.log`, not DenseGen `outputs/meta/events.jsonl`.
- Running DenseGen from login nodes: submit compute runs with `qsub`.
- Missing registry: USR output requires a valid `registry.yaml` at the USR root.
- Implicit config selection: always pass `-c` in HPC/CI scripts.

---

## Appendix: SLURM variant (non-BU clusters)

Use this only on non-BU clusters that run Slurm.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=densegen
#SBATCH --array=0-31
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

export USR_ACTOR_TOOL=densegen
export USR_ACTOR_RUN_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

uv run dense run --no-plot -c /path/to/config.yaml
```
