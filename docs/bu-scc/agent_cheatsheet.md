# BU SCC Agent Cheat Sheet (`dnadesign`)

Pragmatic command + resource defaults for BU SCC runs.

Use this with:
- `docs/bu-scc/quickstart.md`
- `docs/bu-scc/batch-notify.md`
- `docs/bu-scc/jobs/README.md`

## Core rules

- Always set `-P <project>`.
- Always set `h_rt` explicitly.
- Keep `densegen.solver.threads <= -pe omp <slots>`.
- DenseGen runs are CPU jobs (no GPU request).
- Evo2 runs require GPU resources (`-l gpus=1 -l gpu_c=8.9`).
- Notify watches USR `.events.log`, not DenseGen `outputs/meta/events.jsonl`.
- Use transfer-node queue (`-l download`) for large model/data transfers.

## Task to resource mapping

| Task | Queue type | Starter resources | Notes |
| --- | --- | --- | --- |
| DenseGen interactive smoke/debug | interactive (`qrsh`) CPU | `-l h_rt=01:00:00 -pe omp 8 -l mem_per_core=8G` | Use for short validation and debugging only. |
| DenseGen batch (CBC/GUROBI) | batch CPU | `-l h_rt=08:00:00 -pe omp 16 -l mem_per_core=8G` | Scale slots with plan complexity; keep solver threads aligned. |
| DenseGen watcher only (Notify) | batch CPU | `-l h_rt=24:00:00 -pe omp 1 -l mem_per_core=2G` | Low-footprint long-running watcher. |
| Evo2 inference/smoke | batch GPU | `-l h_rt=04:00:00 -pe omp 4 -l mem_per_core=8G -l gpus=1 -l gpu_c=8.9` | Load matching CUDA/GCC modules in job script. |
| Large downloads / model prefetch / dataset transfer | transfer-node | `-l download -l h_rt=24:00:00 -pe omp 1` | Do not run compute-heavy tasks here. |

## Copy/paste commands

### 1) Interactive CPU shell (1 hour)

```bash
qrsh -P <project> -l h_rt=01:00:00 -pe omp 8 -l mem_per_core=8G -cwd -V -now n
```

### 2) DenseGen CPU batch submit

```bash
qsub -P <project> \
  -pe omp 16 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

### 3) DenseGen config preflight (before long runs)

```bash
uv run dense validate-config --probe-solver -c <config.yaml>
uv run dense inspect config --probe-solver -c <config.yaml>
```

### 4) Evo2 GPU submit

```bash
qsub -P <project> \
  -v CUDA_MODULE=cuda/<version>,GCC_MODULE=gcc/<version> \
  docs/bu-scc/jobs/evo2-gpu-infer.qsub
```

### 5) Notify profile setup + watcher submit

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
  docs/bu-scc/jobs/notify-watch.qsub
```

### 6) Transfer-node job for large artifacts

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
# transfer-only operations here
QSUB
```

## Monitoring quick commands

```bash
qstat -u "$USER"
qstat -j <job_id>
tail -f outputs/logs/<job_name>.<job_id>.out
```
