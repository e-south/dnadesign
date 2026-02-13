# BU SCC Ops Cheat Sheet (`dnadesign`)

Pragmatic command + resource defaults for BU SCC runs.

Use this with:
- `docs/hpc/bu_scc_quickstart.md`
- `docs/hpc/bu_scc_batch_notify.md`
- `docs/hpc/jobs/README.md`

## Core rules

- Always set `-P <project>`.
- Always set `h_rt` explicitly.
- Prefer `h_rt <= 12:00:00` when feasible to improve scheduling access on shared nodes.
- Keep `densegen.solver.threads <= -pe omp <slots>`.
- On shared nodes, prefer `-pe omp` sizes from BU guidance: `1-4`, `8`, `16`, `28`, or `36`.
- DenseGen runs are CPU jobs (no GPU request).
- Evo2 runs require GPU resources (`-l gpus=1 -l gpu_c=8.9`).
- Notify watches USR `.events.log`, not DenseGen `outputs/meta/events.jsonl`.
- Notify/USR contract requires structured events with `event_version`; legacy logs are rejected.
- Use transfer-node queue (`-l download`) for large model/data transfers.
- OnDemand policy: sessions requesting >12h and/or extra resources count toward the 5 active-session limit.

## Task to resource mapping

| Task | Queue type | Starter resources | Notes |
| --- | --- | --- | --- |
| DenseGen interactive smoke/debug | interactive (`qrsh`) CPU | `-l h_rt=01:00:00 -pe omp 8 -l mem_per_core=8G` | Use for short validation and debugging only. |
| DenseGen batch (CBC/GUROBI) | batch CPU | `-l h_rt=08:00:00 -pe omp 16 -l mem_per_core=8G` | Scale slots with plan complexity; keep solver threads aligned. |
| Notify watcher | batch CPU | `-l h_rt=24:00:00 -pe omp 1 -l mem_per_core=2G` | Low-footprint long-running watcher. |
| Evo2 inference/smoke | batch GPU | `-l h_rt=04:00:00 -pe omp 4 -l mem_per_core=8G -l gpus=1 -l gpu_c=8.9` | Load matching CUDA/GCC modules in job script. |
| Large downloads / model prefetch / dataset transfer | transfer-node | `-l download -l h_rt=24:00:00 -pe omp 1` | Do not run compute-heavy tasks here. |

## Copy/paste commands

### 1) Interactive CPU shell (1 hour)

```bash
qrsh -P <project> -l h_rt=01:00:00 -pe omp 8 -l mem_per_core=8G -cwd -now n
```

### 2) DenseGen CPU batch submit

```bash
qsub -P <project> \
  -pe omp 16 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml \
  docs/hpc/jobs/bu_scc_densegen_cpu.qsub
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
  docs/hpc/jobs/bu_scc_evo2_gpu_infer.qsub
```

### 5) Notify profile setup + watcher submit

```bash
CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml
NOTIFY_DIR="<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/outputs/notify/densegen"

# Preflight resolver: fails fast if config is not wired for USR .events.log output.
uv run notify setup resolve-events --tool densegen --config "$CONFIG"

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
