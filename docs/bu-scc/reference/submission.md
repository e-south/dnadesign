## BU SCC Submission Reference (`dnadesign`)

Command and resource defaults for BU SCC runs.

Use this with:
- `docs/bu-scc/setup/quickstart.md`
- `docs/bu-scc/runbooks/batch-notify.md`
- `docs/bu-scc/jobs/README.md`

### Core rules

- Always set `-P <project>`.
- Always set `h_rt` explicitly.
- Prefer `h_rt <= 12:00:00` when feasible to improve scheduling access on shared nodes.
- Keep `densegen.solver.threads <= -pe omp <slots>`.
- On shared nodes, start DenseGen at `-pe omp 12` and tune based on measured throughput.
- DenseGen runs are CPU jobs (no GPU request).
- DenseGen submit commands must pass `DENSEGEN_RUN_ARGS` with exactly one of `--fresh` or `--resume`.
- Evo2 `evo2_7b` runs require GPU resources (`-l gpus=1 -l gpu_c=8.9`).
- Evo2 `evo2_20b` runs require Hopper-class-or-newer GPU resources
  (`-l gpus=1 -l gpu_c=9.0`) plus enough VRAM for the run config.
- Notify watches USR `.events.log`, not DenseGen `outputs/meta/events.jsonl`.
- Prefer `WEBHOOK_FILE` for watcher submits so webhook values are not exposed in `qstat -j` env metadata.
- Notify/USR contract requires structured events with `event_version`; legacy logs are rejected.
- Use transfer-node queue (`-l download`) for large model/data transfers.
- OnDemand policy: sessions requesting >12h and/or extra resources count toward the 5 active-session limit.

### Task to resource mapping

| Task | Queue type | Starter resources | Notes |
| --- | --- | --- | --- |
| DenseGen interactive smoke/debug | interactive (`qrsh`) CPU | `-l h_rt=01:00:00 -pe omp 8 -l mem_per_core=8G` | Use for short validation and debugging only. |
| DenseGen batch (CBC/GUROBI) | batch CPU | `-l h_rt=08:00:00 -pe omp 12 -l mem_per_core=8G` | Scale slots with plan complexity; keep solver threads aligned. |
| Notify watcher | batch CPU | `-l h_rt=24:00:00 -pe omp 1 -l mem_per_core=2G` | Low-footprint long-running watcher. |
| Evo2 7B inference/smoke | batch GPU | `-l h_rt=04:00:00 -pe omp 4 -l mem_per_core=8G -l gpus=1 -l gpu_c=8.9` | Default SCC lane for `evo2_7b`. |
| Evo2 20B inference | batch GPU | `-l h_rt=04:00:00 -pe omp 4 -l mem_per_core=8G -l gpus=1 -l gpu_c=9.0` | Model-fit floor for `evo2_20b`. If the current `.venv` is family-pinned, also pass an exact selector such as `-l gpu_t=RTXP6000 -l gpu_c=12.0` for the visible SCC Blackwell lane. |
| Permuter closed-loop evaluate | batch GPU when using Evo2 evaluators | `-l h_rt=04:00:00 -pe omp 4 -l mem_per_core=8G -l gpus=1 -l gpu_c=8.9` | Use `permuter-evaluate.qsub`; omit GPU requests only for placeholder or CPU-only evaluators. |
| Large downloads / model prefetch / dataset transfer | transfer-node | `-l download -l h_rt=24:00:00 -pe omp 1` | Do not run compute-heavy tasks here. |

### Copy/paste commands

#### 1) Interactive CPU shell (1 hour)

```bash
qrsh -P <project> -l h_rt=01:00:00 -pe omp 8 -l mem_per_core=8G -cwd -now n
```

#### 2) DenseGen CPU batch submit

```bash
qsub -P <project> \
  -pe omp 12 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,DENSEGEN_RUN_ARGS='--fresh --no-plot' \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

#### 3) DenseGen config preflight (before long runs)

```bash
uv run dense validate-config --probe-solver -c <config.yaml>
uv run dense inspect config --probe-solver -c <config.yaml>
```

#### 4) Evo2 GPU submit

```bash
qsub -P <project> \
  -v INFER_CONFIG=<dnadesign_repo>/src/dnadesign/infer/workspaces/<workspace>/config.yaml,CUDA_MODULE=cuda/<version>,GCC_MODULE=gcc/<version> \
  docs/bu-scc/jobs/evo2-gpu-infer.qsub
```

For `evo2_20b`, prefer the ops runbook path so the resource declaration stays attached to the run config. If you submit directly, add `-l gpus=1 -l gpu_c=9.0` for the generic model floor, or `-l gpus=1 -l gpu_c=12.0 -l gpu_t=RTXP6000` for the current SCC Blackwell lane.

#### 5) Permuter closed-loop evaluate submit

```bash
qsub -P <project> \
  -v PERMUTER_WORKSPACE=<dnadesign_repo>/src/dnadesign/permuter/workspaces/<workspace>/config.yaml,PERMUTER_REF=<ref_name>,PERMUTER_RUN_FIRST=1,PERMUTER_EVALUATE_ARGS='--with llr:evo2_llr:log_likelihood_ratio' \
  docs/bu-scc/jobs/permuter-evaluate.qsub
```

Use the GPU defaults above for Evo2-backed evaluators. For placeholder or CPU-only evaluators, submit the same wrapper with CPU resources and omit `-l gpus`.

#### 6) Notify profile setup + watcher submit

```bash
CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml
NOTIFY_DIR="<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/outputs/notify/densegen"
WEBHOOK_FILE="$HOME/.config/dnadesign/notify_webhook.secret"

mkdir -p "$(dirname "$WEBHOOK_FILE")"
touch "$WEBHOOK_FILE"
chmod 600 "$WEBHOOK_FILE"
uv run notify setup webhook \
  --secret-source file \
  --secret-ref "file://$WEBHOOK_FILE"

# Preflight resolver: fails fast if config is not wired for USR .events.log output.
uv run notify setup resolve-events --tool densegen --config "$CONFIG"

uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source file \
  --secret-ref "file://$WEBHOOK_FILE" \
  --no-store-webhook \
  --policy densegen

qsub -P <project> \
  -v NOTIFY_PROFILE="$NOTIFY_DIR/profile.json",WEBHOOK_FILE="$WEBHOOK_FILE" \
  docs/bu-scc/jobs/notify-watch.qsub
```

#### 7) Transfer-node job for large artifacts

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

### Monitoring quick commands

```bash
qstat -u "$USER"
qstat -j <job_id>
tail -f outputs/logs/<job_name>.<job_id>.out
```

### Submission pressure quick check

```bash
qstat -u "$USER" | awk '
  $1 ~ /^[0-9]+$/ {
    state=$5
    if (state ~ /r/) running++
    if (state ~ /q/) queued++
    if (state ~ /Eqw/) eqw++
  }
  END { printf "running_jobs=%d queued_jobs=%d eqw_jobs=%d\n", running, queued, eqw }
'
```

- if `running_jobs > 3`, avoid burst submits and confirm before adding jobs
- choose `qsub -t` arrays for independent workloads
- choose `-hold_jid` chains for ordered stages
- respect the queue and do not skip the line
