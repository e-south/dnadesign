# BU SCC Quickstart: dnadesign (Interactive -> Batch -> Notify)

## At a glance

**Intent:** Provide one copy/paste path from SCC login through first batch submissions and Notify setup.

**When to use:**
- You need DenseGen CPU jobs or Evo2 GPU jobs on BU SCC.
- You want scheduler-managed runtime/resource limits and durable logs.
- You want webhook monitoring from USR `.events.log`.

**Not for:**
- Deep dependency/debugging workflows (use [BU SCC Install bootstrap](bu_scc_install.md)).
- Non-BU clusters (use scheduler-specific docs for your platform).

## Contents

- [Connect to SCC](#connect-to-scc)
- [0) Scheduler constraints you must set](#0-scheduler-constraints-you-must-set)
- [1) Install uv](#1-install-uv)
- [2) Clone repo](#2-clone-repo)
- [3) Set environment and caches](#3-set-environment-and-caches)
- [4) Load modules](#4-load-modules)
- [5) Sync dependencies](#5-sync-dependencies)
- [6) Smoke tests](#6-smoke-tests)
- [7) Submit first jobs](#7-submit-first-jobs)
- [8) Add Notify](#8-add-notify)

## Connect to SCC

Run the quickstart on an SCC login shell:

```bash
ssh <BU_USERNAME>@scc1.bu.edu
# Some clients need:
# ssh -o PasswordAuthentication=no <BU_USERNAME>@scc1.bu.edu
```

Reference: [BU SCC SSH access](https://www.bu.edu/tech/support/research/system-usage/connect-scc/ssh/)

## 0) Scheduler constraints you must set

BU SCC scheduler is SGE (`qsub`) with `#$` directives.

For reproducible runs:
- pass `-P <project>` on submit
- request walltime (`h_rt`) and resources (`pe omp`, memory, GPU as needed)
- keep solver thread caps aligned with requested CPU slots

Important BU defaults/limits:
- if `h_rt` is omitted, default walltime is 12 hours
- runtime ceilings differ by job class (single/OMP, MPI, GPU)

References:
- [BU SCC submitting jobs](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/)
- [BU SCC interactive jobs](https://www.bu.edu/tech/support/research/system-usage/running-jobs/interactive-jobs/)

## 1) Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Details: [BU SCC Install bootstrap: Install uv](bu_scc_install.md#1-install-uv-once)

## 2) Clone repo

```bash
git clone https://github.com/e-south/dnadesign.git
cd dnadesign
```

Details: [BU SCC Install bootstrap: Clone the repository](bu_scc_install.md#2-clone-the-repository)

## 3) Set environment and caches

```bash
export UV_PROJECT_ENVIRONMENT="/projectnb/<project>/$USER/dnadesign/.venv"
export SCC_SCRATCH="${TMPDIR:-/scratch/$USER}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCC_SCRATCH/uv-cache}"
export HF_HOME="${HF_HOME:-/projectnb/<project>/$USER/huggingface}"
```

Details: [BU SCC Install bootstrap: Configure environment location and caches](bu_scc_install.md#3-configure-environment-location-and-caches)

## 4) Load modules

```bash
module purge
module avail cuda
module avail gcc
module load cuda/<version>
module load gcc/<version>
```

Details: [BU SCC Install bootstrap: Load toolchain modules](bu_scc_install.md#4-load-toolchain-modules)

## 5) Sync dependencies

```bash
uv python install 3.12
uv sync --locked
uv sync --locked --extra infer-evo2
```

Details: [BU SCC Install bootstrap: Sync dependencies](bu_scc_install.md#5-sync-dependencies)

## 6) Smoke tests

```bash
uv run python - <<'PY'
import torch
print('python ok')
print('cuda available', torch.cuda.is_available())
PY
```

For extended TE/FlashAttention/Evo2 checks:
[BU SCC Install bootstrap: Smoke tests](bu_scc_install.md#6-smoke-tests)

## 7) Submit first jobs

### 7.1 DenseGen CPU (template defaults)

```bash
qsub -P <project> \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml \
  docs/hpc/jobs/bu_scc_densegen_cpu.qsub
```

### 7.2 DenseGen + GUROBI (16-slot example)

Use this when your config sets `densegen.solver.backend: GUROBI`.

```bash
qsub -P <project> \
  -pe omp 16 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml \
  docs/hpc/jobs/bu_scc_densegen_cpu.qsub
```

In config, keep `densegen.solver.threads <= 16`.

### 7.3 Evo2 GPU inference

```bash
qsub -P <project> \
  -v CUDA_MODULE=cuda/<version>,GCC_MODULE=gcc/<version> \
  docs/hpc/jobs/bu_scc_evo2_gpu_infer.qsub
```

Template details and overrides:
[HPC job templates](jobs/README.md)

Operational guidance:
[BU SCC Batch + Notify runbook](bu_scc_batch_notify.md)

## 8) Add Notify

Notify watches USR `.events.log` only. It does not consume DenseGen `outputs/meta/events.jsonl`.

Recommended deployment: submit a dedicated watcher batch job.

Create a profile from workspace/run config before submitting watcher jobs:

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
```

Preferred mode (profile-driven, secure by default):

```bash
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

Secure onboarding and wizard flow:
[Notify USR events operator manual](../notify/usr_events.md)

Deployment patterns and transfer-node guidance:
[BU SCC Batch + Notify runbook](bu_scc_batch_notify.md)

---

Back: [HPC index](README.md)

Next: [BU SCC Install bootstrap](bu_scc_install.md)

Next: [BU SCC Batch + Notify runbook](bu_scc_batch_notify.md)
