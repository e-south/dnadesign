# BU SCC Quickstart: dnadesign (Interactive -> Batch -> Notify)

## At a glance

**Intent:** Follow one end-to-end operator path for BU SCC from environment bootstrap through first batch runs and Notify watcher setup.

**When to use:**
- You want one copy/paste sequence that gets you to a working SCC run.
- You need DenseGen CPU jobs or Evo2 GPU jobs on BU SCC.
- You want restart-safe webhook monitoring from USR `.events.log`.

**Not for:**
- Deep troubleshooting of build/runtime failures (use the install and batch docs).
- Generic non-BU clusters (use package-specific HPC docs for your scheduler).

## Contents

- [Connect to SCC](#connect-to-scc)
- [0) Node and GPU requirements](#0-node-and-gpu-requirements)
- [1) Install uv](#1-install-uv)
- [2) Clone repo](#2-clone-repo)
- [3) Set environment and caches](#3-set-environment-and-caches)
- [4) Load modules](#4-load-modules)
- [5) Sync dependencies](#5-sync-dependencies)
- [6) Smoke tests](#6-smoke-tests)
- [7) First real run](#7-first-real-run)
- [8) Add Notify](#8-add-notify)

## Connect to SCC

Run the quickstart on an SCC login shell:

```bash
ssh <BU_USERNAME>@scc1.bu.edu
# Some clients need:
# ssh -o PasswordAuthentication=no <BU_USERNAME>@scc1.bu.edu
```

Reference: [BU SCC SSH access](https://www.bu.edu/tech/support/research/system-usage/connect-scc/ssh/)

## 0) Node and GPU requirements

- BU SCC scheduler is SGE (`qsub`) with `#$` directives.
- Evo2 GPU jobs should request `-l gpus=1 -l gpu_c=8.9`.
- For detailed constraints and references, see:
  [BU SCC Install bootstrap: Node and GPU requirements](bu_scc_install.md#0-node-and-gpu-requirements)

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

## 7) First real run

### 7.1 DenseGen CPU batch run

```bash
qsub -P <project> -v DENSEGEN_CONFIG=/abs/path/to/config.yaml docs/hpc/jobs/bu_scc_densegen_cpu.qsub
```

### 7.2 Evo2 GPU inference batch run

```bash
qsub -P <project> -v CUDA_MODULE=cuda/<version>,GCC_MODULE=gcc/<version> docs/hpc/jobs/bu_scc_evo2_gpu_infer.qsub
```

Template details and overrides:
[HPC job templates](jobs/README.md)

Operational guidance:
[BU SCC Batch + Notify runbook](bu_scc_batch_notify.md)

## 8) Add Notify

Notify watches USR `.events.log` only. It does not consume DenseGen `outputs/meta/events.jsonl`.

Minimal watcher invocation:

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

Secure onboarding and wizard flow:
[Notify USR events operator manual](../notify/usr_events.md)

SCC watcher deployment patterns:
[BU SCC Batch + Notify runbook: Notify patterns](bu_scc_batch_notify.md#4-notify-patterns-bu-scc-safe)

---

Back: [HPC index](README.md)

Next: [BU SCC Install bootstrap](bu_scc_install.md)

Next: [BU SCC Batch + Notify runbook](bu_scc_batch_notify.md)
