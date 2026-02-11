# BU SCC Install: `dnadesign` Interactive Bootstrap (CPU + Evo2 GPU)

## At a glance

**Intent:** One-time environment bootstrap for running `dnadesign` on BU SCC for CPU workflows (for example DenseGen with CBC/Gurobi) and Linux GPU workflows (Evo2 inference stack).

**When to use:**
- You need SCC modules, solvers, or GPUs to run `dnadesign` workloads.
- You want a reproducible `uv`-managed environment for both interactive sessions and batch jobs.
- You want to verify toolchain health before running long jobs.

**Not for:**
- Long-running production jobs (use [BU SCC Batch + Notify runbook](bu_scc_batch_notify.md)).
- Operational monitoring setup (use [BU SCC Batch + Notify runbook](bu_scc_batch_notify.md) and [Notify USR events operator manual](../notify/usr_events.md)).
- Large model or dataset transfers on compute nodes (use BU data transfer node workflows).

## Choose your path

- Interactive install + smoke tests (this doc).
- Batch usage + Notify setup ([BU SCC Batch + Notify runbook](bu_scc_batch_notify.md)).

Run this from an SCC login shell:
- `ssh <BU_USERNAME>@scc1.bu.edu`
- SSH reference: <https://www.bu.edu/tech/support/research/system-usage/connect-scc/ssh/>

---

## 0) Node and GPU requirements

Evo2 requirements are Linux + CUDA 12.1+, cuDNN 9.3+, Python 3.12, and GPU Compute Capability 8.9+ for FP8 workflows:
- Evo2 repo: <https://github.com/ArcInstitute/evo2>

BU SCC GPU requests support explicit GPU count and capability constraints. Use:
- `-l gpus=1`
- `-l gpu_c=8.9`

Useful SCC commands:
- `qgpus` to inspect available GPU classes
- `qsub` for batch submission

BU SCC scheduler and GPU docs:
- <https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/>

---

## 1) Install `uv` (once)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2) Clone the repository

```bash
git clone https://github.com/e-south/dnadesign.git
cd dnadesign
```

Optional reset:

```bash
rm -rf .venv
uv cache clean
```

---

## 3) Configure environment location and caches

Use a persistent project location for the environment so expensive GPU builds are reusable.

```bash
# Choose a persistent location for the project environment (recommended)
export UV_PROJECT_ENVIRONMENT="/projectnb/<your_project>/$USER/dnadesign/.venv"

# Use job-local scratch for caches/temp when available; fall back to /scratch
export SCC_SCRATCH="${TMPDIR:-/scratch/$USER}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCC_SCRATCH/uv-cache}"

# Network robustness for cluster outbound calls
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"

# Optional: model cache on project storage
export HF_HOME="${HF_HOME:-/projectnb/<your_project>/$USER/huggingface}"

printf 'UV_PROJECT_ENVIRONMENT=%s\n' "$UV_PROJECT_ENVIRONMENT"
printf 'UV_CACHE_DIR=%s\n' "$UV_CACHE_DIR"
printf 'SCC_SCRATCH=%s\n' "$SCC_SCRATCH"
```

---

## 4) Load toolchain modules

Use SCC modules that match your target workflow.

```bash
module purge
module avail cuda
module avail gcc

module load cuda/<version>
module load gcc/<version>

export CC="$(which gcc)"
export CXX="$(which g++)"
export CUDAHOSTCXX="$(which g++)"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"

nvcc --version
gcc --version
g++ --version
```

Use `#!/bin/bash -l` in batch scripts when relying on `module` commands:
- <https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/>

---

## 5) Sync dependencies

```bash
uv python install 3.12
uv sync --locked
uv sync --locked --extra infer-evo2
```

`infer-evo2` is Linux-only and intended for SCC GPU nodes.

---

## 6) Smoke tests

### 6.1 Quick environment check

```bash
uv run python - <<'PY'
import sys
import torch
print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('torch.cuda', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0), 'cc', torch.cuda.get_device_capability(0))
PY
```

### 6.2 Extended TE/FlashAttention/Evo2 check

```bash
uv run python - <<'PY'
import torch

if not torch.cuda.is_available():
    print('CUDA not available on this node; skipping GPU import checks.')
    raise SystemExit(0)

from transformer_engine.pytorch import Linear
from flash_attn import flash_attn_func
import evo2

dev = torch.device('cuda:0')
x = torch.randn(128, 128, device=dev, dtype=torch.float16)
lin = Linear(128, 128).to(dev).half()
_ = lin(x)

q = torch.randn(1, 4, 4, 64, device=dev, dtype=torch.float16)
k = torch.randn(1, 4, 4, 64, device=dev, dtype=torch.float16)
v = torch.randn(1, 4, 4, 64, device=dev, dtype=torch.float16)
out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
print('flash-attn output shape', tuple(out.shape))
print('evo2 import ok', getattr(evo2, '__version__', '(no __version__)'))
PY
```

---

## 7) Next step

For long jobs, arrays, Notify watchers, and transfer-node workflows, use:
- [BU SCC Batch + Notify runbook](bu_scc_batch_notify.md)

---

## Troubleshooting and build throttles

Use conservative build caps when TE/flash-attn builds fail due to memory pressure:

```bash
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"
```

If CUDA headers are not discovered during Transformer Engine builds:

```bash
export NVTE_CUDA_INCLUDE_PATH="${NVTE_CUDA_INCLUDE_PATH:-$CUDA_HOME/include}"
```

### Optional deep diagnostics (full import + extension checks)

```bash
uv run python - <<'PY'
import os, sys, glob, subprocess
import importlib.util as iu
import importlib.metadata as im

def v(dist):
    try:
        return im.version(dist)
    except im.PackageNotFoundError:
        return None

def sh(cmd):
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()

print("Python:", sys.version.split()[0])
print("Exe:", sys.executable)
print("Platform:", sys.platform)
print("UV_CACHE_DIR:", os.environ.get("UV_CACHE_DIR"))
print("UV_PROJECT_ENVIRONMENT:", os.environ.get("UV_PROJECT_ENVIRONMENT"))
print("TMPDIR:", os.environ.get("TMPDIR"))

for dist in ["torch", "torchvision", "torchaudio", "transformer-engine", "flash-attn", "evo2"]:
    print(f"{dist}:", v(dist) or "(not installed)")

def ldd_torch_so():
    spec = iu.find_spec("torch")
    if not spec or not spec.submodule_search_locations:
        return
    torch_dir = list(spec.submodule_search_locations)[0]
    cands = glob.glob(os.path.join(torch_dir, "_C*.so"))
    if not cands:
        return
    so = cands[0]
    print("\\n[diag] torch extension:", so)
    _rc, out = sh(["ldd", so])
    print(out)

try:
    import torch
except Exception as e:
    print("\\n[FAIL] torch import failed:", repr(e))
    ldd_torch_so()
    raise SystemExit(2)

print("\\n[OK] torch:", torch.__version__, "torch CUDA:", torch.version.cuda)
print("[OK] cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("[WARN] CUDA not available (likely not on a GPU node).")
    raise SystemExit(0)

dev = torch.device("cuda:0")
print("[OK] GPU:", torch.cuda.get_device_name(0), "CC:", torch.cuda.get_device_capability(0))

x = torch.randn(128, 128, device=dev, dtype=torch.float16)
y = x @ x
print("[OK] torch matmul:", tuple(y.shape))

try:
    from transformer_engine.pytorch import Linear
    lin = Linear(128, 128).to(dev).half()
    z = lin(x)
    print("[OK] transformer-engine Linear:", tuple(z.shape))
except Exception as e:
    print("[FAIL] transformer-engine:", repr(e))
    raise SystemExit(3)

try:
    from flash_attn import flash_attn_func
    q = torch.randn(1, 4, 4, 64, device=dev, dtype=torch.float16)
    k = torch.randn(1, 4, 4, 64, device=dev, dtype=torch.float16)
    v_ = torch.randn(1, 4, 4, 64, device=dev, dtype=torch.float16)
    out = flash_attn_func(q, k, v_, dropout_p=0.0, causal=False)
    print("[OK] flash-attn:", tuple(out.shape), "any NaN:", bool(torch.isnan(out).any().item()))
except Exception as e:
    print("[FAIL] flash-attn:", repr(e))
    raise SystemExit(4)

try:
    import evo2
    print("[OK] evo2 imports:", getattr(evo2, "__version__", "(no __version__)"))
except Exception as e:
    print("[FAIL] evo2:", repr(e))
    raise SystemExit(5)

print("\\nAll deep diagnostics passed.")
PY
```

### Optional Evo2 callable check

```bash
uv run python - <<'PY'
import torch
from evo2 import Evo2

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available on this node.")

evo2_model = Evo2("evo2_7b")
sequence = "ACGT"
input_ids = torch.tensor(
    evo2_model.tokenizer.tokenize(sequence),
    dtype=torch.int,
).unsqueeze(0).to("cuda:0")
outputs, _ = evo2_model(input_ids)
logits = outputs[0]
print("Logits:", logits)
print("Shape (batch, length, vocab):", logits.shape)
PY
```

---

Back: [HPC index](README.md)

Next: [BU SCC Quickstart](bu_scc_quickstart.md)

Next: [BU SCC Batch + Notify runbook](bu_scc_batch_notify.md)
