## BU SCC Install: `dnadesign` Interactive Bootstrap (CPU + Evo2 GPU)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-07

### Purpose

**Intent:** One-time environment bootstrap for running `dnadesign` on BU SCC for CPU workflows (for example DenseGen with CBC/Gurobi) and Linux GPU workflows (Evo2 inference stack).

**When to use:**
- You need SCC modules, solvers, or GPUs to run `dnadesign` workloads.
- You want a reproducible `uv`-managed environment for both interactive sessions and batch jobs.
- You want to verify toolchain health before running long jobs.

**Not for:**
- Long-running production jobs (use [BU SCC Batch + Notify runbook](batch-notify.md)).
- Operational monitoring setup (use [BU SCC Batch + Notify runbook](batch-notify.md) and [Notify USR events operator manual](../notify/usr-events.md)).
- Large model or dataset transfers on compute nodes (use BU data transfer node workflows).

### Scope

- This doc covers environment setup and validation for CPU and Evo2 GPU workflows.
- For batch submission and Notify operations, use [BU SCC Batch + Notify runbook](batch-notify.md).

Run this from an SCC login shell:
- `ssh <BU_USERNAME>@scc1.bu.edu`
- SSH reference: <https://www.bu.edu/tech/support/research/system-usage/connect-scc/ssh/>

---

### 0) Node and GPU requirements

Evo2 requirements are Linux + CUDA 12.1+, cuDNN 9.3+, Python 3.12, and Hopper-class GPUs for FP8 workflows:
- Evo2 repo: <https://github.com/ArcInstitute/evo2>

BU SCC GPU requests support explicit GPU count and capability constraints. Use:
- `-l gpus=1`
- `-l gpu_c=8.9` for `evo2_7b`
- `-l gpu_c=9.0` for `evo2_20b`

Useful SCC commands:
- `qgpus` to inspect available GPU classes
- `qsub` for batch submission

BU SCC scheduler and GPU docs:
- <https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/>

---

### 1) Install `uv` (once)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### 2) Clone the repository

```bash
mkdir -p /project/<your_project>/$USER
cd /project/<your_project>/$USER
git clone https://github.com/e-south/dnadesign.git
cd dnadesign
```

Reset the local environment only when needed:

```bash
rm -rf .venv
uv cache clean
```

---

### 3) Configure environment location and caches

Use one canonical environment location in the repository root and explicit cache roots.

```bash
# Keep one canonical environment in the repo root.
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"

# Keep uv cache outside the repo.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/project/<your_project>/$USER/cache/uv}"

# Network timeout and retry settings for outbound package fetches
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"

# Infer model-cache policy:
# - 7B infer smoke and routine infer runs use /project.
# - 20B Hopper runs also use /project, but with a separate cache root.
# - 40B is optional and not part of the default SCC path.
export HF_HOME_7B="${HF_HOME_7B:-/project/<your_project>/$USER/cache/huggingface/evo2_7b}"
export HF_HOME_20B="${HF_HOME_20B:-/project/<your_project>/$USER/cache/huggingface/evo2_20b}"
export TARGET_MODEL_ID="${TARGET_MODEL_ID:-evo2_7b}"
case "$TARGET_MODEL_ID" in
  evo2_7b) export HF_HOME="${HF_HOME:-$HF_HOME_7B}" ;;
  evo2_20b) export HF_HOME="${HF_HOME:-$HF_HOME_20B}" ;;
  *)
    printf 'Unsupported TARGET_MODEL_ID=%s\n' "$TARGET_MODEL_ID" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

printf 'UV_PROJECT_ENVIRONMENT=%s\n' "$UV_PROJECT_ENVIRONMENT"
printf 'UV_CACHE_DIR=%s\n' "$UV_CACHE_DIR"
printf 'TARGET_MODEL_ID=%s\n' "$TARGET_MODEL_ID"
printf 'HF_HOME=%s\n' "$HF_HOME"
printf 'HF_HOME_7B=%s\n' "$HF_HOME_7B"
printf 'HF_HOME_20B=%s\n' "$HF_HOME_20B"
printf 'HF_HUB_CACHE=%s\n' "$HF_HUB_CACHE"
printf 'HUGGINGFACE_HUB_CACHE=%s\n' "$HUGGINGFACE_HUB_CACHE"
printf 'TRANSFORMERS_CACHE=%s\n' "$TRANSFORMERS_CACHE"
```

---

### 3a) Infer runtime transient paths

Keep long-lived model shards in `HF_HOME`, and route infer runtime transients inside the infer workspace so transient outputs stay colocated with the run context.

```bash
export INFER_WORKSPACE_ROOT="${INFER_WORKSPACE_ROOT:-/project/<your_project>/$USER/dnadesign/src/dnadesign/infer/workspaces/test_stress_ethanol}"
export INFER_RUNTIME_ROOT="${INFER_RUNTIME_ROOT:-$INFER_WORKSPACE_ROOT/outputs/runtime/evo2-gpu}"
export TMPDIR="${TMPDIR:-$INFER_RUNTIME_ROOT/tmp}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$INFER_RUNTIME_ROOT/torch-extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$INFER_RUNTIME_ROOT/triton-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$INFER_RUNTIME_ROOT/pycache}"

mkdir -p \
  "$TMPDIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TRITON_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX"

printf 'TMPDIR=%s\n' "$TMPDIR"
printf 'TORCH_EXTENSIONS_DIR=%s\n' "$TORCH_EXTENSIONS_DIR"
printf 'TRITON_CACHE_DIR=%s\n' "$TRITON_CACHE_DIR"
printf 'PYTHONPYCACHEPREFIX=%s\n' "$PYTHONPYCACHEPREFIX"
printf 'HF_HOME=%s\n' "$HF_HOME"
```

---

### 3b) Evo2 checkpoint placement policy

Use this policy in runbooks and sessions:

- Keep `evo2_7b` under `HF_HOME_7B` on `/project`.
- Keep `evo2_20b` under `HF_HOME_20B` on `/project`.
- Keep `HF_HOME` pointed to the model-specific root selected by `TARGET_MODEL_ID`.
- Treat `evo2_40b` as optional and non-default on SCC until a dedicated multi-Hopper lane is validated.
- Export `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` to `HF_HOME` subpaths to override inherited SCC cache defaults.

---

### 4) Load toolchain modules

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

### GPU setup and verification runbook

Use this sequence as the default SCC Evo2 infer setup path.

```bash
cd dnadesign

# Toolchain and compiler roots.
module purge
module load cuda/<version>
module load gcc/<version>
export CC="$(which gcc)"
export CXX="$(which g++)"
export CUDAHOSTCXX="$(which g++)"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"

# Runtime/cache roots.
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export INFER_WORKSPACE_ROOT="${INFER_WORKSPACE_ROOT:-/project/<your_project>/$USER/dnadesign/src/dnadesign/infer/workspaces/test_stress_ethanol}"
export INFER_RUNTIME_ROOT="${INFER_RUNTIME_ROOT:-$INFER_WORKSPACE_ROOT/outputs/runtime/evo2-gpu}"
export TARGET_MODEL_ID="${TARGET_MODEL_ID:-evo2_7b}"
export HF_HOME_7B="${HF_HOME_7B:-/project/<your_project>/$USER/cache/huggingface/evo2_7b}"
export HF_HOME_20B="${HF_HOME_20B:-/project/<your_project>/$USER/cache/huggingface/evo2_20b}"
case "$TARGET_MODEL_ID" in
  evo2_7b) export HF_HOME="${HF_HOME:-$HF_HOME_7B}" ;;
  evo2_20b) export HF_HOME="${HF_HOME:-$HF_HOME_20B}" ;;
  *)
    printf 'Unsupported TARGET_MODEL_ID=%s\n' "$TARGET_MODEL_ID" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$INFER_RUNTIME_ROOT/uv-cache}"
export TMPDIR="${TMPDIR:-$INFER_RUNTIME_ROOT/tmp}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$INFER_RUNTIME_ROOT/torch-extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$INFER_RUNTIME_ROOT/triton-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$INFER_RUNTIME_ROOT/pycache}"
mkdir -p \
  "$UV_CACHE_DIR" \
  "$TMPDIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TRITON_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE"

# Prepare Python and base runtime before computing wheel include paths.
uv python install 3.12
uv sync --locked

# Include headers from CUDA and NVIDIA wheel packages for TE/flash-attn builds.
NVIDIA_INCLUDE_DIRS="$("$UV_PROJECT_ENVIRONMENT"/bin/python - <<'PY'
import site
from pathlib import Path
parts = []
for sp in site.getsitepackages():
    nvidia = Path(sp) / "nvidia"
    if nvidia.exists():
        for include_dir in sorted(nvidia.glob("*/include")):
            parts.append(str(include_dir))
print(":".join(parts))
PY
)"
export CPATH="$CUDA_HOME/include${NVIDIA_INCLUDE_DIRS:+:$NVIDIA_INCLUDE_DIRS}${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CPATH"

# Build controls.
# Apply the capacity/profile gate in section 6.4 first.
# These defaults are fallback values.
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"
export MAX_JOBS="${MAX_JOBS:-2}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

# Keep explicit flags for reproducibility and constrained fallback runs.
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-89}"

# Sync Evo2 GPU stack.
uv sync --locked --extra infer-evo2

# Verify package + infer wiring with fail-fast imports.
uv run python - <<'PY'
import importlib
import importlib.metadata as im
import torch

required_dist = ("torch", "transformer-engine", "flash-attn", "evo2", "vtx")
missing = []

print("cuda_available", torch.cuda.is_available())
for name in required_dist:
    try:
        print(name, im.version(name))
    except Exception:
        missing.append(f"missing_dist:{name}")

for module_name in ("transformer_engine.pytorch", "flash_attn", "evo2"):
    try:
        importlib.import_module(module_name)
        print(module_name, "import_ok")
    except Exception as exc:
        missing.append(f"import_failed:{module_name}:{type(exc).__name__}:{exc}")

if missing:
    print("MISSING_REQUIRED")
    for item in missing:
        print(item)
    raise SystemExit(1)
PY
uv run infer adapters list
```

If the verification block prints `MISSING_REQUIRED`, rebuild the two compiled extensions explicitly:

```bash
uv sync --locked --extra infer-evo2 \
  --reinstall-package flash-attn \
  --reinstall-package transformer-engine-torch
```

### Why these settings are required

- UV dependency model:
  - `uv sync --locked` installs base runtime only.
  - `uv sync --locked --extra infer-evo2` adds Evo2 GPU dependencies.
  - `uv sync --locked --group dev --extra infer-evo2` installs dev tooling and Evo2 GPU dependencies together.
  - `pyproject.toml` sets `[tool.uv] default-groups = []`, so dev/test groups are opt-in.
  - use `uv add` / `uv remove` only for dependency declaration changes (`pyproject.toml` + `uv.lock` updates).
  - use `uv sync --reinstall-package <pkg>` for environment rebuilds without dependency graph changes.
  - each `uv sync` realizes exactly the requested groups/extras; if `--extra infer-evo2` is omitted later, GPU packages are removed from the environment.
- explicit compiler/module setup:
  - Python GPU extension builds require the active `gcc` and `nvcc` toolchain to match headers/libraries on host.
- explicit transient routing:
  - infer runtime churn (tmp, torch extensions, triton cache, bytecode) stays out of near-full model storage paths.
- explicit flash-attn source build controls:
  - flash-attn is sdist-only in `uv.lock`, so source compilation is expected in the current lock state.
  - `FLASH_ATTENTION_FORCE_BUILD` and `FLASH_ATTN_CUDA_ARCHS` keep build behavior explicit.
- explicit include-path composition:
  - `CPATH` and `CPLUS_INCLUDE_PATH` include CUDA and wheel-provided `nvidia/*/include` headers to avoid missing `nccl.h` and related build failures.
- fail-fast runtime verification:
  - `MISSING_REQUIRED` + `raise SystemExit(1)` prevents partial extension installs from being treated as success.

---

### 5) Sync dependencies

```bash
uv python install 3.12
uv sync --locked
uv sync --locked --extra infer-evo2
```

`infer-evo2` is Linux-only and intended for SCC GPU nodes.

If this same environment also needs test/lint tools:

```bash
uv sync --locked --group dev --extra infer-evo2
```

---

### 6) Smoke tests

#### 6.1 Quick environment check

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

#### 6.2 Extended TE/FlashAttention/Evo2 check

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

#### 6.3 Model support: 7B and FP8 checkpoints

- 7B checkpoints (`evo2_7b`, `evo2_7b_base`, `evo2_7b_262k`) can run without Transformer Engine and fit the default L40S-style lane (`gpu_c=8.9`).
- FP8 checkpoints (`evo2_20b`, `evo2_40b`, `evo2_40b_base`, `evo2_1b_base`) require Transformer Engine and Hopper-class GPUs.
- On BU SCC, use `qgpus` and request `gpu_c=9.0` for `evo2_20b`; H200 is the relevant currently visible Hopper lane.
- `dnadesign` currently pins torch in the infer extra to `2.8.x`; Evo2 upstream docs recommend `2.6.x` or `2.7.x`. Always run smoke tests after sync on the target host.
- `infer` currently supports `evo2_7b`, `evo2_20b`, and `evo2_40b`; 400B is not a supported `model.id` in this stack.

```bash
uv run infer adapters list
uv run python - <<'PY'
import importlib.metadata as im
for name in ("torch", "transformer-engine", "flash-attn", "evo2", "vtx"):
    try:
        print(name, im.version(name))
    except Exception:
        print(name, "(not installed)")
PY
```

Minimal infer execution smoke (`evo2_7b`, one sequence):

```bash
uv run infer extract \
  --model-id evo2_7b \
  --device cuda:0 \
  --precision bf16 \
  --alphabet dna \
  --batch-size 1 \
  --fn evo2.log_likelihood \
  --format float \
  --seq ACGTACGTACGT \
  --no-progress
```

Model prefetch without runtime:

```bash
TARGET_MODEL_ID=evo2_7b HF_HOME="$HF_HOME_7B" uv run python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("arcinstitute/evo2_7b"))
PY

TARGET_MODEL_ID=evo2_20b HF_HOME="$HF_HOME_20B" uv run python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("arcinstitute/evo2_20b"))
PY
```

20B preflight on a non-Hopper GPU should fail fast:

```bash
TARGET_MODEL_ID=evo2_20b uv run infer validate config --config <path_to_evo2_20b_config.yaml>
# expected shape:
# CAPACITY_FAIL model_id=evo2_20b ... requires Hopper-class GPUs ...
```

40B preflight on one L40S-class GPU should fail fast:

```bash
uv run infer validate config --config <path_to_evo2_40b_config.yaml>
# expected shape:
# CAPACITY_FAIL model_id=evo2_40b precision=bf16 required_gib=... usable_gib=...
```

---

#### 6.4 Capacity gate and resource profile

Run this once before build or submit. It selects build parallelism from `NSLOTS`, sets `FLASH_ATTN_CUDA_ARCHS` from detected GPU capability, and fails when the requested model/precision is not a safe fit.

```bash
export TARGET_MODEL_ID="${TARGET_MODEL_ID:-evo2_7b}"
export TARGET_PRECISION="${TARGET_PRECISION:-bf16}"

eval "$(
uv run python - <<'PY'
import os
import subprocess
import sys

model_id = os.environ.get("TARGET_MODEL_ID", "evo2_7b")
precision = os.environ.get("TARGET_PRECISION", "bf16")
params_b = {"evo2_7b": 7.0, "evo2_20b": 20.0, "evo2_40b": 40.0}
bytes_per = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0}

if model_id not in params_b:
    raise SystemExit(f"Unsupported TARGET_MODEL_ID={model_id}")
if precision not in bytes_per:
    raise SystemExit(f"Unsupported TARGET_PRECISION={precision}")

line = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total,compute_cap", "--format=csv,noheader,nounits"],
    text=True,
).splitlines()[0]
gpu_total_mib, gpu_cc = [part.strip() for part in line.split(",")]
gpu_total_gib = int(gpu_total_mib) / 1024.0
gpu_usable_gib = gpu_total_gib * 0.90
flash_arch = gpu_cc.replace(".", "")
gpu_cc_tuple = tuple(int(part) for part in gpu_cc.split("."))

if model_id in {"evo2_20b", "evo2_40b"} and gpu_cc_tuple < (9, 0):
    print(
        "RUN_CAPACITY_FAIL "
        f"model={model_id} precision={precision} gpu_cc={gpu_cc} "
        "requires Hopper-class GPUs for the current Evo2 upstream contract",
        file=sys.stderr,
    )
    print(
        "Use gpu_c=9.0 on SCC and schedule onto a Hopper lane such as H200 for evo2_20b.",
        file=sys.stderr,
    )
    raise SystemExit(2)

weight_gib = params_b[model_id] * 1e9 * bytes_per[precision] / (1024.0 ** 3)
required_gib = weight_gib * 1.25
if required_gib > gpu_usable_gib:
    print(
        "RUN_CAPACITY_FAIL "
        f"model={model_id} precision={precision} "
        f"gpu_total_gib={gpu_total_gib:.1f} required_gib={required_gib:.1f}",
        file=sys.stderr,
    )
    print(
        "single L40S-class 45-48 GiB GPUs are a safe fit for evo2_7b in this infer stack; "
        "evo2_20b/evo2_40b require Hopper-class GPUs and additional memory headroom.",
        file=sys.stderr,
    )
    raise SystemExit(2)

slots = max(1, int(os.environ.get("NSLOTS", "1")))
build_jobs = 1 if slots <= 2 else 2 if slots <= 4 else 4

print("export UV_CONCURRENT_BUILDS=1")
print("export UV_CONCURRENT_INSTALLS=1")
print(f"export MAX_JOBS={build_jobs}")
print(f"export CMAKE_BUILD_PARALLEL_LEVEL={build_jobs}")
print(f"export OMP_NUM_THREADS={build_jobs}")
print("export FLASH_ATTENTION_FORCE_BUILD=TRUE")
print(f"export FLASH_ATTN_CUDA_ARCHS={flash_arch}")
print(
    "echo RESOURCE_GATE_OK "
    f"model={model_id} precision={precision} "
    f"gpu_total_gib={gpu_total_gib:.1f} required_gib={required_gib:.1f} "
    f"nslots={slots} build_jobs={build_jobs} flash_arch={flash_arch}"
)
PY
)"
```

Expected result for `TARGET_MODEL_ID=evo2_7b` on one L40S: `RESOURCE_GATE_OK`.
Expected result for `TARGET_MODEL_ID=evo2_20b` on one H200 (`gpu_c=9.0`, `gpu_memory_gib=80.0`): `RESOURCE_GATE_OK`.
Expected result for `TARGET_MODEL_ID=evo2_40b` on one L40S: `RUN_CAPACITY_FAIL`.

---

### 7) Next step

For long jobs, arrays, Notify watchers, and transfer-node workflows, use:
- [BU SCC Batch + Notify runbook](batch-notify.md)

---

### Troubleshooting and build throttles

Use conservative build caps when TE/flash-attn builds fail due to memory pressure:

```bash
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}"
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"
```

If CUDA headers are not discovered during Transformer Engine builds:

```bash
export NVTE_CUDA_INCLUDE_PATH="${NVTE_CUDA_INCLUDE_PATH:-$CUDA_HOME/include}"
```

If Transformer Engine fails on `nccl.h`:

```bash
NVIDIA_INCLUDE_DIRS="$("$UV_PROJECT_ENVIRONMENT"/bin/python - <<'PY'
import site
from pathlib import Path
parts = []
for sp in site.getsitepackages():
    nvidia = Path(sp) / "nvidia"
    if nvidia.exists():
        for include_dir in sorted(nvidia.glob("*/include")):
            parts.append(str(include_dir))
print(":".join(parts))
PY
)"
export CPATH="$CUDA_HOME/include${NVIDIA_INCLUDE_DIRS:+:$NVIDIA_INCLUDE_DIRS}${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CPATH"
```

#### Optional deep diagnostics (full import + extension checks)

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

#### Optional Evo2 callable check

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

Back: [BU SCC index](README.md)

Next: [BU SCC Quickstart](quickstart.md)

Next: [BU SCC Batch + Notify runbook](batch-notify.md)
