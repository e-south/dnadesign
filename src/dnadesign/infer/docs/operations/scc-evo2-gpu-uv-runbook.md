## SCC Evo2 GPU Environment Runbook (UV + infer)

Use this page when you need a deterministic SCC GPU environment build for infer.

For BU SCC platform details and scheduler policy, see [BU SCC install bootstrap](../../../../../docs/bu-scc/install.md).

### Assumptions

- Linux `x86_64` host with CUDA modules available.
- You are on an SCC interactive GPU node or equivalent GPU-capable shell.
- You want infer Evo2 support (`infer-evo2` extra) and deterministic build behavior.

### Path policy

- Keep one canonical uv environment at `<dnadesign_repo>/.venv`.
- Keep infer model cache for routine runs (`HF_HOME`) on `/project`.
- Keep large external Evo2 artifacts (for example 400B assets) on `/projectnb`.
- Keep runtime transients inside infer workspace `outputs/runtime/...`.

### Lockfile preflight

flash-attn is sdist-only in `uv.lock`, so this environment currently compiles flash-attn from source during `uv sync --locked --extra infer-evo2`.

```bash
cd /project/dunlop/esouth/dnadesign
sed -n '632,650p' uv.lock
```

### Capacity and build profile gate

Run this once per interactive session. It sets build knobs from `NSLOTS`, sets `FLASH_ATTN_CUDA_ARCHS` from detected GPU capability, and fails fast when the requested model/precision is not a safe fit for the detected GPU memory.

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
    raise SystemExit(
        f"Unsupported TARGET_MODEL_ID={model_id}. "
        "Supported: evo2_7b, evo2_20b, evo2_40b."
    )
if precision not in bytes_per:
    raise SystemExit(
        f"Unsupported TARGET_PRECISION={precision}. "
        "Supported: fp32, fp16, bf16."
    )

try:
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()[0]
except Exception as exc:
    raise SystemExit(f"nvidia-smi query failed: {exc}")

parts = [p.strip() for p in line.split(",")]
if len(parts) != 2:
    raise SystemExit(f"Unexpected nvidia-smi output: {line}")

gpu_total_mib = int(parts[0])
gpu_cc = parts[1]
gpu_total_gib = gpu_total_mib / 1024.0
gpu_usable_gib = gpu_total_gib * 0.90
flash_arch = gpu_cc.replace(".", "")

weight_gib = params_b[model_id] * 1e9 * bytes_per[precision] / (1024.0 ** 3)
required_gib = weight_gib * 1.25

if required_gib > gpu_usable_gib:
    print(
        "RUN_CAPACITY_FAIL "
        f"model={model_id} precision={precision} "
        f"gpu_total_gib={gpu_total_gib:.1f} gpu_usable_gib={gpu_usable_gib:.1f} "
        f"required_gib={required_gib:.1f}",
        file=sys.stderr,
    )
    print(
        "single L40S-class 45-48 GiB GPUs are a safe fit for evo2_7b in this infer stack; "
        "evo2_20b/evo2_40b require additional GPU memory headroom and currently fail this gate.",
        file=sys.stderr,
    )
    print(
        "infer exposes fp32/fp16/bf16 only. Quantized/offloaded 40B execution is not currently wired.",
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

`infer` currently supports `evo2_7b`, `evo2_20b`, and `evo2_40b`. A 400B model is out of scope for this stack and is not a supported `model.id`.

### Setup and verification steps

```bash
cd /project/dunlop/esouth/dnadesign

module purge
module load cuda/12.8
module load gcc/13.2.0

export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export INFER_WORKSPACE_ROOT=/project/dunlop/esouth/dnadesign/src/dnadesign/infer/workspaces/test_stress_ethanol
export INFER_RUNTIME_ROOT="${INFER_RUNTIME_ROOT:-$INFER_WORKSPACE_ROOT/outputs/runtime/evo2-gpu}"
export HF_HOME_7B="${HF_HOME_7B:-/project/dunlop/esouth/cache/huggingface/evo2_7b}"
export HF_HOME_LARGE="${HF_HOME_LARGE:-/projectnb/dunlop/esouth/cache/huggingface/evo2_large}"
export HF_HOME="${HF_HOME:-$HF_HOME_7B}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export UV_CACHE_DIR="$INFER_RUNTIME_ROOT/uv-cache"
export TMPDIR="$INFER_RUNTIME_ROOT/tmp"
export TORCH_EXTENSIONS_DIR="$INFER_RUNTIME_ROOT/torch-extensions"
export TRITON_CACHE_DIR="$INFER_RUNTIME_ROOT/triton-cache"
export PYTHONPYCACHEPREFIX="$INFER_RUNTIME_ROOT/pycache"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$HF_HOME" "$HF_HUB_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

uv python install 3.12
uv sync --locked

export CC="$(which gcc)"
export CXX="$(which g++)"
export CUDAHOSTCXX="$(which g++)"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"

NVIDIA_INCLUDE_DIRS="$($UV_PROJECT_ENVIRONMENT/bin/python - <<'PY'
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
# Apply the profile gate above first. These defaults are fallback values.
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"
export MAX_JOBS="${MAX_JOBS:-2}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

# Keep explicit flags for reproducibility and constrained fallback runs.
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-89}"

uv sync --locked --extra infer-evo2

# Fail-fast runtime verification. Do not continue when any required import is missing.
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
uv run infer validate config --config src/dnadesign/infer/workspaces/test_stress_ethanol/config.yaml

# Real execution smoke (loads evo2_7b and runs one inference).
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

### API pressure checks (forward, embeddings, generation)

Use these checks to verify Evo2 usage contracts in infer:

- logits/embedding pooling uses sequence dimension with `pool.dim=1`.
- `pool.dim=0` is rejected to avoid consuming batch axis.
- `evo2.embedding` requires `layer`.
- mean pooling follows `e = (1/n) * Σ_j E_j` over token positions.

```bash
uv run python - <<'PY'
from dnadesign.infer import run_extract, run_generate

seqs = ["ACGTACGT", "ACGT"]

logits = run_extract(
    seqs,
    model_id="evo2_7b",
    device="cuda:0",
    precision="bf16",
    alphabet="dna",
    batch_size=2,
    outputs=[{
        "id": "logits_mean",
        "fn": "evo2.logits",
        "params": {"pool": {"method": "mean", "dim": 1}},
        "format": "list",
    }],
)

emb = run_extract(
    seqs,
    model_id="evo2_7b",
    device="cuda:0",
    precision="bf16",
    alphabet="dna",
    batch_size=2,
    outputs=[{
        "id": "emb_mean",
        "fn": "evo2.embedding",
        "params": {"layer": "blocks.28.mlp.l3", "pool": {"method": "mean", "dim": 1}},
        "format": "list",
    }],
)

gen = run_generate(
    ["ACGTACGT"],
    model_id="evo2_7b",
    device="cuda:0",
    precision="bf16",
    alphabet="dna",
    batch_size=1,
    params={"max_new_tokens": 4, "temperature": 1.0, "top_k": 4, "seed": 7},
)

print("logits_widths", [len(row) for row in logits["logits_mean"]])
print("embedding_widths", [len(row) for row in emb["emb_mean"]])
print("generated", gen["gen_seqs"][0])
PY
```

`infer validate config` checks capacity when local GPUs are visible. On GPU-less hosts it validates schema/contracts and reports that capacity checks were skipped; use `ops runbook plan` with declared GPU resources for deterministic scheduler-side preflight.

### Recovery after interrupted or partial installs

If the verification block prints `MISSING_REQUIRED`, rebuild the two compiled extensions explicitly:

```bash
uv sync --locked --extra infer-evo2 \
  --reinstall-package flash-attn \
  --reinstall-package transformer-engine-torch
```

If this same environment also needs test/lint tools, keep extras and group together:

```bash
uv sync --locked --group dev --extra infer-evo2
```

If the node is memory-constrained, rerun with:

```bash
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export OMP_NUM_THREADS=1
```

### Why this setup works

- UV default groups:
  - `pyproject.toml` sets `[tool.uv] default-groups = []`, so baseline `uv sync --locked` installs runtime deps only.
- `infer-evo2 extra`:
  - `uv sync --locked --extra infer-evo2` adds the Evo2 GPU stack (`flash-attn`, `transformer-engine`, `evo2`, torch CUDA wheels).
- canonical UV mutation policy:
  - use `uv add` / `uv remove` only when changing dependency declarations.
  - use `uv sync` (including `--reinstall-package`) for environment realization and rebuilds.
- Why source-build controls are explicit:
  - `flash-attn` is sdist-only in the current lock, so source compilation is expected.
  - `FLASH_ATTENTION_FORCE_BUILD` and `FLASH_ATTN_CUDA_ARCHS` keep build behavior explicit.
- Why `CPATH`/`CPLUS_INCLUDE_PATH` include nvidia wheel headers:
  - Transformer Engine build can fail on `nccl.h` if only CUDA include paths are exported.
  - combining `$CUDA_HOME/include` and `site-packages/nvidia/*/include` avoids this mismatch.
- Why infer verification is included:
  - package metadata can look complete while runtime extension imports still fail.
  - the fail-fast `MISSING_REQUIRED` gate catches this before any job submission.
- Why build wall-time can be high:
  - on `NSLOTS=4` with `MAX_JOBS=2`, one full flash-attn source build took about 70 minutes in validation runs.
  - this is expected when the lock contains sdist-only flash-attn and no reusable wheel is already in the UV cache.

### Follow-on path

- For scheduler-managed pressure tests, continue with:
  - [Agnostic model + USR pressure test](pressure-test-agnostic-models.md)
  - [BU SCC Batch + Notify runbook](../../../../../docs/bu-scc/batch-notify.md)
