## SCC Evo2 GPU Environment Runbook (UV + infer)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-13

Use this page when you need a deterministic SCC GPU environment build for infer.

For BU SCC platform details and scheduler policy, see [BU SCC install bootstrap](../../../../../docs/bu-scc/setup/install.md).

### DNA input contract

Before running Evo2 lanes, verify that every persisted biological DNA source
product intended for model input is uppercase A/C/G/T. Evo2 tokenizes uppercase
and lowercase bases as different symbols. Infer canonicalizes incoming DNA
strings to uppercase before adapter calls, but sidecars generated before this
contract from lowercase or mixed-case source records can contain collapsed
geometry and invalid rank diagnostics. Regenerate those sidecars.

### Assumptions

- Linux `x86_64` host with CUDA modules available.
- You are on an SCC interactive GPU node or equivalent GPU-capable shell.
- You want infer Evo2 support (`infer-evo2` extra) and deterministic build behavior.

### Path policy

- Keep one active uv environment at `<dnadesign_repo>/.venv` for the current
  GPU-family contract.
- Keep `evo2_7b` and `evo2_20b` caches on `/project`, with one explicit root per model.
- Keep `HF_HOME` pointed at the active model-specific cache root.
- Keep runtime transients inside infer workspace `outputs/runtime/...`.

Pragmatic portability note:

- A single `.venv` is fine while current work stays on one GPU family.
- Once `flash-attn` is built from source, do not assume that environment is
  portable across Hopper, Blackwell, and smaller GPU families.
- Treat a working `.venv` as family-bound until a real `infer extract` smoke
  proves otherwise on the target family.

### Lockfile preflight

flash-attn is sdist-only in `uv.lock`, so this environment currently compiles flash-attn from source during `uv sync --locked --extra infer-evo2`.

```bash
cd /project/dunlop/esouth/dnadesign # Move to repo root on SCC storage.
sed -n '632,650p' uv.lock # Inspect locked flash-attn package entries.
```

### Capacity and build profile gate

Run this once per interactive session. It sets build knobs from `NSLOTS`, sets `FLASH_ATTN_CUDA_ARCHS` from detected GPU capability, and fails fast when the requested model/precision is not a safe fit for the detected GPU memory.

```bash
export TARGET_MODEL_ID="${TARGET_MODEL_ID:-evo2_7b}" # Select model lane for capacity gating.
export TARGET_PRECISION="${TARGET_PRECISION:-bf16}" # Select precision for memory-fit checks.

# Compute build/runtime gate exports from current GPU and slot state.
eval "$(
uv run python - <<'PY' # Emit export statements and fail fast on unsafe capacity.
import os
import subprocess
import sys

model_id = os.environ.get("TARGET_MODEL_ID", "evo2_7b")
precision = os.environ.get("TARGET_PRECISION", "bf16")

params_b = {"evo2_7b": 7.0, "evo2_20b": 20.0}
bytes_per = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0}

if model_id not in params_b:
    raise SystemExit(
        f"Unsupported TARGET_MODEL_ID={model_id}. "
        "Supported: evo2_7b, evo2_20b."
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
gpu_cc_tuple = tuple(int(part) for part in gpu_cc.split("."))

if model_id == "evo2_20b" and gpu_cc_tuple < (9, 0):
    print(
        "RUN_CAPACITY_FAIL "
        f"model={model_id} precision={precision} gpu_cc={gpu_cc} "
        "requires Hopper-class GPUs for the current Evo2 upstream contract",
        file=sys.stderr,
    )
    print(
        "Use gpu_c=9.0 as the generic evo2_20b model floor on SCC. For the current Blackwell-pinned dnadesign environment, request gpu_t=RTXP6000 and gpu_c=12.0.",
        file=sys.stderr,
    )
    raise SystemExit(2)

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
        "evo2_20b requires Hopper-class GPUs and additional memory headroom.",
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
# Close the eval command substitution after emitting exports.
)"
```

This SCC runbook documents the promoted Evo2 lane set for `infer`: `evo2_7b` and `evo2_20b`. A 400B model is out of scope for this stack and is not a supported `model.id`.

Use `evo2_7b` as the default SCC smoke and pressure-test lane. Use `evo2_20b`
only on GPU lanes that satisfy `gpu_c >= 9.0`; H200 is common on SCC, but
newer higher-capability lanes also qualify when memory is sufficient. When the
current `.venv` is family-pinned, add an exact selector; on the current SCC
probe surface, the visible Blackwell lane is `gpu_t=RTXP6000` with
`gpu_c=12.0`.

Model fit and environment portability are separate. Passing the capacity gate
does not prove that the current `.venv` can execute there. If this environment
was built on a different GPU family, require a real `infer extract` smoke on
the landed family before trusting batch portability.

### Setup and verification steps

```bash
cd /project/dunlop/esouth/dnadesign # Enter repo root used for SCC setup.

module purge # Clear inherited module state before deterministic loads.
module load cuda/12.8 # Load CUDA toolchain for torch/flash-attn builds.
module load gcc/13.2.0 # Load compiler toolchain compatible with CUDA build flow.

export UV_PROJECT_ENVIRONMENT="$PWD/.venv" # Use the active uv environment path for the current GPU family.
export INFER_WORKSPACE_ROOT=/project/dunlop/esouth/dnadesign/workspaces/demo_usr_pressure # Pin infer workspace root.
export INFER_RUNTIME_ROOT="${INFER_RUNTIME_ROOT:-$INFER_WORKSPACE_ROOT/outputs/runtime/evo2-gpu}" # Keep runtime artifacts workspace-scoped.
export TARGET_MODEL_ID="${TARGET_MODEL_ID:-evo2_7b}" # Default to the 7B SCC lane.
export HF_HOME_7B="${HF_HOME_7B:-/project/dunlop/esouth/cache/huggingface/evo2_7b}" # Define cache root for evo2_7b.
export HF_HOME_20B="${HF_HOME_20B:-/project/dunlop/esouth/cache/huggingface/evo2_20b}" # Define cache root for evo2_20b.
case "$TARGET_MODEL_ID" in # Select active HF cache by model lane.
  evo2_7b) export HF_HOME="${HF_HOME:-$HF_HOME_7B}" ;; # Resolve HF cache for evo2_7b.
  evo2_20b) export HF_HOME="${HF_HOME:-$HF_HOME_20B}" ;; # Resolve HF cache for evo2_20b.
  *) # Reject unsupported model ids.
    printf 'Unsupported TARGET_MODEL_ID=%s\n' "$TARGET_MODEL_ID" >&2 # Emit unsupported-model error to stderr.
    return 2 2>/dev/null || exit 2
    ;;
esac
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}" # Keep hub artifacts under selected model cache root.
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}" # Mirror legacy cache env to shared hub path.
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}" # Keep transformers cache colocated with model cache.
export UV_CACHE_DIR="$INFER_RUNTIME_ROOT/uv-cache" # Store uv cache under workspace runtime root.
export TMPDIR="$INFER_RUNTIME_ROOT/tmp" # Keep temporary build/runtime files local to workspace.
export TORCH_EXTENSIONS_DIR="$INFER_RUNTIME_ROOT/torch-extensions" # Place torch extension builds in runtime root.
export TRITON_CACHE_DIR="$INFER_RUNTIME_ROOT/triton-cache" # Place Triton kernels in runtime root.
export PYTHONPYCACHEPREFIX="$INFER_RUNTIME_ROOT/pycache" # Redirect Python bytecode cache out of source tree.
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$HF_HOME" "$HF_HUB_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" # Create all runtime/cache directories.

uv python install 3.12 # Ensure expected Python runtime is available to uv.
uv sync --locked # Realize baseline locked environment before Evo2 extra.

export CC="$(which gcc)" # Pin C compiler for extension builds.
export CXX="$(which g++)" # Pin C++ compiler for extension builds.
export CUDAHOSTCXX="$(which g++)" # Set CUDA host compiler explicitly.
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")" # Resolve CUDA install prefix from nvcc path.

NVIDIA_INCLUDE_DIRS="$($UV_PROJECT_ENVIRONMENT/bin/python - <<'PY' # Gather include dirs from nvidia wheels.
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
)" # Close include-dir probe substitution.
export CPATH="$CUDA_HOME/include${NVIDIA_INCLUDE_DIRS:+:$NVIDIA_INCLUDE_DIRS}${CPATH:+:$CPATH}" # Combine CUDA and nvidia-wheel headers.
export CPLUS_INCLUDE_PATH="$CPATH" # Mirror include path for C++ compilation.

# Build controls.
# Apply the profile gate above first. These defaults are fallback values.
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}" # Bound concurrent source-build workers.
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}" # Bound concurrent install workers.
export MAX_JOBS="${MAX_JOBS:-2}" # Bound compile parallelism to safe default.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}" # Bound CMake job fanout.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" # Bound OpenMP threads during builds.

# Keep explicit flags for reproducibility and constrained fallback runs.
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}" # Keep flash-attn source-build behavior explicit.
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-89}" # Keep CUDA arch targeting explicit.

uv sync --locked --extra infer-evo2 # Install Evo2 GPU dependency stack from lockfile.

# Fail-fast runtime verification. Do not continue when any required import is missing.
uv run python - <<'PY'
import importlib
import importlib.metadata as im
import torch

required_dist = ("torch", "transformer-engine", "flash-attn", "evo2", "vtx")
required_modules = ("transformer_engine.pytorch", "flash_attn", "evo2", "vortex")
missing = []

print("cuda_available", torch.cuda.is_available())
for name in required_dist:
    try:
        print(name, im.version(name))
    except Exception:
        missing.append(f"missing_dist:{name}")

for module_name in required_modules:
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

uv run infer adapters list # Confirm Evo2 adapter visibility in this environment.
uv run infer validate config --config "$INFER_WORKSPACE_ROOT/config.yaml" # Validate workspace config contracts.

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

Do not replace the real extract smoke above with `infer validate config` or
`infer run --dry-run`. Those are necessary config checks, but they do not prove
that the compiled CUDA extension stack is portable to the current GPU family.

### First real write-back gate

Treat live study collection as four explicit gates:

1. the checked-in study snapshot is current
2. the study preflight is green on the current host
3. Evo2 actually imports and loads on the live GPU host
4. the canonical USR `infer` namespace is registered before first write-back

For study-owned USR write-back lanes, run:

```bash
uv run ops progress show studies.stress-ethanol-cipro-growth.status --json # Confirm the checked-in study snapshot is current.
NOTIFY_WEBHOOK_FILE=<...> SSL_CERT_FILE=<...> uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --json # Confirm the current host is execution-ready.
uv run infer validate usr-registry --config <lane-config> # Derive the canonical infer namespace registration contract.
uv run usr --root src/dnadesign/usr/datasets namespace show infer # Confirm the shared USR root already knows the infer namespace.
```

If `namespace show infer` fails, use the register command emitted by
`infer validate usr-registry` before the first real write-back.

### First live signals on a GPU node

For `evo2_20b`, a healthy cold start can sit for a short window in
`fetch -> weight hydration -> GPU residency -> first attach events` before the
target dataset shows new rows in its event stream.

Before declaring the run hung, check:

- `nvidia-smi` shows the infer process and rising or stable memory residency
- the target dataset `.events.log` gains `attach` events with `completed_rows`
- the watcher cursor advances if Notify is following the same stream
- the watcher spool remains empty

For study-owned interactive watchers on an existing event stream, seed the
cursor to the current `.events.log` size before `notify usr-events watch
--follow` unless replay is intentional.

### API pressure checks (forward, embeddings, generation)

Use these checks to verify Evo2 usage contracts in infer:

- logits/embedding pooling uses sequence dimension with `pool.dim=1`.
- `pool.dim=0` is rejected to avoid consuming batch axis.
- `evo2.embedding` defaults to a model-aware selector when `params.layer` is omitted.
- `evo2_7b` defaults to `block26_mlp_out`, which maps to `blocks.26.mlp.l3`.
- `evo2_20b` defaults to `block23_mlp_out`, which maps to `blocks.23.mlp.l3`.
- `params.layer: mid` resolves to the default pooled embedding layer.
- `params.layer: final` resolves to the last Evo2 embedding block exposed by the loaded torch module.
- set `params.layer` to an explicit adapter-specific name only when you need a particular block.
- mean pooling follows `e = (1/n) * Σ_j E_j` over token positions. For causal
  Evo2 outputs, each `E_j` is prefix-conditioned on the emitted sequence up to
  position `j`; pooling does not give earlier tokens downstream context.

```bash
uv run python - <<'PY' # Run API-level extraction and generation sanity checks.
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
        "params": {"pool": {"method": "mean", "dim": 1}},
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

Model prefetch without runtime:

```bash
TARGET_MODEL_ID=evo2_7b HF_HOME="$HF_HOME_7B" uv run python - <<'PY' # Prefetch evo2_7b weights into cache.
from huggingface_hub import snapshot_download
print(snapshot_download("arcinstitute/evo2_7b"))
PY

TARGET_MODEL_ID=evo2_20b HF_HOME="$HF_HOME_20B" uv run python - <<'PY' # Prefetch evo2_20b weights into cache.
from huggingface_hub import snapshot_download
print(snapshot_download("arcinstitute/evo2_20b"))
PY
```

`infer validate config` checks capacity when local GPUs are visible. On GPU-less hosts it validates schema/contracts and reports that capacity checks were skipped; use `ops runbook plan` with declared GPU resources for deterministic scheduler-side preflight.

### Recovery after interrupted or partial installs

If the verification block prints `MISSING_REQUIRED`, rebuild the two compiled extensions explicitly:

```bash
# Rebuild compiled GPU extensions in the locked Evo2 environment.
uv sync --locked --extra infer-evo2 \
  --reinstall-package flash-attn \
  --reinstall-package transformer-engine-torch
```

If this same environment also needs test/lint tools, keep extras and group together:

```bash
uv sync --locked --group dev --extra infer-evo2 # Add dev tooling alongside Evo2 runtime extras.
```

If the node is memory-constrained, rerun with:

```bash
export UV_CONCURRENT_BUILDS=1 # Serialize concurrent build operations.
export UV_CONCURRENT_INSTALLS=1 # Serialize concurrent install operations.
export MAX_JOBS=1 # Limit compiler worker jobs.
export CMAKE_BUILD_PARALLEL_LEVEL=1 # Limit CMake parallelism.
export OMP_NUM_THREADS=1 # Limit OpenMP thread usage.
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
  - [BU SCC Batch + Notify runbook](../../../../../docs/bu-scc/runbooks/batch-notify.md)
