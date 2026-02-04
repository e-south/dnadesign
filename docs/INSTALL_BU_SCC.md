## Installing `dnadesign` on the BU SCC (GPU / Evo 2)

Some pipelines in `dnadesign` are designed to run on a [shared computing cluster](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/), such as solving dense arrays with [Gurobi](https://www.gurobi.com/), or running inference with [Evo 2](https://github.com/ArcInstitute/evo2), which requires access to CUDA and GPUs.

A GPU inference stack is specified in the `pyproject.toml` as an optional extra (`infer-evo2`) and is Linux-only.

What you get:

- `uv sync --locked` → CPU-safe base environment (macOS/Linux).
- `uv sync --locked --extra infer-evo2` → full inference stack including Evo2.

#### 0. Get onto the right kind of node

Evo2 FP8 support requires **Compute Capability 8.9+** (Ada/Hopper/Blackwell). On SCC, request a GPU node accordingly. SCC Interactive Session Resource Request Example:

> - **densegen** workflow:
>   - Modules: gurobi
>   - Cores: 16
>   - GPUs: 0
> - **infer** workflow:
>   - Modules: cuda
>   - Cores: 4
>   - GPUs: 1
>   - GPU Compute Capability: 8.9
>   - Extra options: `-l mem_per_core=8G`

Check cluster documentation for submission details.

#### 1. Install uv (once)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Clone repo

```bash
git clone https://github.com/e-south/dnadesign.git
cd dnadesign
```

(Optional) if you're trying to start from scratch:

- `rm -rf .venv` and then wipe uv caches with `uv cache clean` or `uv cache prune`.

#### 3. Put **caches outisde of home directory**

On the SCC you often have more space in `/project/...` or `/projectnb/...` than $HOME. uv supports relocating the project environment via `UV_PROJECT_ENVIRONMENT`.

```bash
# SCC gives you TMPDIR in jobs; fall back if not set
export SCC_SCRATCH="${TMPDIR:-/scratch/$USER}"

# Put uv cache on scratch (big, disposable)
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCC_SCRATCH/uv-cache}"

# Network robustness (clusters sometimes have flaky outbound)
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"

echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
echo "TMPDIR=$TMPDIR"
echo "UV_LINK_MODE=$UV_LINK_MODE"

export HF_HOME="/projectnb/dunlop/$USER/huggingface"
```

#### 4. Toolchain modules (required to compile TE + flash-attn)

Evo2 prerequisites require CUDA toolkit and a C++17-capable compiler; GCC 9+ is explicitly recommended in Evo2 docs .

```bash
module purge
module load cuda/12.8
module load gcc/10.2.0

export CC="$(which gcc)"
export CXX="$(which g++)"
export CUDAHOSTCXX="$(which g++)"

# Helpful for some builds if CUDA headers aren't found
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"

nvcc --version
gcc --version
g++ --version
```

Transformer Engine docs note that if CUDA headers aren’t in a standard path, you may need to set `NVTE_CUDA_INCLUDE_PATH`:

```bash
export NVTE_CUDA_INCLUDE_PATH="${NVTE_CUDA_INCLUDE_PATH:-$CUDA_HOME/include}"
```

#### 5. Safe compile caps (prevents RAM blowups)

flash-attn specifically recommends capping parallel jobs with `MAX_JOBS` on RAM-limited machines .

These defaults are conservative (slow but reliable):

```bash
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"

export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"

# TE-specific build throttle (common env var used in TE builds)
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"
```

#### 6. Install (base, then GPU extra)

```bash
uv python install 3.12
uv sync --locked
uv sync --locked --extra infer-evo2
```

`flash-attn` and `transformer-engine-torch` are configured in `pyproject.toml` to build **without isolation**, so they build against the already-installed torch in the project env (and uv will install torch first in a two-phase pass).

#### 7. Smoke tests (torch + CUDA + TE + flash-attn + evo2 import)

Run this to confirm (a) torch imports, (b) CUDA is visible on your node, and (c) TE/FlashAttention load.


```bash
uv run python - <<'PY'
import os, sys, glob, subprocess
import importlib.util as iu
import importlib.metadata as im

def v(dist):
    try: return im.version(dist)
    except im.PackageNotFoundError: return None

def sh(cmd):
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()

print("Python:", sys.version.split()[0])
print("Exe:", sys.executable)
print("Platform:", sys.platform)
print("UV_CACHE_DIR:", os.environ.get("UV_CACHE_DIR"))
print("UV_PROJECT_ENVIRONMENT:", os.environ.get("UV_PROJECT_ENVIRONMENT"))
print("TMPDIR:", os.environ.get("TMPDIR"))

for dist in ["torch","torchvision","torchaudio","transformer-engine","flash-attn","evo2"]:
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
    print("\n[diag] torch extension:", so)
    rc, out = sh(["ldd", so])
    print(out)

try:
    import torch
except Exception as e:
    print("\n[FAIL] torch import failed:", repr(e))
    ldd_torch_so()
    raise SystemExit(2)

print("\n[OK] torch:", torch.__version__, "torch CUDA:", torch.version.cuda)
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

print("\nAll smoke tests passed.")
PY
```

Also check whether Evo 2 models can be called:
```bash
uv run python - <<'PY'
import torch
from evo2 import Evo2

evo2_model = Evo2('evo2_7b')

sequence = 'ACGT'
input_ids = torch.tensor(
    evo2_model.tokenizer.tokenize(sequence),
    dtype=torch.int,
).unsqueeze(0).to('cuda:0')

outputs, _ = evo2_model(input_ids)
logits = outputs[0]

print('Logits: ', logits)
print('Shape (batch, length, vocab): ', logits.shape)
PY
```

---

@e-south
