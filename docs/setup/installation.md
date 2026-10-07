## Installation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-06

This guide is the first-run setup sequence for local development and CLI usage: confirm platform/version contracts, run the required install and baseline verification, then run additional sections only when required by the workload.

### 1) Platform support

| Environment | Status | Notes |
| --- | --- | --- |
| macOS (`arm64`, `x86_64`) | Supported | Use this guide directly. |
| Linux (`x86_64`) | Supported | Use this guide directly. |
| Windows 10/11 via WSL2 | Supported | Run commands in the WSL Linux shell. |
| Native Windows (PowerShell/CMD) | Not supported | Use WSL2. |

Reason:
- `pyproject.toml` `[tool.uv] environments` targets `darwin` and `linux x86_64`.
- `pixi.toml` `[workspace] platforms` targets `osx-arm64`, `osx-64`, and `linux-64`.

### 2) Confirm version contracts
Treat these as installation requirements:

- Python: `>=3.12,<3.13` (`pyproject.toml` `[project] requires-python`)
- uv: `>=0.12.3,<0.13` (`pyproject.toml` `[tool.uv] required-version`)

The distribution name is `dnadesign-tools`; Python imports remain `dnadesign`
and tool commands remain unchanged. Do not install the unrelated `dnadesign`
PyPI project. DNADesign uses one distribution version for its tools. Python's
`dnadesign.__version__` and tool software-version fields read installed package
metadata; API and artifact schema versions remain independent. Version 0.2.0
introduces the lightweight installed-library boundary below. A source version
alone is not a release receipt: consumers must pin a qualified source revision
or built artifact and retain its checksum.

### 2a) UV dependency model

- Checkout install (`uv sync --locked`):
  - installs the full tool runtime through the default `tools` dependency group.
  - test/lint tools still require the `dev` group.
- Installed library (`pip install <dnadesign-wheel>`):
  - requires NumPy only; supports the public MSRB scoring API.
  - install the wheel with its `full` extra for campaign execution, other tool
    APIs and CLIs. This retains the complete tool dependency set.
  - use `uv sync --locked --no-default-groups` only for a scoring-only checkout.
- Development tools (`uv sync --locked --group dev`):
  - installs test/lint tooling.
- GPU infer stack (`uv sync --locked --extra infer-evo2`):
  - installs Evo2 GPU stack additions (`flash-attn`, `transformer-engine`, `evo2`) on Linux `x86_64`.
- GPU plus development tools (`uv sync --locked --group dev --extra infer-evo2`):
  - installs both dev tooling and Evo2 GPU stack in one environment realization.
- Dependency declaration changes (`uv add` / `uv remove`):
  - use when changing project dependencies so `pyproject.toml` and `uv.lock` remain canonical.
- Environment rebuild (`uv sync --reinstall-package ...`):
  - use when rebuilding a package in-place without changing dependency declarations.
- Important:
  - each `uv sync` realizes exactly the requested groups/extras.
  - if you run `uv sync --locked --group dev` on a GPU env, it removes `infer-evo2` extras unless `--extra infer-evo2` is also included.

The lightweight library and full tool installation use the same distribution and
scoring implementation. Installing the lightweight library does not qualify
campaign fitting, sequence inference or rendering. See the
[MSRB Python example](../../src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md#python-scoring-without-a-campaign).
The `full` extra declares the same immutable Dense Arrays Git source used by the
checkout, because that dependency is not on PyPI. Full installs require Git and
access to that source. These distributions can be shared as versioned build
artifacts; PyPI publication requires replacing the direct Git requirement with
a qualified published dependency first.

### 3) Required path: install and baseline verify
Run this full block in order:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# If uv is not on PATH yet, start a new shell session.

git clone https://github.com/e-south/dnadesign.git
cd dnadesign

uv --version
uv python install 3.12
uv sync --locked

# Baseline verification.
uv run python -c "import dnadesign, pandas, pyarrow; print('ok')"
uv run usr --help
```

You are done with checkout installation when:
- the import check prints `ok`
- `uv run usr --help` exits successfully

### 4) Development tools (when needed)
Run this when you will lint or test locally.

```bash
uv sync --locked --group dev
uv run ruff --version
uv run pytest --version
```

If this same environment also needs Evo2 infer:

```bash
uv sync --locked --group dev --extra infer-evo2
```

### 5) System binaries with pixi (when needed)
Run this when a workflow needs pixi-managed tools such as MEME/FIMO.

```bash
pixi install --locked
# Verify MEME/FIMO is available in the pixi environment.
pixi run -- fimo --version
```

Dependency maintenance operations are documented in **[Dependency maintenance](dependencies.md)**.

### 6) GPU extra (`infer-evo2`)
This command is Linux `x86_64` only.

```bash
uv sync --locked --extra infer-evo2
```

If this same environment also needs test/lint tools:

```bash
uv sync --locked --group dev --extra infer-evo2
```

For SCC GPU setup, including environment exports and build controls, use:
- [BU SCC install bootstrap: GPU setup and verification runbook](../bu-scc/setup/install.md#gpu-setup-and-verification-runbook)
- [BU SCC install bootstrap: capacity gate and resource profile](../bu-scc/setup/install.md#64-capacity-gate-and-resource-profile)
- [infer SCC Evo2 GPU environment runbook](../../src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md)

SCC infer path policy:
- keep one environment at `<dnadesign_repo>/.venv` (`UV_PROJECT_ENVIRONMENT="$PWD/.venv"`).
- keep infer model caches for routine runs on `/project` (`HF_HOME_7B`, `HF_HOME_20B`) and set `HUGGINGFACE_HUB_CACHE` plus `TRANSFORMERS_CACHE` under `HF_HOME`.
- keep `TARGET_MODEL_ID` aligned with the selected cache root so `HF_HOME` points to the active model cache.
- keep transient build/runtime outputs in infer workspace `outputs/runtime/...`.

Current lock behavior note:
- flash-attn is currently sdist-only in `uv.lock`, so SCC `infer-evo2` setup compiles flash-attn from source.
- infer currently supports `evo2_7b`, `evo2_20b`, and `evo2_40b`; 400B is not a supported `model.id`.

### 7) Troubleshooting quick checks
Use these checks before deeper debugging:

- If `uv` is not found after install, start a new shell and rerun `uv --version`.
- If matplotlib cache warnings appear, set `MPLCONFIGDIR` to a writable path:

```bash
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
```

- If you are on Windows, confirm commands are running inside WSL2 (`uname -a` should report Linux).

### 8) Continue with workflow docs
- If you want the shortest shell-first route after baseline verification, start with `uv run ops catalog list --simple`.
- For workflow execution from the docs map, use [docs/README.md](../README.md).
- For BU SCC workflows, follow this order: [BU SCC quickstart](../bu-scc/setup/quickstart.md), [BU SCC install bootstrap](../bu-scc/setup/install.md), then [BU SCC batch plus Notify runbook](../bu-scc/runbooks/batch-notify.md).
