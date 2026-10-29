## Installation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28

Choose the runtime for the tools you need. Combine profiles by repeating `--extra`.
Add `--group dev` when running tests or lint checks.

| Use | Checkout command |
| --- | --- |
| NumPy-only scoring API | `uv sync --locked --no-default-groups` |
| Sequence datasets and overlays | `uv sync --locked --no-default-groups --extra seqledger` |
| Variant generation and plots | `uv sync --locked --no-default-groups --extra permuter` |
| Inference configuration and dry runs | `uv sync --locked --no-default-groups --extra infer` |
| OPAL campaigns and plots | `uv sync --locked --no-default-groups --extra opal` |
| OPAL with interactive notebooks | `uv sync --locked --no-default-groups --extra opal-notebook` |
| Sequence construction | `uv sync --locked --no-default-groups --extra construct` |
| ViennaRNA folding | `uv sync --locked --no-default-groups --extra folding` |
| Sequence rendering | `uv sync --locked --no-default-groups --extra baserender` |
| Motif optimization | `uv sync --locked --no-default-groups --extra cruncher` |
| Oligo planning | `uv sync --locked --no-default-groups --extra junction` |
| Clustering | `uv sync --locked --no-default-groups --extra cluster` |
| Representation analysis | `uv sync --locked --no-default-groups --extra latentdna` |
| Workflow discovery and orchestration | `uv sync --locked --no-default-groups --extra ops` |
| Event notifications | `uv sync --locked --no-default-groups --extra notify` |
| Complete toolkit | `uv sync --locked` |

Use `latentdna-notebook` or `cruncher-notebook` for interactive notebooks.
CPU profiles do not install Torch, Evo 2 or DenseArrays. OPS and Notify install
their own command runtimes; add the profile of any producer whose configuration
or outputs you will inspect or execute. Scheduler binaries are needed only for
the corresponding orchestration commands. Rendering video requires FFmpeg.

Activate the chosen environment with `source .venv/bin/activate`. Its installed
commands then work from a campaign or data directory outside the checkout.
When using `uv run` instead, repeat the same group/extra flags to preserve the
selected dependency scope.

### Platform support

| Environment | Status | Notes |
| --- | --- | --- |
| macOS (`arm64`, `x86_64`) | Supported | Use this guide directly. |
| Linux (`x86_64`) | Supported | Use this guide directly. |
| Windows 10/11 via WSL2 | Supported | Run commands in the WSL Linux shell. |
| Native Windows (PowerShell/CMD) | Not supported | Use WSL2. |

Reason:
- `pyproject.toml` `[tool.uv] environments` targets `darwin` and `linux x86_64`.
- `pixi.toml` `[workspace] platforms` targets `osx-arm64`, `osx-64`, and `linux-64`.

### Required versions

- Python: `>=3.12,<3.13` (`pyproject.toml` `[project] requires-python`)
- uv: `>=0.12.3,<0.13` (`pyproject.toml` `[tool.uv] required-version`)

### Install and verify
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
uv run seqledger --help
```

You are done with checkout installation when:
- the import check prints `ok`
- `uv run seqledger --help` exits successfully

### UV dependency model

The distribution is named `dnadesign-tools`; Python code imports `dnadesign`.
Use the checkout instructions above or a qualified wheel with the runtime profile
you need. The PyPI project named `dnadesign` is unrelated.

- Checkout install (`uv sync --locked`):
  - installs the full tool runtime through the default `tools` dependency group.
  - test/lint tools still require the `dev` group.
- Installed library (`pip install <dnadesign-wheel>`):
  - requires NumPy only; supports the public MSRB scoring API.
  - install the wheel with `opal` for campaigns or `opal-notebook` for interactive
    analysis. The `full` extra installs the complete toolkit.
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

The base library supports scoring. Use the named profiles above for individual
tools and `full` for the complete toolkit. Model execution also needs its backend
runtime; the `infer` profile supports configuration and dry runs without GPU libraries. See the
[MSRB Python example](../../src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md#python-scoring-without-a-campaign).
The `full` extra includes DenseArrays from PyPI. Checkout installs resolve its
version through `uv.lock`. For dependency changes, use
[dependency maintenance](dependencies.md).

### Development tools (when needed)
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

### System binaries with pixi (when needed)
Run this when a workflow needs pixi-managed tools such as MEME/FIMO.

```bash
pixi install --locked
# Verify MEME/FIMO is available in the pixi environment.
pixi run -- fimo --version
```

Dependency maintenance operations are documented in **[Dependency maintenance](dependencies.md)**.

### GPU extra (`infer-evo2`)
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
- flash-attn is currently sdist-only in `uv.lock`. The SCC runbook sets
  `FLASH_ATTENTION_FORCE_BUILD=TRUE` to compile that source rather than letting
  its build script download a precompiled wheel.
- Use `infer adapters list` to inspect registered models. Evo2 feature bundles support `evo2_7b` and `evo2_20b`; see the [provider reference](../../src/dnadesign/infer/docs/reference/evo2-provider.md).

### Troubleshooting quick checks
Use these checks before deeper debugging:

- If `uv` is not found after install, start a new shell and rerun `uv --version`.
- If matplotlib cache warnings appear, set `MPLCONFIGDIR` to a writable path:

```bash
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
```

- If you are on Windows, confirm commands are running inside WSL2 (`uname -a` should report Linux).

### Continue with workflow docs
- If you want the shortest shell-first route after baseline verification, start with `uv run ops catalog list --simple`.
- For workflow execution from the docs map, use [docs/README.md](../README.md).
- For BU SCC workflows, follow this order: [BU SCC quickstart](../bu-scc/setup/quickstart.md), [BU SCC install bootstrap](../bu-scc/setup/install.md), then [BU SCC batch plus Notify runbook](../bu-scc/runbooks/batch-notify.md).
