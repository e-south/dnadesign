## Installation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28

This document is the setup path for new machines. It starts with the shortest runnable install, then adds optional paths for development tools, pixi-managed system binaries, and Linux GPU extras.

### Platform support
Use this table to pick the correct setup path before running commands.

| Environment | Status | Notes |
| --- | --- | --- |
| macOS (`arm64`, `x86_64`) | Supported | Run the quick install path below. |
| Linux (`x86_64`) | Supported | Primary `uv.lock` target. |
| Windows 10/11 via WSL2 | Supported | Run all commands inside your WSL Linux shell. |
| Native Windows (PowerShell/CMD) | Not supported | `pyproject.toml` `tool.uv.environments` and `pixi.toml` platforms do not include Windows. |

### Version contracts
These versions are constrained by repository configuration and should be treated as installation requirements.

- Python: `>=3.12,<3.13` (`pyproject.toml` `[project] requires-python`)
- uv: `>=0.9.18,<0.10` (`pyproject.toml` `[tool.uv] required-version`)
- pixi platforms: `osx-arm64`, `osx-64`, `linux-64` (`pixi.toml` `[workspace] platforms`)

### Quick install (Linux/macOS/WSL)
Run this path first. It sets up a reproducible base environment from committed lockfiles.

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

### Environment note (matplotlib cache)
Some CLI help paths import matplotlib. On systems where `~/.matplotlib` is not writable, set `MPLCONFIGDIR` to avoid cache warnings and slower first-run behavior.

```bash
# Use a writable cache location for matplotlib-backed CLI commands.
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
```

### Optional dev-tool setup
Install this only when you will run lint/test commands locally.

```bash
uv sync --locked --group dev
uv run ruff --version
uv run pytest --version
```

### Optional pixi system dependencies
Use this when workflows need system binaries such as MEME/FIMO.

```bash
pixi install --locked
pixi run cruncher -- doctor -c src/dnadesign/cruncher/workspaces/demo_pairwise/configs/config.yaml
```

### Optional GPU extra (`infer-evo2`)
This extra is Linux `x86_64` only.

```bash
uv sync --locked --extra infer-evo2
```

### BU SCC links
This section points to the canonical BU SCC setup docs.

- **[BU SCC quickstart](bu-scc/quickstart.md)**
- **[BU SCC install bootstrap](bu-scc/install.md)**
- **[BU SCC batch plus Notify runbook](bu-scc/batch-notify.md)**

### System dependencies note
This section keeps installation concise and defers pixi specifics to the canonical dependency doc.

System dependencies managed with pixi are documented in **[docs/dependencies.md](dependencies.md)**.
