## Installation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28

This guide is the first-run setup path for local development and CLI usage: confirm platform/version contracts, run the required install and baseline verification, then add optional branches only when needed.

### 1) Choose your platform lane

| Environment | Status | Required lane |
| --- | --- | --- |
| macOS (`arm64`, `x86_64`) | Supported | Use this guide directly. |
| Linux (`x86_64`) | Supported | Use this guide directly. |
| Windows 10/11 via WSL2 | Supported | Run all commands inside the WSL Linux shell. |
| Native Windows (PowerShell/CMD) | Not supported | Use WSL2 instead. |

Reason:
- `pyproject.toml` `[tool.uv] environments` targets `darwin` and `linux x86_64`.
- `pixi.toml` `[workspace] platforms` targets `osx-arm64`, `osx-64`, and `linux-64`.

### 2) Confirm version contracts
Treat these as installation requirements:

- Python: `>=3.12,<3.13` (`pyproject.toml` `[project] requires-python`)
- uv: `>=0.9.18,<0.10` (`pyproject.toml` `[tool.uv] required-version`)

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

You are done with base installation when:
- the import check prints `ok`
- `uv run usr --help` exits successfully

### 4) Optional branch: development tools
Run this only when you will lint or test locally.

```bash
uv sync --locked --group dev
uv run ruff --version
uv run pytest --version
```

### 5) Optional branch: system binaries with pixi
Run this only when a workflow needs pixi-managed tools such as MEME/FIMO.

```bash
pixi install --locked
# Verify MEME/FIMO is available in the pixi environment.
pixi run -- fimo --version
```

Dependency maintenance operations (add/update/remove) are documented in **[docs/dependencies.md](dependencies.md)**.

### 6) Optional branch: GPU extra (`infer-evo2`)
This branch is Linux `x86_64` only.

```bash
uv sync --locked --extra infer-evo2
```

### 7) Troubleshooting quick checks
Use these checks before deeper debugging:

- If `uv` is not found after install, start a new shell and rerun `uv --version`.
- If matplotlib cache warnings appear, set `MPLCONFIGDIR` to a writable path:

```bash
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
```

- If you are on Windows, confirm commands are running inside WSL2 (`uname -a` should report Linux).

### 8) Next paths by goal
- For workflow execution paths, use [docs/README.md](README.md).
- For BU SCC workflows, use [BU SCC quickstart](bu-scc/quickstart.md), [BU SCC install bootstrap](bu-scc/install.md), and [BU SCC batch plus Notify runbook](bu-scc/batch-notify.md).
