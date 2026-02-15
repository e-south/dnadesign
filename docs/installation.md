## Installation


## Contents
- [Installation](#installation)
- [Install uv](#install-uv)
- [Clone the repo](#clone-the-repo)
- [Local install](#local-install)
- [Dev tools (tests + lint)](#dev-tools-tests-lint)
- [Running dnadesign CLIs](#running-dnadesign-clis)
- [System dependencies (pixi)](#system-dependencies-pixi)

### Install uv

macOS/Linux (for other OSs see the uv docs):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# ensure your uv bin dir is on PATH
```

### Clone the repo

```bash
git clone https://github.com/e-south/dnadesign.git
cd dnadesign
```

### Local install

This is the default way to start working with most pipelines. For BU SCC CUDA/GPU setup and batch operations, see [BU SCC quickstart](hpc/bu_scc_quickstart.md), [BU SCC install bootstrap](hpc/bu_scc_install.md), and [BU SCC batch + Notify runbook](hpc/bu_scc_batch_notify.md).

1) Ensure Python 3.12 is available:

```bash
uv python install 3.12
```

2) Create/sync the environment from the committed lockfile:

```bash
uv sync --locked
```

3) Sanity checks:

```bash
uv run python -c "import dnadesign, pandas, pyarrow; print('ok')"
uv run usr ls || true
```

### HPC (BU SCC)

- [BU SCC quickstart](hpc/bu_scc_quickstart.md)
- [BU SCC install bootstrap](hpc/bu_scc_install.md)
- [BU SCC batch + Notify runbook](hpc/bu_scc_batch_notify.md)

### Dev tools (tests + lint)

Dev tooling is opt-in via a dependency group:

```bash
uv sync --locked --group dev
uv run ruff --version
uv run pytest -q
```

### Running dnadesign CLIs

This repo defines console scripts that can be run via:

#### Option A: no `.venv` activation

```bash
uv run usr --help
uv run usr ls

uv run opal --help
uv run dense --help
uv run infer --help
uv run cluster --help
uv run permuter --help
uv run baserender --help
```

#### Option B: traditional `.venv` activation

```bash
source .venv/bin/activate
usr --help
usr ls
deactivate
```

### System dependencies (pixi)

Some subpackages rely on non-Python tools (e.g., MEME Suite for Cruncher). These are managed separately via `pixi`, using the repo-level `pixi.toml`.

Division of labor:

- **uv** is the source of truth for Python packages and the project virtualenv.
- **pixi** pins system binaries (e.g., MEME Suite) that are not Python packages.
- For MEME-dependent **cruncher** workflows, **use `pixi run cruncher -- <subcommand> ...`** so
  system tools are on `PATH` while `uv` keeps Python deps synced under the hood. (Pixi inserts
  `--` before task args, so put global options like `-c` after the subcommand.)

Install pixi (one-time, system-level):

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Install the pinned system toolchain (recommended when using **cruncher** + MEME Suite):

```bash
pixi install
```

Optional: define a short runner in your shell (zsh doesn’t split multi-word variables):

```bash
cruncher() { pixi run cruncher -- "$@"; }
# Note: place -c/--config after the subcommand when using pixi:
# cruncher doctor -c path/to/config.yaml
```

Run MEME-dependent workflows via pixi so system tools are on `PATH` while `uv`
keeps Python deps synced:

```bash
pixi run cruncher -- --help
pixi run cruncher -- doctor -c src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml
```

If you update `pixi.toml`, regenerate the lock:

```bash
pixi lock
```

See `docs/dependencies.md` for more detail and maintenance commands.

---

@e-south
