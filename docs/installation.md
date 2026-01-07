## Installation

`dnadesign` is managed with [uv](https://docs.astral.sh/uv/):

- `pyproject.toml` declares dependencies (runtime + optional extras)
- `uv.lock` is the fully pinned dependency graph
- `.venv/` is the project virtual environment (created automatically by uv)

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

This is the default way to start working with most pipelines. For CUDA/GPU notes, see `docs/INSTALL_BU_SCC.md`.

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

---

@e-south
