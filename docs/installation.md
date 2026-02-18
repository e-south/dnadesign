## Installation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-18

This document covers initial repository setup for local development and CLI usage. Read it when bootstrapping a new environment; detailed dependency maintenance guidance lives in `docs/dependencies.md`.

### Install uv
This section installs the package manager used by this monorepo.

```bash
# Install uv on macOS/Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone the repository
This section checks out the project source locally.

```bash
# Clone the repository and enter the project directory.
git clone https://github.com/e-south/dnadesign.git
cd dnadesign
```

### Sync the project environment
This section creates or updates the project virtual environment from the committed lockfile.

```bash
# Ensure Python 3.12 is available to uv.
uv python install 3.12

# Create or sync the project environment from uv.lock.
uv sync --locked

# Run a basic import sanity check.
uv run python -c "import dnadesign, pandas, pyarrow; print('ok')"
```

### Optional dev-tool setup
This section installs lint/test tooling used during development.

```bash
# Install optional dev dependencies.
uv sync --locked --group dev

# Confirm linter and test runner are available.
uv run ruff --version
uv run pytest -q
```

### Run CLI entrypoints
This section shows the standard non-activated workflow for project CLIs.

```bash
# Show help for key repository CLIs.
uv run usr --help
uv run dense --help
uv run notify --help
uv run cruncher --help
uv run baserender --help
```

### BU SCC links
This section points to the canonical BU SCC setup docs.

- **[BU SCC quickstart](bu-scc/quickstart.md)**
- **[BU SCC install bootstrap](bu-scc/install.md)**
- **[BU SCC batch plus Notify runbook](bu-scc/batch-notify.md)**

### System dependencies note
This section keeps installation concise and defers pixi specifics to the canonical dependency doc.

System dependencies managed with pixi are documented in **[docs/dependencies.md](dependencies.md)**.
