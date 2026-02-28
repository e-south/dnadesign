## Dependency maintenance

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28

This document is the canonical home for dependency management, including Python packages (`uv`) and system binaries (`pixi`).

### Platform and version scope
These constraints come from repository configuration and define supported dependency targets.

| Source | Constraint | Operational meaning |
| --- | --- | --- |
| `pyproject.toml` `[project] requires-python` | `>=3.12,<3.13` | Use Python 3.12 for all local/CI environments. |
| `pyproject.toml` `[tool.uv] required-version` | `>=0.9.18,<0.10` | Keep `uv` within the pinned resolver/runtime range. |
| `pyproject.toml` `[tool.uv] environments` | `darwin`, `linux x86_64` | `uv.lock` is resolved for macOS and Linux x86_64. |
| `pixi.toml` `[workspace] platforms` | `osx-arm64`, `osx-64`, `linux-64` | `pixi` environments are supported on macOS and Linux. |

Windows users should run this repository inside WSL2 and use the Linux path.

### Python dependencies (uv)
This section covers Python package lifecycle operations managed by `uv`.

```bash
# Sync from lockfile (baseline environment).
uv sync --locked

# Add a runtime dependency to the default dependency set.
uv add <package>

# Add a dependency to a named group (for example, dev tools).
uv add --group dev <package>

# Remove a dependency from pyproject and lockfile.
uv remove <package>
```

If you edit `pyproject.toml` directly, regenerate the lockfile:

```bash
# Recompute uv.lock from pyproject constraints.
uv lock
```

Optional sync targets:

```bash
# Install lint/test groups.
uv sync --locked --group dev

# Linux x86_64 only: install Evo2 GPU extra.
uv sync --locked --extra infer-evo2
```

### System dependencies (pixi)
This section covers non-Python binaries pinned via `pixi.toml`.

Use pixi for toolchains such as MEME Suite that are required by some workflows but are not Python packages.

```bash
# Install or update the pinned pixi environment.
pixi install --locked

# Run a task that depends on pixi-managed binaries.
pixi run cruncher -- doctor -c src/dnadesign/cruncher/workspaces/demo_pairwise/configs/config.yaml

# Add or update a pixi-managed system package.
pixi add bioconda::meme
pixi update

# Regenerate pixi lockfile after dependency changes.
pixi lock
```

### Lockfile and commit checklist
This section ensures dependency changes are reproducible for other operators.

- Commit `pyproject.toml` and `uv.lock` for Python dependency changes.
- Commit `pixi.toml` and `pixi.lock` for system dependency changes.
- Run `uv sync --locked` after pulling dependency updates.
