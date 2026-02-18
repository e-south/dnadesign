## Dependency maintenance

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-18

This document is the canonical home for dependency management in this repository, including Python packages and pixi-managed system tools. Read it when adding, removing, or updating dependencies so lockfiles and runtime expectations stay aligned.

### Python dependencies (uv)
This section covers Python package lifecycle operations managed by `uv`.

```bash
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

### System dependencies (pixi)
This section covers non-Python binaries pinned via `pixi.toml`.

Use pixi for toolchains such as MEME Suite that are required by some workflows but are not Python packages.

```bash
# Install or update the pinned pixi environment.
pixi install

# Run a task that depends on pixi-managed binaries.
pixi run cruncher -- doctor -c src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml

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
