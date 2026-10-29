## Dependency maintenance

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-29

This document is the canonical home for dependency management, including Python packages (`uv`) and system binaries (`pixi`).

### Platform and version scope
These constraints come from repository configuration and define supported dependency targets.

| Source | Constraint | Operational meaning |
| --- | --- | --- |
| `pyproject.toml` `[project] requires-python` | `>=3.12,<3.13` | Use Python 3.12 for all local/CI environments. |
| `pyproject.toml` `[tool.uv] required-version` | `>=0.12.3,<0.13` | Keep `uv` within the pinned resolver/runtime range. |
| `pyproject.toml` `[tool.uv] environments` | `darwin`, `linux x86_64` | `uv.lock` is resolved for macOS and Linux x86_64. |
| `pixi.toml` `[workspace] platforms` | `osx-arm64`, `osx-64`, `linux-64` | `pixi` environments are supported on macOS and Linux. |

Windows users should run this repository inside WSL2 and use the Linux path.

### Python dependencies (uv)
This section covers Python package lifecycle operations managed by `uv`.

```bash
# Sync from lockfile (baseline environment).
uv sync --locked

# Add a dependency used by tool APIs or CLIs.
uv add --optional full <package>

# Add a dependency to a named group (for example, dev tools).
uv add --group dev <package>

# Remove a tool dependency from pyproject and lockfile.
uv remove --optional full <package>
```

Use unqualified `uv add` only when the lightweight installed scoring API needs
the dependency. If you edit `pyproject.toml` directly, regenerate the lockfile:

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

Cruncher dependency checks in this repository use:
`src/dnadesign/cruncher/workspaces/demo_pairwise/configs/config.yaml`
as the reference workspace config path.

```bash
# Install or update the pinned pixi environment.
pixi install --locked

# Verify MEME/FIMO is available in the pixi environment.
pixi run -- fimo --version

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

### Release security gate

The release workflow runs `uv audit --locked` across every extra and dependency
group, including platform-specific pins. It also audits the resolved build
requirements and installed release environment with `pip-audit`. Known
vulnerabilities and audit errors block publication. PR dependency review remains
a separate check on changed dependencies.

```bash
uv audit --locked
```

The [uv audit command](https://docs.astral.sh/uv/reference/cli/#uv-audit) checks
published advisories and adverse package statuses without installing GPU runtimes.
Its JSON output is retained as release evidence. This does not establish that every
reported vulnerability is reachable through DNADesign.

The optional `infer-evo2` profile requires FlashAttention 2.8.3.post2 or newer.
The [upstream release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post2)
restricts checkpoint deserialization. Continue auditing the resolved environment before each release;
a clean dependency audit does not establish GPU execution coverage.
