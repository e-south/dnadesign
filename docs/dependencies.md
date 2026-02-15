## Maintaining dependencies


## Contents
- [Maintaining dependencies](#maintaining-dependencies)
- [Non-Python system dependencies (pixi)](#non-python-system-dependencies-pixi)

### Non-Python system dependencies (pixi)

Some tools (e.g., MEME Suite for **cruncher**) are system-level binaries and are not installed via `uv`. We use [pixi](https://pixi.sh/) to pin and manage those dependencies in a reproducible way.

This repo includes a minimal `pixi.toml` at the root with a `cruncher` task that wraps `uv run cruncher`.

Typical commands:

```bash
# Install/resolve the toolchain defined in pixi.toml
pixi install

# Run Cruncher with system tools on PATH (pixi env + uv Python env)
# Note: pixi inserts `--` before task args, so put -c/--config after the subcommand.
pixi run cruncher -- --help
pixi run cruncher -- doctor -c src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml

# Add or update a system dependency
pixi add bioconda::meme
pixi update

# Generate/update the lockfile
pixi lock
```

Optional: define a short runner in your shell (zsh doesn’t split multi-word variables):

```bash
cruncher() { pixi run cruncher -- "$@"; }
# Example: cruncher doctor -c path/to/config.yaml
```

By default, pixi installs into `.pixi/`. If you prefer not to use `pixi run`, export an absolute `MEME_BIN` so Cruncher can find the MEME Suite binaries:

```bash
export DNADESIGN_ROOT="$(git rev-parse --show-toplevel)"
export MEME_BIN="$DNADESIGN_ROOT/.pixi/envs/default/bin"
```

If you prefer `motif_discovery.tool_path`, use an absolute path (or a path relative to your `config.yaml` location).

Commit `pixi.toml` and `pixi.lock` alongside any updates.

---

@e-south
