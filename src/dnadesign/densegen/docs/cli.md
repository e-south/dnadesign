# DenseGen CLI

DenseGen ships a Typer CLI via the `dense` console script.

## Invocation

```bash
uv run dense --help
# or
python -m dnadesign.densegen.src.cli --help
```

## Config option

- `-c, --config PATH` — config YAML path. Defaults to `src/dnadesign/densegen/runs/smoke_v2/config.yaml` inside the package.
  - May be passed **globally** (`dense -c path validate`) or **per command**
    (`dense validate -c path`).

Input paths resolve against the **config file directory** (no fallback search).
Output, logs, and plots must resolve inside `densegen.run.root` (run-scoped I/O).
USR integration is optional; Parquet-only runs should not require USR modules.
Config files must include `densegen.schema_version` (currently `2.1`) and `densegen.run`.

## Commands

### `dense validate`
Validate the config YAML (schema + sanity checks). Fails fast on unknown keys or invalid values.
Options:
- `--probe-solver` — also probe the solver backend (fails fast if unavailable).

### `dense plan`
Print the resolved quota plan per constraint bucket.

### `dense describe`
Summarize resolved inputs, outputs, plan items, and solver settings.
Options:
- `--show-constraints` — print full fixed elements per plan item.
- `--probe-solver` — verify the solver backend before reporting.

### `dense run`
Run the full generation pipeline.

Options:
- `--no-plot` — skip auto-plotting even if `plots` is configured in YAML.
- `--log-file PATH` — override the log file path. Otherwise DenseGen writes
  to `logging.log_dir/<run_id>.log` inside the run directory. The override path
  must still resolve inside `densegen.run.root`.

### `dense plot`
Generate plots from existing outputs.

Options:
- `--only NAME1,NAME2` — run a subset of plots by name.

### `dense ls-plots`
List available plot names and descriptions.

### `dense stage`
Stage a new run directory with `config.yaml`, `inputs/`, `outputs/`, `logs/`, and `plots/`.
Options:
- `--id` — run identifier (directory name).
- `--root` — runs root directory (default: package `runs/` directory).
- `--template` — template config YAML to copy.
- `--copy-inputs` — copy file-based inputs into `run/inputs` and rewrite paths.

### `dense ls-runs`
List runs under a root directory and summarize artifacts.
Options:
- `--root` — runs root directory (default: package `runs/` directory).
- `--limit` — maximum number of runs to display.
- `--all` — include directories without `config.yaml`.

## Examples

```bash
uv run dense validate -c src/dnadesign/densegen/runs/smoke_v2/config.yaml
uv run dense plan     -c src/dnadesign/densegen/runs/smoke_v2/config.yaml
uv run dense describe -c src/dnadesign/densegen/runs/smoke_v2/config.yaml
uv run dense run      -c src/dnadesign/densegen/runs/smoke_v2/config.yaml
uv run dense plot     -c src/dnadesign/densegen/runs/smoke_v2/config.yaml --only tf_usage,tf_coverage
```

Smoke run (small, Parquet-only config):

```bash
uv run dense run -c src/dnadesign/densegen/runs/smoke_v2/config.yaml --no-plot
```
