## DenseGen CLI

DenseGen ships a Typer CLI via the `dense` console script. The CLI is strict: config paths are
explicit, inputs resolve relative to the config file, and all outputs/logs/plots must stay inside
the run root. USR is optional and is only imported when configured.

### Contents
- [Invocation](#invocation) - how to call the CLI.
- [Config option](#config-option) - global or per-command config path.
- [Commands](#commands) - validate, plan, describe, run, plot, and utilities.
- [`dense validate`](#dense-validate) - schema and sanity checks.
- [`dense plan`](#dense-plan) - resolved quota plan.
- [`dense describe`](#dense-describe) - resolved inputs, outputs, and solver.
- [`dense run`](#dense-run) - end-to-end generation.
- [`dense plot`](#dense-plot) - render plots from outputs.
- [`dense ls-plots`](#dense-ls-plots) - list available plots.
- [`dense stage`](#dense-stage) - scaffold a run directory.
- [`dense summarize`](#dense-summarize) - summarize run_manifest.json or list runs.
- [Examples](#examples) - common command sequences.

---

### Invocation

```bash
uv run dense --help
# or
python -m dnadesign.densegen.src.cli --help
```

---

### Config option

- `-c, --config PATH` - config YAML path. Defaults to
  `src/dnadesign/densegen/runs/demo/config.yaml` inside the package.
  - May be passed globally (`dense -c path validate`) or per command
    (`dense validate -c path`).

Input paths resolve against the config file directory. Output, logs, and plots must resolve
inside `densegen.run.root` (run-scoped I/O). Config files must include `densegen.schema_version`
(currently `2.1`) and `densegen.run`.

---

### Commands

### `dense validate`
Validate the config YAML (schema + sanity checks). Fails fast on unknown keys or invalid values.
Options:
- `--probe-solver` - also probe the solver backend (fails fast if unavailable).

---

#### `dense plan`
Print the resolved quota plan per constraint bucket.

---

#### `dense describe`
Summarize resolved inputs, outputs, plan items, and solver settings.
Options:
- `--show-constraints` - print full fixed elements per plan item.
- `--probe-solver` - verify the solver backend before reporting.

---

#### `dense run`
Run the full generation pipeline.

Options:
- `--no-plot` - skip auto-plotting even if `plots` is configured in YAML.
- `--log-file PATH` - override the log file path. Otherwise DenseGen writes
  to `logging.log_dir/<run_id>.log` inside the run directory. The override path
  must still resolve inside `densegen.run.root`.

---

#### `dense plot`
Generate plots from existing outputs.

Options:
- `--only NAME1,NAME2` - run a subset of plots by name.

---

#### `dense ls-plots`
List available plot names and descriptions.

---

#### `dense stage`
Stage a new run directory with `config.yaml`, `inputs/`, `outputs/`, `logs/`, and `plots/`.
Options:
- `--id` - run identifier (directory name).
- `--root` - runs root directory (default: package `runs/` directory).
- `--template` - template config YAML to copy.
- `--copy-inputs` - copy file-based inputs into `run/inputs` and rewrite paths.

---

#### `dense summarize`
Summarize a run manifest (`run_manifest.json`).
Options:
- `--run` - run directory (defaults to `densegen.run.root` from config).
- `--root` - list runs under a root directory (replacement for `dense ls-runs`).
- `--limit` - limit runs displayed when using `--root`.
- `--all` - include directories without `config.yaml` when using `--root`.
- `--config` - config path (used to resolve run root when `--run` is not set).
- `--verbose` - show failure breakdown columns (constraint filters + duplicate solutions).

Note: `dense ls-runs` is a legacy alias and may be removed in the future.

---

### Examples

```bash
uv run dense validate -c src/dnadesign/densegen/runs/demo/config.yaml
uv run dense plan     -c src/dnadesign/densegen/runs/demo/config.yaml
uv run dense describe -c src/dnadesign/densegen/runs/demo/config.yaml
uv run dense run      -c src/dnadesign/densegen/runs/demo/config.yaml
uv run dense plot     -c src/dnadesign/densegen/runs/demo/config.yaml --only tf_usage,tf_coverage
uv run dense summarize --run src/dnadesign/densegen/runs/demo
uv run dense summarize --root src/dnadesign/densegen/runs
```

Demo run (small, Parquet-only config):

```bash
uv run dense run -c src/dnadesign/densegen/runs/demo/config.yaml --no-plot
```

---

@e-south
