## DenseGen CLI reference

Use this page when you need exact command behavior and flag names.
For end-to-end guided runs, use:
- [binding-sites baseline demo](../tutorials/demo_tfbs_baseline.md)
- [sampling baseline demo](../tutorials/demo_sampling_baseline.md)

### Contents
This section indexes the command surfaces covered in this reference.

- [How to inspect CLI surface quickly](#how-to-inspect-cli-surface-quickly)
- [Config resolution](#config-resolution)
- [`dense validate-config`](#dense-validate-config)
- [`dense inspect inputs`](#dense-inspect-inputs)
- [`dense inspect plan`](#dense-inspect-plan)
- [`dense inspect config`](#dense-inspect-config)
- [`dense inspect run`](#dense-inspect-run)
- [`dense stage-a build-pool`](#dense-stage-a-build-pool)
- [`dense stage-b build-libraries`](#dense-stage-b-build-libraries)
- [`dense workspace init`](#dense-workspace-init)
- [`dense workspace where`](#dense-workspace-where)
- [`dense run`](#dense-run)
- [`dense campaign-reset`](#dense-campaign-reset)
- [`dense plot`](#dense-plot)
- [`dense ls-plots`](#dense-ls-plots)
- [`dense notebook generate`](#dense-notebook-generate)
- [`dense notebook run`](#dense-notebook-run)

### How to inspect CLI surface quickly
This section provides the fastest way to discover active command groups and flags.

```bash
# Show top-level commands.
uv run dense --help

# Show command-specific flags.
uv run dense inspect run --help
uv run dense stage-a build-pool --help
uv run dense stage-b build-libraries --help
uv run dense notebook run --help
```

### Config resolution
This section defines how DenseGen resolves config paths when `-c/--config` is present or omitted.

- `-c, --config PATH` is supported globally and on command groups.
- If `--config` is omitted, DenseGen resolves config in this order:
  1. `DENSEGEN_CONFIG_PATH`
  2. `./config.yaml`
- DenseGen does not scan parent directories.
- If neither path exists, the command fails fast.
- Input paths resolve relative to the config file directory.
- Output paths must stay inside `outputs/` under `densegen.run.root`.
- Config must include `densegen.schema_version` and `densegen.run`.

Operational guidance:
- CI/HPC: pass `-c /abs/path/to/config.yaml`.
- Local workspace shell: `./config.yaml` is usually enough.

### `dense validate-config`

Validates schema and config sanity checks.

Key options:
- `--probe-solver / --no-probe-solver`
- `-c, --config PATH`

### `dense inspect inputs`

Shows resolved input sources plus Stage-A pool status.

Key options:
- `--verbose`
- `--absolute`
- `--show-motif-ids`
- `-c, --config PATH`

### `dense inspect plan`

Shows resolved plan quotas by plan item.

Key options:
- `-c, --config PATH`

### `dense inspect config`

Shows resolved output wiring, Stage-A/Stage-B sampling settings, and solver settings.

Key options:
- `--show-constraints`
- `--probe-solver`
- `--absolute`
- `-c, --config PATH`

### `dense inspect run`

Summarizes a run manifest, or lists workspaces under a root.

Key options:
- `--run, -r PATH`
- `--root PATH`
- `--limit INTEGER`
- `--all`
- `--absolute`
- `--verbose`
- `--library`
- `--show-tfbs`
- `--show-motif-ids`
- `--events`
- `--usr-events-path` (prints USR `.events.log` path and exits)
- `-c, --config PATH`

Notes:
- `--root` and `--usr-events-path` are mutually exclusive.
- `--usr-events-path` requires a config that writes to USR outputs.

### `dense stage-a build-pool`

Builds Stage-A TFBS pools and pool metadata.

Key options:
- `--out TEXT` (default: `outputs/pools`)
- `--n-sites INTEGER` (override Stage-A `n_sites` for PWM inputs)
- `--batch-size INTEGER` (override Stage-A mining batch size)
- `--max-seconds FLOAT` (override Stage-A mining max seconds)
- `--input, -i TEXT` (repeatable)
- `--fresh`
- `--show-motif-ids`
- `--verbose`
- `-c, --config PATH`

Outputs:
- `outputs/pools/pool_manifest.json`
- `outputs/pools/<input>__pool.parquet`
- candidate artifacts (when candidate logging is enabled)

### `dense stage-b build-libraries`

Builds Stage-B libraries from Stage-A pools.

Key options:
- `--out TEXT` (default: `outputs/libraries`)
- `--pool PATH` (default: `outputs/pools`)
- `--input, -i TEXT` (repeatable)
- `--plan, -p TEXT` (repeatable)
- `--overwrite`
- `--append`
- `--show-motif-ids`
- `-c, --config PATH`

Behavior:
- If library artifacts already exist, pass either `--append` or `--overwrite`.
- `--append` preserves previous artifacts and requires manifest compatibility.

Outputs:
- `outputs/libraries/library_builds.parquet`
- `outputs/libraries/library_members.parquet`
- `outputs/libraries/library_manifest.json`

### `dense workspace init`

Creates a run workspace with `config.yaml`, `inputs/`, and `outputs/` subfolders.

Key options:
- `--id, -i TEXT` (required)
- `--root PATH`
- `--from-workspace TEXT`
- `--from-config PATH`
- `--copy-inputs / --no-copy-inputs`
- `--output-mode local|usr|both`

Notes:
- `--output-mode usr|both` seeds `outputs/usr_datasets/registry.yaml` when a seed file is available.
- `--output-mode usr|both` sets `output.usr.dataset` to the workspace id so each initialized workspace writes to its own USR dataset path.

### `dense workspace where`

Shows effective workspace roots that `workspace init` will use.

Key options:
- `--format text|json`

### `dense run`

Runs sampling, solving, output writing, and optional plotting.

Key options:
- `--no-plot`
- `--fresh`
- `--resume`
- `--extend-quota INTEGER`
- `--log-file PATH`
- `--show-tfbs`
- `--show-solutions`
- `-c, --config PATH`

Notes:
- If prior run outputs exist, default behavior is resume-like unless `--fresh` is set.
- Missing/stale Stage-A pools are rebuilt automatically.
- For FIMO-backed inputs, ensure `fimo` is available (for example via `pixi run ...`).

### `dense campaign-reset`

Deletes run outputs while preserving config and inputs.

Key options:
- `-c, --config PATH`

### `dense plot`

Generates plots from existing outputs.

Key options:
- `--only NAME1,NAME2`
- `--absolute`
- `-c, --config PATH`

### `dense ls-plots`

Lists available plot names and descriptions.

Key options:
- `-c, --config PATH` (optional config validation before listing)

### `dense notebook generate`

Generates a workspace-scoped marimo notebook for the run.

Key options:
- `--out PATH` (default: `<run_root>/outputs/notebooks/densegen_run_overview.py`)
- `--force`
- `--absolute`
- `-c, --config PATH`

Notes:
- Uses one records source parquet file, selected by output wiring:
  - if `output.targets` has one sink, use that sink (`parquet` or `usr`)
  - if `output.targets` has both sinks, use `plots.source`
- Source path resolution:
  - `parquet` source -> `output.parquet.path`
  - `usr` source -> `<output.usr.root>/<output.usr.dataset>/records.parquet`

### `dense notebook run`

Launches a DenseGen marimo notebook.

Key options:
- `--path PATH` (default: `<run_root>/outputs/notebooks/densegen_run_overview.py`)
- `--mode run|edit` (default: `run`)
- `--headless` (run mode only)
- `--absolute`
- `-c, --config PATH`

Notes:
- `--mode run` launches a read-only notebook app for analysis.
- `--mode edit` launches editable marimo cells.
- `--headless` suppresses browser auto-open for remote/non-GUI shells.

---

@e-south
