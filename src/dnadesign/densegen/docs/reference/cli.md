## DenseGen CLI

DenseGen ships a Typer CLI via the `dense` console script. The CLI is strict: config paths are
explicit, inputs resolve relative to the config file, and all outputs/logs must stay inside
the run root. USR is optional and is only imported when configured.

### Contents
- [Invocation](#invocation) - how to call the CLI.
- [Config option](#config-option) - global or per-command config path.
- [Commands](#commands) - validate, inspect, stage helpers, run, plot, report.
- [`dense validate-config`](#dense-validate-config) - schema and sanity checks.
- [`dense inspect inputs`](#dense-inspect-inputs) - resolved inputs + PWM sampling summary.
- [`dense inspect plan`](#dense-inspect-plan) - resolved quota plan.
- [`dense inspect config`](#dense-inspect-config) - resolved inputs/outputs/solver details.
- [`dense inspect run`](#dense-inspect-run) - summarize run manifests or list workspaces.
- [`dense stage-a build-pool`](#dense-stage-a-build-pool) - build TFBS pools (Stage‑A).
- [`dense stage-b build-libraries`](#dense-stage-b-build-libraries) - build solver libraries (Stage‑B).
- [`dense workspace init`](#dense-workspace-init) - scaffold a workspace.
- [`dense run`](#dense-run) - end-to-end generation.
- [`dense plot`](#dense-plot) - render plots from outputs.
- [`dense ls-plots`](#dense-ls-plots) - list available plots.
- [`dense report`](#dense-report) - write audit-grade report summary.
- [Examples](#examples) - common command sequences.

---

### Invocation

```bash
uv run dense --help
# or
python -m dnadesign.densegen --help
```

---

### Config option

- `-c, --config PATH` - config YAML path. Defaults to
  `src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml` inside the package.
  - May be passed globally (`dense -c path inspect inputs`) or per command
    (`dense inspect inputs -c path`).

Input paths resolve against the config file directory. Outputs and logs must resolve
inside `densegen.run.root` (run-scoped I/O). Config files must include `densegen.schema_version`
(currently `2.4`) and `densegen.run`.

---

### Commands

### `dense validate-config`
Validate the config YAML (schema + sanity checks). Fails fast on unknown keys or invalid values.

Options:
- `--probe-solver` - also probe the solver backend (fails fast if unavailable).

---

#### `dense inspect inputs`
Print resolved inputs plus a PWM sampling summary (Stage‑A details).

---

#### `dense inspect plan`
Print the resolved quota plan per constraint bucket.

---

#### `dense inspect config`
Summarize resolved inputs, outputs, plan items, and solver settings.

Options:
- `--show-constraints` - print full fixed elements per plan item.
- `--probe-solver` - verify the solver backend before reporting.

---

#### `dense inspect run`
Summarize a run manifest (`outputs/meta/run_manifest.json`) or list workspaces.

Options:
- `--run` - workspace directory (defaults to `densegen.run.root` from config).
- `--root` - list workspaces under a root directory.
- `--limit` - limit workspaces displayed when using `--root`.
- `--all` - include directories without `config.yaml` when using `--root`.
- `--config` - config path (used to resolve run root when `--run` is not set).
- `--verbose` - show failure breakdown columns (constraint filters + duplicate solutions).
- `--library` - include offered-vs-used summaries (TF/TFBS usage).
- `--top` - number of rows to show in library summaries.
- `--by-library/--no-by-library` - group library summaries per build attempt.
- `--top-per-tf` - limit TFBS rows per TF when summarizing.
- `--show-library-hash/--short-library-hash` - toggle full vs short library hashes.

Tip:
- For large runs, prefer `--no-by-library` or lower `--top`/`--top-per-tf` to keep output readable.

---

#### `dense stage-a build-pool`
Build Stage‑A TFBS pools from inputs and write a pool manifest.

Options:
- `--out` - output directory relative to run root (default: `outputs/pools`).
- `--input/-i` - input name(s) to build (defaults to all).
- `--overwrite` - overwrite existing pool files.

Outputs:
- `pool_manifest.json`
- `<input>__pool.parquet` per input

---

#### `dense stage-b build-libraries`
Build Stage‑B libraries (one per input + plan) from Stage‑A pools.

Options:
- `--out` - output directory relative to run root (default: `outputs/libraries`).
- `--pool` - pool directory from `stage-a build-pool` (defaults to `outputs/pools` in the workspace).
- `--input/-i` - input name(s) to build (defaults to all).
- `--plan/-p` - plan item name(s) to build (defaults to all).
- `--overwrite` - overwrite existing library artifacts.

Outputs:
- `library_builds.parquet`
- `library_members.parquet`
- `library_manifest.json`

---

#### `dense workspace init`
Stage a new workspace with `config.yaml`, `inputs/`, `outputs/`, plus `outputs/logs/` and `outputs/meta/`.

Options:
- `--id` - run identifier (directory name).
- `--root` - workspaces root directory (default: package `workspaces/` directory).
- `--template` - template config YAML to copy.
- `--copy-inputs` - copy file-based inputs into `workspace/inputs` and rewrite paths.

---

#### `dense run`
Run the full generation pipeline.

Options:
- `--no-plot` - skip auto-plotting even if `plots` is configured in YAML.
- `--log-file PATH` - override the log file path. Otherwise DenseGen writes
  to `logging.log_dir/<run_id>.log` inside the workspace. The override path
  must still resolve inside `densegen.run.root`.

Notes:
- If you enable `scoring_backend: fimo`, run via `pixi run dense ...` (or ensure `fimo` is on PATH).

---

#### `dense plot`
Generate plots from existing outputs.

Options:
- `--only NAME1,NAME2` - run a subset of plots by name.

---

#### `dense ls-plots`
List available plot names and descriptions.

---

#### `dense report`
Generate an audit-grade report summary for a run. Outputs are run-scoped under `outputs/` by default.

Options:
- `--run` - run directory (defaults to config run root).
- `--out` - output directory relative to run root (default: `outputs`).
- `--format` - `json`, `md`, `html`, or `all` (comma-separated allowed).

Report outputs:
- `report.json`, `report.md`, `report.html`
- `report_assets/` (plots referenced by the HTML report)

---

### Examples

```bash
RUN_ROOT=/tmp/densegen-demo-$(date +%Y%m%d-%H%M)
uv run dense workspace init --id demo_press --root "$RUN_ROOT" \
  --template src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml \
  --copy-inputs
CFG="$RUN_ROOT/demo_press/config.yaml"

pixi run dense validate-config -c "$CFG"
pixi run dense inspect inputs -c "$CFG"
pixi run dense inspect plan   -c "$CFG"
pixi run dense inspect config -c "$CFG"
pixi run dense run            -c "$CFG"
pixi run dense plot           -c "$CFG" --only tf_usage,tf_coverage,tfbs_positional_histogram,diversity_health
pixi run dense inspect run     --run "$RUN_ROOT/demo_press"
pixi run dense inspect run     --root "$RUN_ROOT"
pixi run dense report          -c "$CFG" --format all
```

Demo run (small, Parquet-only config):

```bash
pixi run dense run -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml --no-plot
```

FIMO-backed sampling (pixi):

```bash
pixi run dense run -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml --no-plot
```

---

@e-south
