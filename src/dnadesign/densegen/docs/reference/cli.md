## DenseGen CLI

DenseGen exposes a Typer CLI via `dense`. This page is an operator manual (commands + flags). For a progressive, end‑to‑end walkthrough, see the [demo](../demo/demo_basic.md).

### Contents
- [Config resolution](#config-resolution) - where `dense` looks for config.yaml.
- [`dense validate-config`](#dense-validate-config) - schema and sanity checks.
- [`dense inspect inputs`](#dense-inspect-inputs) - Stage‑A input + sampling summary.
- [`dense inspect plan`](#dense-inspect-plan) - resolved quota/fraction plan.
- [`dense inspect config`](#dense-inspect-config) - resolved inputs, outputs, Stage‑A/Stage‑B settings.
- [`dense inspect run`](#dense-inspect-run) - summarize run manifests or list workspaces.
- [`dense stage-a build-pool`](#dense-stage-a-build-pool) - build Stage‑A TFBS pools.
- [`dense stage-b build-libraries`](#dense-stage-b-build-libraries) - build Stage‑B solver libraries.
- [`dense workspace init`](#dense-workspace-init) - scaffold a workspace.
- [`dense run`](#dense-run) - run Stage‑A + Stage‑B + optimization.
- [`dense campaign-reset`](#dense-campaign-reset) - remove outputs for a clean rerun (hidden command).
- [`dense plot`](#dense-plot) - render plots from outputs.
- [`dense ls-plots`](#dense-ls-plots) - list available plots.
- [`dense report`](#dense-report) - write audit-grade report summary.

---

### Config resolution

- `-c, --config PATH` — config YAML path (global or per‑command).
- If `-c/--config` is omitted, DenseGen uses `./config.yaml` in the **current directory** only.
- If `./config.yaml` is missing, the CLI exits non‑zero with:
  “No config found. cd into a workspace containing config.yaml, or pass -c path/to/config.yaml.”
- Input paths resolve against the config file directory.
- Outputs/tables/logs/plots/report must resolve inside `outputs/` under `densegen.run.root`.
- Config files must include `densegen.schema_version` (currently `2.5`) and `densegen.run`.

---

### `dense validate-config`
Validate the config YAML (schema + sanity checks). Fails fast on unknown keys or invalid values.

Options:
- `--probe-solver` — also probe the solver backend (fails fast if unavailable).

---

#### `dense inspect inputs`
Print resolved inputs plus a Stage‑A PWM sampling summary.

---

#### `dense inspect plan`
Print the resolved quota plan per constraint bucket.

---

#### `dense inspect config`
Summarize resolved inputs, outputs, Stage‑A sampling, Stage‑B sampling, and solver settings.

Options:
- `--show-constraints` — print full fixed elements per plan item.
- `--probe-solver` — verify the solver backend before reporting.

---

#### `dense inspect run`
Summarize a run manifest (`outputs/meta/run_manifest.json`) or list workspaces.

Options:
- `--run` — workspace directory (defaults to `densegen.run.root` from config).
- `--root` — list workspaces under a root directory.
- `--limit` — limit workspaces displayed when using `--root`.
- `--all` — include directories without `config.yaml` when using `--root`.
- `--config` — config path (used to resolve run root when `--run` is not set).
- `--verbose` — show failure breakdown columns (constraint filters + duplicate solutions).
- `--library` — include Stage‑B offered‑vs‑used summaries (TF/TFBS usage).
- `--library-limit` — limit library builds shown in per‑library summaries (`0` = all).
- `--top` — number of rows to show in library summaries.
- `--by-library/--no-by-library` — group library summaries per build attempt.
- `--top-per-tf` — limit TFBS rows per TF when summarizing.
- `--show-library-hash/--short-library-hash` — toggle full vs short library hashes.
- `--events` — show event summary (stalls/resamples, library rebuilds).

---

#### `dense stage-a build-pool`
Build Stage‑A TFBS pools from inputs and write a pool manifest.

Options:
- `--out` — output directory relative to run root (default: `outputs/pools`; must be inside `outputs/`).
- `--input/-i` — input name(s) to build (defaults to all).
- `--overwrite` — overwrite existing pool files.

Outputs:
- `pool_manifest.json`
- `<input>__pool.parquet` per input
- `outputs/pools/candidates/candidates.parquet` + `candidates_summary.parquet` (when candidate logging is enabled)

---

#### `dense stage-b build-libraries`
Build Stage‑B libraries (one per input + plan) from Stage‑A pools.

Options:
- `--out` — output directory relative to run root (default: `outputs/libraries`; must be inside `outputs/`).
- `--pool` — pool directory from `stage-a build-pool` (defaults to `outputs/pools` in the workspace;
  must be inside `outputs/`).
- `--input/-i` — input name(s) to build (defaults to all).
- `--plan/-p` — plan item name(s) to build (defaults to all).
- `--overwrite` — overwrite existing library artifacts.

Outputs:
- `library_builds.parquet`
- `library_members.parquet`
- `library_manifest.json`

---

#### `dense workspace init`
Stage a new workspace with `config.yaml`, `inputs/`, `outputs/`, plus `outputs/logs/`, `outputs/meta/`,
`outputs/tables/`, `outputs/plots/`, and `outputs/report/`.

Options:
- `--id` — run identifier (directory name).
- `--root` — workspace root directory (default: current directory).
- `--template-id` — packaged template id (e.g., `demo_meme_two_tf`).
- `--template` — template config YAML to copy (path).
- `--copy-inputs` — copy file-based inputs into `workspace/inputs` and rewrite paths.

---

#### `dense run`
Run the full pipeline (Stage‑A sampling → Stage‑B sampling → optimization → outputs).

Options:
- `--no-plot` — skip auto‑plotting even if `plots` is configured in YAML.
- `--fresh` — delete the workspace `outputs/` directory before running.
- `--resume` — resume from existing outputs in the workspace.
- `--log-file PATH` — override the log file path. Otherwise DenseGen writes to
  `logging.log_dir/<run_id>.log` inside the workspace. The override path must still resolve
  inside `outputs/` under `densegen.run.root`.

Notes:
- If Stage‑A sampling uses `scoring_backend: fimo`, ensure `fimo` is on PATH (e.g., via `pixi run`).
- If the workspace already has run outputs (e.g., `outputs/tables/*.parquet` or
  `outputs/meta/run_state.json`), you must choose `--resume` or `--fresh`.
  Stage‑A/Stage‑B artifacts in `outputs/pools` or `outputs/libraries` do not trigger this guard.

---

#### `dense campaign-reset`
Remove the entire `outputs/` directory under the configured run root. This is a hidden command
intended for demo and pressure‑testing workflows; it is not listed in `dense --help`.

Options:
- `--config` — config path (used to resolve run root).

Notes:
- Inputs and configs are preserved; only run outputs/state are deleted.

---

#### `dense plot`
Generate plots from existing outputs.

Options:
- `--only NAME1,NAME2` — run a subset of plots by name.

---

#### `dense ls-plots`
List available plot names and descriptions.

---

#### `dense report`
Generate an audit-grade report summary for a run. Outputs are run‑scoped under `outputs/report/` by default.

Options:
- `--run` — run directory (defaults to config run root).
- `--out` — output directory relative to run root (default: `outputs/report`; must be inside `outputs/`).
- `--format` — `json`, `md`, `html`, or `all` (comma‑separated allowed).

Report outputs:
- `report.json`, `report.md`, `report.html`
- `assets/` (plots referenced by the HTML report)
- `assets/composition.csv` (full composition table when available)

---

@e-south
