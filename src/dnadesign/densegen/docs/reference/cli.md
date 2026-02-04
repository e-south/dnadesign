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
- [`dense run`](#dense-run) - run Stage‑B sampling + optimization.
- [`dense campaign-reset`](#dense-campaign-reset) - remove outputs for a clean rerun (hidden command).
- [`dense plot`](#dense-plot) - render plots from outputs.
- [`dense ls-plots`](#dense-ls-plots) - list available plots.
- [`dense report`](#dense-report) - write audit-grade report summary.

---

### Config resolution

- `-c, --config PATH` — config YAML path (global or per‑command).
- If `-c/--config` is omitted, DenseGen resolves config in this order:
  1) `DENSEGEN_CONFIG_PATH` (if set)
  2) `./config.yaml` in the current directory
  3) nearest parent directory containing `config.yaml`
  4) a single auto‑detected workspace (if exactly one match is found)
- When auto‑detecting, the CLI prints the chosen config path and continues; if multiple workspaces are found,
  it exits non‑zero and lists the candidates so you can pass `-c`.
- If no config is found, the CLI exits non‑zero with an actionable error message.
- Input paths resolve against the config file directory.
- Outputs/tables/logs/plots/report must resolve inside `outputs/` under `densegen.run.root`.
- Config files must include `densegen.schema_version` (currently `2.9`) and `densegen.run`.

---

### `dense validate-config`
Validate the config YAML (schema + sanity checks). Fails fast on unknown keys or invalid values.

Options:
- `--probe-solver` — also probe the solver backend (fails fast if unavailable).

---

#### `dense inspect inputs`
Print resolved inputs plus Stage‑A pool status.

Options:
- `--verbose` — show full source file lists.
- `--absolute` — show absolute paths instead of workspace‑relative.
- `--show-motif-ids` — show full motif IDs instead of short TF labels.

---

#### `dense inspect plan`
Print the resolved quota plan per constraint bucket.

---

#### `dense inspect config`
Summarize outputs, Stage‑A sampling policy, Stage‑B sampling policy, and solver settings. Input sources are
listed in `dense inspect inputs`.

Options:
- `--show-constraints` — print full fixed elements per plan item.
- `--probe-solver` — verify the solver backend before reporting.
- `--absolute` — show absolute paths instead of workspace‑relative.

---

#### `dense inspect run`
Summarize a run manifest (`outputs/meta/run_manifest.json`) or list workspaces.

Options:
- `--run` — workspace directory (defaults to `densegen.run.root` from config).
- `--root` — list workspaces under a root directory.
- `--limit` — limit workspaces displayed when using `--root`.
- `--all` — include directories without `config.yaml` when using `--root`.
- `--config` — config path (used to resolve run root when `--run` is not set).
- `--absolute` — show absolute paths instead of workspace‑relative.
- `--verbose` — show failure breakdown columns (constraint filters + duplicate solutions).
- `--library` — include Stage‑B offered‑vs‑used summaries aggregated across all libraries.
- `--show-tfbs` — include TFBS sequences in library summaries.
- `--show-motif-ids` — show full motif IDs instead of short TF labels.
- `--events` — show event summary (stalls/resamples, library rebuilds).

---

#### `dense stage-a build-pool`
Build Stage‑A TFBS pools from inputs and write a pool manifest.

Options:
- `--out` — output directory relative to run root (default: `outputs/pools`; must be inside `outputs/`).
- `--input/-i` — input name(s) to build (defaults to all).
- `--fresh` — replace existing pool files (default is append + dedupe).
- `--show-motif-ids` — show full motif IDs instead of short TF labels.

Outputs:
- `pool_manifest.json`
- `<input>__pool.parquet` per input
- `outputs/pools/candidates/candidates.parquet` + `candidates_summary.parquet` (when candidate logging is enabled)

---

#### `dense stage-b build-libraries`
Build Stage‑B libraries (one per plan) from plan‑scoped pools derived from
`generation.plan[].sampling.include_inputs`.

Options:
- `--out` — output directory relative to run root (default: `outputs/libraries`; must be inside `outputs/`).
- `--pool` — pool directory from `stage-a build-pool` (defaults to `outputs/pools` in the workspace;
  must be inside `outputs/`).
- `--input/-i` — input name(s) to filter plan pools (defaults to all).
- `--plan/-p` — plan item name(s) to build (defaults to all).
- `--overwrite` — overwrite existing library artifacts (destructive).
- `--append` — append new libraries to existing artifacts (cumulative). Requires that the existing
  library manifest matches the current config hash and Stage‑A pool fingerprint; otherwise fails fast.

Behavior:
- If library artifacts exist and neither `--overwrite` nor `--append` is provided, the command exits
  non‑zero with an actionable error.
- `--input` filters plans by membership: a plan is built only if its `include_inputs` contains all
  requested input names.

Output summary:
- CLI output aggregates libraries per plan pool/plan and reports min/median/max for sites, TF counts, and bp totals.
- Per-library details are written to the Parquet artifacts under `outputs/libraries/`.

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
- `--template-id` — packaged template id (e.g., `demo_meme_three_tfs`).
- `--template` — template config YAML to copy (path).
- `--copy-inputs` — copy file-based inputs into `workspace/inputs` and rewrite paths.

---

#### `dense run`
Run the pipeline (Stage‑B sampling → optimization → outputs) using existing Stage‑A pools.

Options:
- `--no-plot` — skip auto‑plotting even if `plots` is configured in YAML.
- `--fresh` — delete the workspace `outputs/` directory before running.
- `--resume` — resume from existing outputs in the workspace.
- `--rebuild-stage-a` — rebuild Stage‑A pools before running (required if pools are missing or stale).
- `--log-file PATH` — override the log file path. Otherwise DenseGen writes to
  `logging.log_dir/<run_id>.log` inside the workspace. The override path must still resolve
  inside `outputs/` under `densegen.run.root`.
- `--show-tfbs` — include TFBS sequences in progress output.
- `--show-solutions` — include full solution sequences in progress output.

Notes:
- `dense run` requires Stage‑A pools under `outputs/pools` by default. If they are missing or stale,
  run `dense stage-a build-pool --fresh` or pass `--rebuild-stage-a`.
- Stage‑A sampling uses FIMO; ensure `fimo` is on PATH (e.g., via `pixi run`).
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
- `--absolute` — show absolute paths instead of workspace‑relative.

---

#### `dense ls-plots`
List available plot names and descriptions.

---

#### `dense report`
Generate an audit-grade report summary for a run. Outputs are run‑scoped under `outputs/report/` by default.

Options:
- `--run` — run directory (defaults to config run root).
- `--out` — output directory relative to run root (default: `outputs/report`; must be inside `outputs/`).
- `--absolute` — show absolute paths instead of workspace‑relative.
- `--format` — `json`, `md`, `html`, or `all` (comma‑separated allowed).
- `--strict/--fail-on-missing` — fail if core report inputs are missing.
- `--plots` — `none` or `include` (default: `none`). When `include`, report links plots from
  `outputs/plots/plot_manifest.json` (run `dense plot` first).

Report outputs:
- `report.json`, `report.md`, `report.html`

---

@e-south
