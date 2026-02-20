## OPAL Command Line Interface

The OPAL CLI is a thin layer over OPAL’s application modules. It lets you initialize a campaign, ingest labeled samples, train/score/select for a round, inspect records and models, validate your dataset, and generate plots.

Commands are registry-driven and plugin‑agnostic: they operate on the configured plugin names and enforce only declared contracts.

### Command overview

Each command should do one thing. Usage blocks show required args; optional flags are in brackets.

Round semantics:

- `ingest-y --observed-round` stamps label events with the observed round.
- `run/explain --labels-as-of` chooses the training cutoff (labels with `observed_round <= as_of_round`).

Guided hints:

- Human output for `init`, `validate`, `ingest-y`, `run`, `explain`, and `verify-outputs` prints next-step hints by default.
- Use `--no-hints` to disable hint lines.

### `guide`

Generate a guided runbook from the current campaign config.

**Usage**

```bash
opal guide --config <yaml> [--labels-as-of <r>] [--format text|markdown|json] [--out <path>]
opal guide next --config <yaml> [--observed-round <r>] [--labels-as-of <r>] [--json]
```

**Notes**

* `guide` is read-only. It summarizes plugin wiring, lifecycle steps, round semantics, and deep-dive docs/source pointers.
* `guide next` inspects campaign state + label history and prints the recommended next command sequence.
* Prefer `guide next` in agent/automation loops for state-aware progression.

---

### `demo-matrix`

Run canonical demo workflows end-to-end in isolated temp copies.

**Usage**

```bash
opal demo-matrix [--tmp-root <dir>] [--rounds 0|0,1] [--json] [--keep] [--fail-fast]
```

**Notes**

* Runs `demo_rf_sfxi_topn`, `demo_gp_topn`, and `demo_gp_ei`.
* Executes reset/init/validate/ingest/run/verify checks per flow and round.
* Exits non-zero when any flow fails or verify mismatches occur.
* Intended for demo pressure testing and CI-style validation.

### `init`

Initialize/validate a campaign workspace and write `state.json`.

**Usage**

```bash
opal init --config <yaml> [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* Ensures the campaign `workdir` has `outputs/`.
* Ensures the label history column exists in `records.parquet`.
* Writes/updates `state.json` with campaign identity, data location, and settings.

---

### `ingest-y`

Transform a tidy CSV/Parquet/XLSX to model-ready **Y**, preview, confirm, and append to label history.

**Usage**

```bash
opal ingest-y --config <yaml> --observed-round <r> --in <path> \
  [--transform <name>] [--params <transform_params.json>] \
  [--unknown-sequences create|drop|error] [--infer-missing-required] \
  [--if-exists fail|skip|replace] [--apply] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round, -r, --observed-round`: Observed round stamp for these labels.
* `--csv, --in`: CSV/Parquet/XLSX input (`.csv`, `.parquet`, `.pq`, or `.xlsx`).
* `--transform`: Override YAML `transforms_y.name`.
* `--params`: JSON file (.json) with transform params (overrides YAML `transforms_y.params`).
* `--unknown-sequences`: How to handle sequences not found in records (default: `create`). Use `drop` to skip
  unknown sequences when required columns are missing or when you want a strict in‑place update.
* `--infer-missing-required`: Auto-fill missing required columns for new sequences (`bio_type`, `alphabet`)
  using the most common values found in `records.parquet`.
* `--if-exists`: Behavior if `(id, round)` already exists in label history (`fail`/`skip`/`replace`).
* `--apply`: Apply ingest without interactive confirmation.
* `--json`: Output as machine-readable JSON (default output is plain text).

**Behavior & checks**

* Uses `transforms_y` from YAML unless overridden by `--transform/--params`.
* **Strict preflights**: schema checks, completeness.
* **Preview is printed** (counts + sample) before any write.
* Duplicate handling is controlled by `ingest.duplicate_policy` (error | keep_first | keep_last).
* **New IDs** allowed if your CSV includes essentials: `sequence`, `bio_type`, `alphabet`, and the configured X column.
* If new sequences are missing required columns, OPAL will prompt to infer defaults for `bio_type`/`alphabet`
  (or use `--infer-missing-required` for non-interactive runs). For other missing columns, use
  `--unknown-sequences drop` or provide the columns.
* If `records.parquet` contains duplicate sequences, `ingest-y` requires an explicit `id` column for all rows
  to avoid ambiguous sequence → id mapping.
* When unknown sequences are missing **X** data, `ingest-y` drops those rows automatically (unless you pass
  `--unknown-sequences error`). This avoids creating partial records without X.
* If adding **new sequences** and X is list-valued, prefer **Parquet** input so the X column remains list-typed
  (CSV will coerce lists to strings).
* Appends to `opal__<slug>__label_hist` and writes the current Y column.
* Emits `label` events into `outputs/ledger/labels.parquet`.

---

### `run`

Train on labels with **`observed_round ≤ R`** (where `R` comes from `--labels-as-of`), score the candidate universe,
apply the objective, select top-k, write artifacts, and append run-aware label history entries.

**Usage**

```bash
opal run --config <yaml> --labels-as-of <r> \
  [--k <n>] [--resume] [--score-batch-size <n>] [--verbose|--quiet] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round, -r, --labels-as-of`: Training cutoff (use labels with `observed_round ≤ r`).
* `--k, -k`: Override `selection.params.top_k`.
* `--score-batch-size`: Override `scoring.score_batch_size` for this run.
* `--resume`: Allow overwriting existing per-round artifacts (required if `outputs/rounds/round_<r>/` already contains artifacts). When set, the round directory is wiped before writing new artifacts.
* `--verbose/--quiet`: Control log verbosity (default: verbose).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Pipeline**

* Pulls effective labels per `training.policy` (cumulative vs current round, dedup policy).
* Predicts in batches (`scoring.score_batch_size` or `--score-batch-size`).
* Evaluates configured objective plugins and emits named score/uncertainty channels.
* Resolves `selection.params.score_ref` (and optional `uncertainty_ref`) to choose channels for selection.
* Selects with the configured strategy + tie handling.
  * If `selection.params.exclude_already_labeled: true` (default), designs already labeled are **excluded**;
    scope is controlled by `training.policy.allow_resuggesting_candidates_until_labeled`.

**Artifacts written** (`outputs/rounds/round_<r>/`)

* `model/`
  * `model.joblib`
  * `model_meta.json`
  * `feature_importance.csv` (optional)
* `selection/`
  * `selection_top_k.csv`
  * `selection_top_k__run_<run_id>.csv` (immutable per-run copy)
* `labels/`
  * `labels_used.parquet` (training snapshot for this run)
* `metadata/`
  * `round_ctx.json`
  * `objective_meta.json`
* `logs/`
  * `round.log.jsonl` — compact JSONL with stage events and prediction batch progress

**Events appended** to **ledger sinks** under `outputs/`

* `run_pred` → `outputs/ledger/predictions/` (one row per candidate with **`pred__y_hat_model`** and
  **`pred__score_selected`**, selection rank/flag, and diagnostics).
* `run_meta` → `outputs/ledger/runs.parquet` (one row per run with model/config/selection snapshot
  and artifact checksums).

`pred__y_hat_model` is **objective-space** (after any Y‑ops inversion), so downstream logic is plugin‑agnostic.

**Reruns & non-interactive mode**

If you rerun a round that already exists in `state.json`, OPAL will prompt before overwriting. In non‑TTY
contexts (e.g., CI), the prompt cannot be shown and the command will exit with a message instructing you to
re‑run with `--resume`.

**Write-backs to `records.parquet` (primary label history)**

* `opal__<slug>__label_hist` — append-only per-record history of observed labels and run-aware predictions.

---

### `predict`

Run **ephemeral** predictions from a frozen model. No writes to `records.parquet`.

**Usage**

```bash
opal predict --config <yaml> \
  [--model-path <path> | --round <r>] \
  [--model-name <registry_name> --model-params <params.json>] \
  [--in <csv|parquet>] [--out <csv|parquet>] \
  [--id-col <name>] [--sequence-col <name>] \
  [--generate-id-from-sequence] [--assume-no-yops]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--model-path`: Path to `model.joblib` (overrides `--round`, e.g. `outputs/rounds/round_<r>/model/model.joblib`).
* `--round, -r`: Round index to resolve model from `state.json` (default: latest). Accepts `latest`.
* `--model-name` / `--model-params`: Required if `model_meta.json` is missing. `--model-params` must be a `.json`.
* `--in`: Optional input CSV/Parquet (`.csv`, `.parquet`, `.pq`; defaults to `records.parquet`).
* `--out`: Optional output CSV/Parquet (`.csv`, `.parquet`, `.pq`; defaults to stdout CSV).
* `--id-col`, `--sequence-col`: Column names in the input table.
* `--generate-id-from-sequence`: Deterministically generate ids if id column is missing.
* `--assume-no-yops`: Skip Y‑ops inversion even if training used Y‑ops.

**Notes**

* `--model-path` and `--round` are mutually exclusive; passing both is an error.
* Defaults to `records.parquet` when `--in` is omitted.
* Writes CSV to stdout by default; use `--out` for CSV/Parquet files (Parquet keeps vectors as list<float>).

---

### `record-show`

Per-record history report (ground truth + per-round predictions/rank/selected).

**Usage**

```bash
opal record-show --config <yaml> \
  [<ID-or-SEQ> | --id <ID> | --sequence <SEQ> | --selected-rank <n>] \
  [--round <k|latest>] \
  [--run-id <id>] [--with-sequence|--no-sequence] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `<ID-or-SEQ>`: Positional id or sequence (use `--id/--sequence` to disambiguate).
* `--id`, `--sequence`: Explicit lookup key (mutually exclusive).
* `--selected-rank`: Resolve id from `selection_top_k.csv` by competition rank (1-based).
* `--round`: Round selector used with `--selected-rank` (default: `latest`).
* `--run-id`: Explicit run_id for ledger predictions (or `latest` to pick the latest ledger run by `(as_of_round, run_id)`).
* `--with-sequence/--no-sequence`: Include the sequence in output (default: on).
* `--json`: Output as JSON.

**Notes**

* `--selected-rank` cannot be combined with `<ID-or-SEQ>`, `--id`, or `--sequence`.
* If reruns exist for a round, pass `--run-id` to avoid mixing predictions.
* If the requested record does not exist, the command exits non-zero.

### `model-show`

Inspect a saved model; optionally dump full feature importances.

**Usage**

```bash
opal model-show \
  [--model-path <path> | --config <yaml> --round <k|latest>] \
  [--model-name <registry_name> --model-params <params.json>] \
  [--out-dir <dir>] [--json]
```

**Flags**

* `--model-path`: Path to `model.joblib` (overrides `--config/--round`, e.g. `outputs/rounds/round_<r>/model/model.joblib`).
* `--config, -c`: Path to `configs/campaign.yaml` (required if resolving from `state.json`).
* `--round, -r`: Round selector (integer or `latest`) to resolve model.
* `--model-name` / `--model-params`: Required if `model_meta.json` is missing. `--model-params` must be a `.json`.
* `--out-dir`: Write `feature_importance_full.csv` and print top-20 in JSON.
* `--json`: Output as machine-readable JSON (default output is plain text).

### `objective-meta`

List objective metadata and diagnostic keys for a round.

**Usage**

```bash
opal objective-meta --config <yaml-or-dir> [--round <k|latest> | --run-id <id>] [--profile|--no-profile]
  [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set). Directories are supported only for `opal plot`, `opal notebook`, and `opal objective-meta`.
* `--round, -r`: Round selector (integer or `latest`; default: latest).
* `--run-id`: Explicit run_id to disambiguate when a round has multiple runs.
* `--profile/--no-profile`: Profile candidate hue/size fields from the selected run.
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* If multiple run_ids exist for the selected round, `--run-id` is required.

---

### `verify-outputs`

Compare selection artifacts against ledger predictions for a single run (run-aware, audit-grade).

**Usage**

```bash
opal verify-outputs --config <yaml> [--round <k|latest> | --run-id <id>] \
  [--selection-path <path>] [--eps <float>] [--json]
```

**Notes**

* Resolves the selection artifact path from `outputs/ledger/runs.parquet` when possible.
* Uses the ledger’s `pred__score_selected` as the primary score source.
* `--selection-path` accepts `.csv` or `.parquet`.
* `--round, -r`: Round selector (integer or `latest`).
* If the selected round has multiple runs, pass `--run-id` to disambiguate.
* Reads from `outputs/ledger/runs.parquet` and `outputs/ledger/predictions/`.
* `--json` writes JSON to stdout; redirect to a file when you need a saved report.

---

### `ctx`

Inspect `round_ctx.json` carriers.

**Usage**

```bash
opal ctx show  --config <yaml> [--round <k|latest>] [--keys <prefix> ...] [--json]
opal ctx audit --config <yaml> [--round <k|latest>] [--json]
opal ctx diff  --config <yaml> --round-a <k|latest> --round-b <k|latest> [--keys <prefix> ...] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round, -r`: Round selector for `show`/`audit`.
* `--round-a`, `--round-b`: Round selectors for `diff`.
* `--keys`: Filter by key prefix (repeatable; applies to `show`/`diff`).

**How to read ctx output**

* `ctx show`: raw key/value snapshot for one round.
* `ctx audit`: per-plugin contract audit (`consumed` / `produced` keys).
* `ctx diff`: key-level change summary between two rounds (`added`, `removed`, `changed`).
* Stage-scoped keys (for example model `predict` summaries) appear as final committed values after stage-end checks.

**Common checks**

```bash
# Model contract keys captured in the latest round
opal ctx show -c <yaml> --round latest --keys core/contracts/model

# Full per-plugin consumed/produced audit
opal ctx audit -c <yaml> --round latest

# What changed in objective/runtime keys between two rounds
opal ctx diff -c <yaml> --round-a 0 --round-b 1 --keys objective/
```

---

### `explain`

Dry-run planner for a round: counts, plan, warnings. **No writes.**

**Usage**

```bash
opal explain --config <yaml> --labels-as-of <k>
  [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round, -r, --labels-as-of`: Training cutoff (alias of `--round`).
* `--json`: Output as machine-readable JSON (default output is plain text).

Prints: number of training labels, candidate universe size, transforms/models/selection used,
vector dimension, and any preflight warnings.

---

### `status`

Dashboard from `state.json`.

**Usage**

```bash
opal status --config <yaml> [--round <k> | --all] [--with-ledger] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round`: Specific round details.
* `--all`: Dump every round (JSON output, even without `--json`).
* `--with-ledger`: Include ledger run_meta summaries in output.
* `--json`: Output as JSON.

### `runs`

List or inspect `run_meta` entries from `outputs/ledger/runs.parquet`.

**Usage**

```bash
opal runs list --config <yaml> [--round <k|latest>] [--json]
opal runs show --config <yaml> [--round <k|latest> | --run-id <rid>] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round, -r`: Round selector (integer or `latest`).
* `--run-id`: Explicit run_id to display (show only).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* `runs show --round <k>` requires `--run-id` if round `<k>` has multiple runs.

---

### `log`

Summarize `round.log.jsonl` for a round.

**Usage**

```bash
opal log --config <yaml> [--round <k|latest>] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--round, -r`: Round selector (integer or `latest`).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* If a round has been re-run (multiple `start` events in the same log), the summary focuses on the **latest run**.

### `validate`

End-to-end table checks (essentials present; X present).

**Usage**

```bash
opal validate --config <yaml>
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).

**Notes**

* Config parsing is strict: malformed YAML or duplicate keys fail as bad-args config errors.
* Verifies **USR essentials** exist in `records.parquet`.
* Verifies the configured **X** column exists.
* If Y is present, validates vector length & numeric/finite cells.

---

### `label-hist`

Validate, repair, or explicitly attach-from-y into the label history column (no silent fixes).

**Usage**

```bash
opal label-hist <validate|repair|attach-from-y> --config <yaml> [--apply] [--round <int>] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `<validate|repair|attach-from-y>`: Action (alias: `check` = `validate`).
* `--apply`: Apply changes for `repair` or `attach-from-y` (default: dry-run).
* `--round, -r`: Required for `attach-from-y`; round stamp to attach.
* `--src`: Optional label_hist source tag for `attach-from-y` (default: `manual_attach`).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* `attach-from-y` is a **manual** fix for datasets with a populated Y column but empty label history.
  It only attaches entries for rows where `label_hist` is empty and Y is finite.

### Records label history

OPAL manages a primary per‑record label history column in `records.parquet`:

* `opal__<slug>__label_hist` — append‑only history of observed labels and run‑aware predictions.

`opal init` will add the label history column if it is missing.

---

### `plot`

Generate plots declared in the campaign’s `plots:` block. Plots are plugin-driven and campaign-scoped.

**Usage**

```bash
opal plot --config <yaml-or-dir> [--plot-config <plots.yaml>] \
  [--round <selector>] [--run-id <id>] [--name <plot-name>] [--tag <tag> ...]
opal plot --list
opal plot --list-config --config <yaml-or-dir>
opal plot --describe <plot-kind>
```

**Flags**

* `--config, -c`: Campaign YAML or campaign directory.
* `--plot-config`: Path to a plots YAML (overrides `plot_config` in configs/campaign.yaml).
* `--list`: List registered plot kinds and exit (does not require config).
* `--list-config`: List plots configured in YAML and exit (requires `--config`).
* `--describe`: Show parameters + required fields for a plot kind.
* `--round, -r`: `latest | all | 3 | 1,3,7 | 2-5` (plugin may define defaults).
* `--run-id`: Explicit run_id to disambiguate ledger predictions (required if multiple runs exist for a round).
* `--name, -n`: Run a single plot by name; omit to run **all**.
* `--tag`: Run plots with the given tag (repeatable).

**Notes**

* Overwrites files by default; continues on error; exit code **1** if any plot failed.
* Output directory defaults to `outputs/plots`, or honors `output.dir` if provided.
* Plot-specific knobs **must** live under `params:`; top-level plotting keys are errors.
* Prefer `plot_config: plots.yaml` in configs/campaign.yaml to keep runtime config lean.
* `plot_defaults` and `plot_presets` reduce redundancy; `preset: <name>` merges into each plot entry.
* Set `enabled: false` on any plot entry to keep it in the YAML without running it.
* If a round has multiple run_ids, plots require `--run-id` to avoid mixing reruns.
* If `--run-id` is provided, OPAL resolves its round from `outputs/ledger/runs.parquet`; `--round all` is invalid and conflicting `--round` values error.

**Campaign YAML (example)**

```yaml
plot_config: plots.yaml
```

**plots.yaml (example)**

```yaml
plot_defaults:
  output:
    format: "png"                       # png/svg/pdf
    dpi: 600

plots:
  - name: score_vs_rank_latest
    kind: scatter_score_vs_rank         # plot plugin id
    params:
      score_field: "pred__score_selected" # field from run_pred rows
      hue: null                         # or "round"
      highlight_selected: false
    output:
      dir: "{campaign}/plots/{kind}/{name}"  # {campaign|workdir|kind|name|round_suffix}
      filename: "{name}{round_suffix}.png"
```

**Data sources**
Plot plugins typically read from the campaign’s **ledger sinks** under `outputs/` and/or **`records.parquet`**.
You may add extra sources per plot entry via:

```yaml
data:
  - name: extra_csv
    path: ./extras/scores.csv
```

Built-ins injected for plots:

* `records`
* `outputs`
* `ledger_predictions_dir`
* `ledger_runs_parquet`
* `ledger_labels_parquet`

---

### `notebook`

Generate or run a campaign-tied marimo notebook for interactive analysis.

**Usage**

```bash
uv run opal notebook
uv run opal notebook generate --config <yaml-or-dir> [--round <latest|k>] [--out <path>] [--name <file>] [--force] [--validate/--no-validate]
uv run opal notebook run --config <yaml-or-dir> [--path <notebook.py>]
```

**Notes**

* `generate` writes a marimo notebook that loads ledger artifacts (runs/predictions/labels).
* `generate` requires the campaign `records.parquet` to exist because the notebook loads records on startup.
* By default, `generate` validates ledger artifacts exist. Use `--no-validate` to scaffold a notebook before any runs.
* When `--validate` is on, `--round` must exist in ledger runs (otherwise use `--no-validate`).
* `run` launches `marimo edit` if marimo is installed; otherwise it prints install guidance.
* `run` resolves the notebook under `<workdir>/notebooks`. If multiple exist, it prompts in TTY or requires `--path` in non-interactive mode.
* Running `uv run opal notebook` (no subcommand) lists available notebooks and nudges the next step.

---

### `campaign-reset` (advanced)

Reset a campaign to a clean slate: prune OPAL-derived columns from `records.parquet`, remove `outputs/`, remove `notebooks/`, and clear `state.json`.

**Usage**

```bash
opal campaign-reset --config <yaml> [--apply] [--backup|--no-backup] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--apply`: Apply reset without interactive confirmation.
* `--backup/--no-backup`: Backup `records.parquet` before pruning (default: no-backup).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* This command is hidden from top-level `opal --help` because it is destructive.
* Uses slug confirmation in interactive mode (or `--apply` in non-interactive mode).

---

### `ledger-compact`

Compact ledger datasets after repeated append cycles.

**Usage**

```bash
opal ledger-compact --config <yaml> --runs [--apply] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--runs`: Compact `outputs/ledger/runs.parquet` (required; command exits if omitted).
* `--apply`: Apply compaction without interactive confirmation.
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* Rewrites ledger datasets in place and should be run when no other OPAL process is writing.

---

### `prune-source`

Remove OPAL-derived columns (`opal__*`) and the configured Y column from `records.parquet`.

**Usage**

```bash
opal prune-source --config <yaml> [--scope any|campaign] [--keep <col> ...] \
  [--apply] [--backup|--no-backup] [--json]
```

**Flags**

* `--config, -c`: Path to `configs/campaign.yaml` (required unless `$OPAL_CONFIG` is set).
* `--scope`: Which opal namespaces to prune: `any` (default) or `campaign` (this campaign’s slug only).
* `--keep, -k`: Column name(s) to keep even if matched for deletion (repeatable).
* `--apply`: Apply prune without interactive confirmation.
* `--backup/--no-backup`: Backup original file before pruning (default: on).
* `--json`: Output as machine-readable JSON (default output is plain text).

**Notes**

* Designed as a **start fresh** option before re-running round 0.

---

### Config resolution

Campaign-scoped commands require explicit config context:

1. **Explicit flag** (`--config`)
2. **Environment variable** (`OPAL_CONFIG`)

Passing a **directory** to `--config` is supported only for `opal plot`, `opal notebook`, and `opal objective-meta`.
Other campaign-scoped commands require a YAML config path.
`plot_config` paths are resolved relative to the `configs/campaign.yaml` that declares them.
For scripts and CI, pass `--config` explicitly.
