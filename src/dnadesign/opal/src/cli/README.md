
## OPAL — Command Line Interface

The OPAL CLI is a thin layer over OPAL’s application modules. It lets you initialize a campaign, ingest labeled samples, train/score/select for a round, inspect records and models, validate your dataset, and generate plots.

Commands are registry-driven and plugin‑agnostic: they operate on the configured plugin names and enforce only declared contracts.

This document is **CLI-focused**:

- [Quick start](#quick-start)
- [Command overview](#command-overview)
- [Typical workflows](#typical-workflows)
- [Typing less](#typing-less)
- [Extending the CLI](#extending-the-cli)
- [CLI directory map](#cli-directory-map)

For architecture & concepts, see the **[Top-level README](../../README.md)**.

---

### Quick start

See all available commands and flags:

```bash
opal --help
```

---

### Command overview

Each command should do one thing. Usage blocks show required args; optional flags are in brackets.

#### `init`

Initialize/validate a campaign workspace and write `state.json`.

**Usage**

```bash
opal init --config <yaml>
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).

**Notes**

* Ensures the campaign `workdir` has `outputs/` and `inputs/`.
* Writes/updates `state.json` with campaign identity, data location, and settings.

#### `ingest-y`

Transform a tidy CSV/Parquet to model-ready **Y**, preview, confirm, and append to label history.

**Usage**

```bash
opal ingest-y --config <yaml> --round <r> --csv <path> \
  [--transform <name>] [--params <transform_params.json>] \
  [--if-exists fail|skip|replace] [--yes]
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).
* `--round, -r, --observed-round`: Observed round stamp for these labels.
* `--csv, --in`: CSV/Parquet with raw reads.
* `--transform`: Override YAML `transforms_y.name`.
* `--params`: JSON file with transform params (overrides YAML `transforms_y.params`).
* `--if-exists`: Behavior if `(id, round)` already exists in label history (`fail`/`skip`/`replace`).
* `--yes, -y`: Skip interactive prompt.

**Behavior & checks**

* Uses `transforms_y` from YAML unless overridden by `--transform/--params`.
* **Strict preflights**: schema checks, completeness.
* **Preview is printed** (counts + sample) before any write.
* Duplicate handling is controlled by `ingest.duplicate_policy` (error | keep_first | keep_last).
* **New IDs** allowed if your CSV includes essentials: `sequence`, `bio_type`, `alphabet`, and the configured X column.
* Appends to `opal__<slug>__label_hist` and writes the current Y column.
* Emits `label` events into `outputs/ledger.labels.parquet`.

#### `run`

Train on labels with **`observed_round ≤ R`** (where `R` comes from `--round`), score the candidate universe,
apply the objective, select top-k, write artifacts, append canonical events, and update caches.

**Usage**

```bash
opal run --config <yaml> --round <r> \
  [--k <n>] [--resume] [--score-batch-size <n>] [--verbose|--quiet]
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).
* `--round, -r, --labels-as-of`: Training cutoff (use labels with `observed_round ≤ r`).
* `--k, -k`: Override `selection.params.top_k`.
* `--score-batch-size`: Override `scoring.score_batch_size` for this run.
* `--resume`: Allow overwriting existing per-round artifacts.
* `--verbose/--quiet`: Control log verbosity (default: verbose).

**Pipeline**

* Pulls effective labels per `training.policy` (cumulative vs current round, dedup policy).
* Predicts in batches (`scoring.score_batch_size` or `--score-batch-size`).
* Applies your **objective** to produce a scalar **selection score**.
* Selects with the configured strategy + tie handling.
  * If `selection.params.exclude_already_labeled: true` (default), designs already labeled are **excluded**;
    scope is controlled by `training.policy.allow_resuggesting_candidates_until_labeled`.

**Artifacts written** (`outputs/round_<r>/`)

* `model.joblib`
* `model_meta.json`
* `selection_top_k.csv`
* `labels_used.parquet` (training snapshot for this run)
* `round_ctx.json`
* `objective_meta.json`
* `round.log.jsonl` — compact JSONL with stage events and prediction batch progress

**Events appended** to **ledger sinks** under `outputs/`

* `run_pred` → `outputs/ledger.predictions/` (one row per candidate with **`pred__y_hat_model`** and
  **`pred__y_obj_scalar`**, selection rank/flag, and diagnostics).
* `run_meta` → `outputs/ledger.runs.parquet` (one row per run with model/config/selection snapshot
  and artifact checksums).

`pred__y_hat_model` is **objective-space** (after any Y‑ops inversion), so downstream logic is plugin‑agnostic.

**Write-backs to `records.parquet` (caches only)**

* `opal__<slug>__latest_as_of_round`
* `opal__<slug>__latest_pred_scalar`

#### `predict`

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

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).
* `--model-path`: Path to `model.joblib` (overrides `--round`).
* `--round, -r`: Round index to resolve model from `state.json` (default: latest).
* `--model-name` / `--model-params`: Required if `model_meta.json` is missing.
* `--in`: Optional input CSV/Parquet (defaults to `records.parquet`).
* `--out`: Optional output CSV/Parquet (defaults to stdout CSV).
* `--id-col`, `--sequence-col`: Column names in the input table.
* `--generate-id-from-sequence`: Deterministically generate ids if id column is missing.
* `--assume-no-yops`: Skip Y‑ops inversion even if training used Y‑ops.

**Notes**

* Defaults to `records.parquet` when `--in` is omitted.
* Writes CSV to stdout by default; use `--out` for CSV/Parquet files (Parquet keeps vectors as list<float>).

#### `record-show`

Per-record history report (ground truth + per-round predictions/rank/selected).

**Usage**

```bash
opal record-show --config <yaml> \
  [<ID-or-SEQ> | --id <ID> | --sequence <SEQ>] \
  [--with-sequence|--no-sequence] [--json]
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).
* `<ID-or-SEQ>`: Positional id or sequence (use `--id/--sequence` to disambiguate).
* `--id`, `--sequence`: Explicit lookup key (mutually exclusive).
* `--with-sequence/--no-sequence`: Include the sequence in output (default: on).
* `--json`: Output as JSON.

#### `model-show`

Inspect a saved model; optionally dump full feature importances.

**Usage**

```bash
opal model-show \
  [--model-path <path> | --config <yaml> --round <r>] \
  [--model-name <registry_name> --model-params <params.json>] \
  [--out-dir <dir>]
```

**Flags**

* `--model-path`: Path to `model.joblib` (overrides `--config/--round`).
* `--config, -c`: Path to `campaign.yaml` (required if resolving from `state.json`).
* `--round, -r`: Round index to resolve model (default: latest).
* `--model-name` / `--model-params`: Required if `model_meta.json` is missing.
* `--out-dir`: Write `feature_importance_full.csv` and print top-20 in JSON.

#### `objective-meta`

List objective metadata and diagnostic keys for a round.

**Usage**

```bash
opal objective-meta --config <yaml> [--round <k|latest>] [--profile|--no-profile]
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (directories are only supported for `opal plot`).
* `--round, -r`: Round selector (integer or `latest`).
* `--profile/--no-profile`: Compute hue/size suitability stats (default: off).

**Notes**

* Reads from `outputs/ledger.runs.parquet` and `outputs/ledger.predictions/`.

#### `ctx`

Inspect `round_ctx.json` carriers.

**Usage**

```bash
opal ctx show  --config <yaml> [--round <k|latest>] [--keys <prefix> ...]
opal ctx audit --config <yaml> [--round <k|latest>]
opal ctx diff  --config <yaml> --round-a <k|latest> --round-b <k|latest> [--keys <prefix> ...]
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).
* `--round, -r`: Round selector for `show`/`audit`.
* `--round-a`, `--round-b`: Round selectors for `diff`.
* `--keys`: Filter by key prefix (repeatable; applies to `show`/`diff`).

#### `explain`

Dry-run planner for a round: counts, plan, warnings. **No writes.**

**Usage**

```bash
opal explain --config <yaml> --round <k>
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).
* `--round, -r, --labels-as-of`: Training cutoff (alias of `--round`).

Prints: number of training labels, candidate universe size, transforms/models/selection used,
vector dimension, and any preflight warnings.

#### `status`

Dashboard from `state.json`.

**Usage**

```bash
opal status --config <yaml> [--round <k> | --all] [--with-ledger] [--json]
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).
* `--round`: Specific round details.
* `--all`: Dump every round (JSON output, even without `--json`).
* `--with-ledger`: Include ledger run_meta summaries in output.
* `--json`: Output as JSON.

#### `runs`

List or inspect `run_meta` entries from `outputs/ledger.runs.parquet`.

**Usage**

```bash
opal runs list --config <yaml> [--round <k|latest>]
opal runs show --config <yaml> [--round <k|latest> | --run-id <rid>]
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).
* `--round, -r`: Round selector (integer or `latest`).
* `--run-id`: Explicit run_id to display (show only).

#### `log`

Summarize `round.log.jsonl` for a round.

**Usage**

```bash
opal log --config <yaml> [--round <k|latest>]
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).
* `--round, -r`: Round selector (integer or `latest`).

#### `validate`

End-to-end table checks (essentials present; X present).

**Usage**

```bash
opal validate --config <yaml>
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).

**Notes**

* Verifies **USR essentials** exist in `records.parquet`.
* Verifies the configured **X** column exists.
* If Y is present, validates vector length & numeric/finite cells.

#### `label-hist`

Validate or repair the label history column (explicit, no silent fixes).

**Usage**

```bash
opal label-hist <validate|repair> --config <yaml> [--apply]
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).
* `<validate|repair>`: Action (alias: `check` = `validate`).
* `--apply`: Apply changes for `repair` (default: dry-run).

#### Records cache columns

OPAL manages a few derived columns in `records.parquet`:

* `opal__<slug>__label_hist` — append-only label history (SSoT)
* `opal__<slug>__latest_as_of_round` — last scored round for each record
* `opal__<slug>__latest_pred_scalar` — latest objective scalar cache

#### `plot`

Generate plots declared in the campaign’s `plots:` block. Plots are plugin-driven and campaign-scoped.

**Usage**

```bash
opal plot --config <yaml-or-dir> [--plot-config <plots.yaml>] \
  [--round <selector>] [--name <plot-name>] [--tag <tag> ...]
```

**Flags**

* `--config, -c`: Campaign YAML or campaign directory (**only** `plot` supports directories).
* `--plot-config`: Path to a plots YAML (overrides `plot_config` in campaign.yaml).
* `--round, -r`: `latest | all | 3 | 1,3,7 | 2-5` (plugin may define defaults).
* `--name, -n`: Run a single plot by name; omit to run **all**.
* `--tag`: Run plots with the given tag (repeatable).

**Notes**

* Overwrites files by default; continues on error; exit code **1** if any plot failed.
* Output directory defaults to `outputs/plots`, or honors `output.dir` if provided.
* Plot-specific knobs **must** live under `params:`; top-level plotting keys are errors.
* Prefer `plot_config: plots.yaml` in campaign.yaml to keep runtime config lean.
* `plot_defaults` and `plot_presets` reduce redundancy; `preset: <name>` merges into each plot entry.
* Set `enabled: false` on any plot entry to keep it in the YAML without running it.

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
    tags: [quick]
    params:
      score_field: "pred__y_obj_scalar" # field from run_pred rows
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

#### `prune-source`

Remove OPAL-derived columns (`opal__*`) and the configured Y column from `records.parquet`.

**Usage**

```bash
opal prune-source --config <yaml> [--scope any|campaign] [--keep <col> ...] \
  [--yes] [--backup|--no-backup]
```

**Flags**

* `--config, -c`: Path to `campaign.yaml` (optional if auto-discovery works).
* `--scope`: Which opal namespaces to prune: `any` (default) or `campaign` (this campaign’s slug only).
* `--keep, -k`: Column name(s) to keep even if matched for deletion (repeatable).
* `--yes, -y`: Skip interactive prompt.
* `--backup/--no-backup`: Backup original file before pruning (default: on).

**Notes**

* Designed as a **start fresh** option before re-running round 0.

---

## Typical workflows

#### Initialize a campaign

```bash
opal init --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml
```

Creates `outputs/` and writes/updates `state.json`.

### Ingest labels observed in round *r*


```bash
opal ingest-y \
  --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml \
  --round 0 \
  --csv data/my_new_data_with_labels.csv
```

Appends to label history (`opal__<slug>__label_hist`) and emits `label` events.
(`--observed-round` is an alias for `--round`.)

#### Train–score–select with labels as of round *r*

```bash
opal run \
  --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml \
  --round 0 \
  --k 12
```

You’ll get per-round artifacts, appended `run_pred`/`run_meta` events, and updated caches.
(`--labels-as-of` is an alias for `--round`.)

#### Ephemeral predictions

```bash
opal predict \
  --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml \
  --model-path src/dnadesign/opal/campaigns/my_campaign/outputs/round_0/model.joblib \
  --in new_candidates.parquet \
  --out preds.csv
```

#### Generate plots

```bash
opal plot --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml
opal plot -c . --name score_vs_rank_latest --round latest
```

---

### Typing less

You can often omit `--config` thanks to **auto-discovery**. The CLI tries, in order:

1. **Explicit flag** (`--config`)
2. **Environment variable** `OPAL_CONFIG`
3. **Workspace marker** `.opal/config` in current or parent folders
4. **Nearest campaign.yaml** in current or parent folders
5. **Single fallback** under `src/dnadesign/opal/campaigns/`

If `$OPAL_CONFIG` or `.opal/config` is set but invalid, OPAL exits with an error (no silent fallback).
Marker paths are resolved **relative to the campaign workdir**.
Passing a **directory** to `--config` is only supported for `opal plot`; other commands require a YAML file.
The `.opal/` folder is a lightweight marker created by `opal init` and contains a single `config` file
pointing to `campaign.yaml`. It is safe to delete and will be regenerated.
`plot_config` paths are resolved **relative to the campaign.yaml** that declares them.

Shell completions:

```bash
opal --install-completion zsh   # bash / fish / powershell also supported
exec $SHELL -l
```

Debug tip:

```bash
export OPAL_DEBUG=1  # full tracebacks on internal errors
```

---

### Extending the CLI

Add your own command:

```python
from ..registry import cli_command

@cli_command("my-cmd", help="What it does.")
def cmd_my_cmd(...):
    ...
```

The CLI auto-discovers via `discover_commands()` and mounts with `install_registered_commands()`.

Guidelines:

* Keep commands thin: parse flags, load config/store, call application code.
* Reuse `_common.py` helpers (`resolve_config_path`, `store_from_cfg`, etc.).
* Raise `OpalError` for user-correctable issues; the CLI manages messaging/exit codes.

---

### CLI directory map

```bash
opal/src/cli/
  app.py            # builds Typer app; root callback & Ctrl-C handling
  registry.py       # @cli_command; discovery; install into app
  commands/
    _common.py      # resolve_config_path, store_from_cfg, json_out, internal_error
    ctx.py
    init.py
    ingest_y.py
    log.py
    run.py
    runs.py
    explain.py
    predict.py
    model_show.py
    objective-meta.py
    record_show.py
    status.py
    validate.py
    label_hist.py
    plot.py
    prune-source.py
```

*One command = one job.* Business logic stays in application modules.

---

@e-south
