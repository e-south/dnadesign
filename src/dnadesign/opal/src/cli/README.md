
# OPAL — Command Line Interface

The OPAL CLI is a thin layer over OPAL’s application modules. It lets you initialize a campaign, ingest labeled samples, train/score/select for a round, inspect records and models, validate your dataset, and generate plots.

This document is **CLI-focused**:

- [Quick start](#quick-start)
- [Command overview](#command-overview)
- [Typical workflows](#typical-workflows)
- [Typing less](#typing-less)
- [Extending the CLI](#extending-the-cli)
- [CLI directory map](#cli-directory-map)

For architecture & concepts, see the **[Top-level README](../../README.md)**.

---

## Quick start

See all available commands and flags:

```bash
opal --help
````

---

## Command overview

Commands are thin wrappers; they call into OPAL’s application layer. Each command should do one thing.

### `init`

Initialize/validate a campaign workspace and write `state.json`.

```
opal init --config <yaml>
```

* Ensures the campaign `workdir` has `outputs/` and `inputs/`.
* Writes/updates `state.json` with campaign identity, data location, and settings.

### `ingest-y`

Transform a tidy CSV/Parquet to model-ready **Y**, which lands in a campaign's `records.parquet`; preview, confirm, and append to label history.

```
opal ingest-y \
  --config <yaml> \
  --observed-round <r> \
  --csv <path> \
  [--transform <name>] \
  [--params <transform_params.json>] \
  [--yes]
```

Behavior & checks:

* Uses `transforms_y` from YAML (overridable via flags).
* **Strict preflights**: schema checks, completeness.
* **Preview is printed** (counts + sample) before any write.
* Duplicate handling is controlled by `ingest.duplicate_policy` (error|keep_first|keep_last).
* **New IDs** allowed if your CSV includes essentials: `sequence`, `bio_type`, `alphabet`, and the configured X column.
* Appends to `opal__<slug>__label_hist` and writes the current Y column.
* Emits `label` events into `outputs/ledger.labels.parquet`.

### `run`

Train on labels with **`observed_round ≤ R`** (where `R` comes from `--labels-as-of`), score the candidate universe, evaluate the objective, select top-k, write artifacts, append canonical events, and update caches.

```
opal run \
  --config <yaml> \
  --labels-as-of <r> \
  [--k <n>] \
  [--resume] \
  [--score-batch-size <n>]
```

Pipeline:

* Pulls effective labels per `training.policy` (cumulative vs current round, dedup policy).
* Predicts in batches (`scoring.score_batch_size` or `--score-batch-size`).
* Applies your **objective** to produce a scalar **selection score**.
* Selects with the configured strategy + tie handling.
  * If `selection.params.exclude_already_labeled: true` (default), designs already labeled are **excluded**; scope is controlled by `training.policy.allow_resuggesting_candidates_until_labeled`.


**Artifacts written** (`outputs/round_<r>/`):

* `model.joblib`
* `model_meta.json`
* `selection_top_k.csv`
* `labels_used.parquet` (training snapshot for this run)
* `round_ctx.json`
* `objective_meta.json`
* `round.log.jsonl` — compact JSONL with stage events and prediction batch progress

**Events appended** to **ledger sinks** under `outputs/`:

* `run_pred` → `outputs/ledger.predictions/` (one row per candidate with **`pred__y_hat_model`** and **`pred__y_obj_scalar`**, selection rank/flag, and diagnostics).
* `run_meta` → `outputs/ledger.runs.parquet` (one row per run with model/config/selection snapshot and artifact checksums).

**Write-backs to `records.parquet` (caches only):**

* `opal__<slug>__latest_as_of_round`
* `opal__<slug>__latest_pred_scalar`

Flags to know:

* `--k` overrides `selection.params.top_k`.
* `--score-batch-size` overrides `scoring.score_batch_size` for this run.
* `--resume` allows overwriting existing per-round artifacts.

### `predict`

Run **ephemeral** predictions from a frozen model. No writes to `records.parquet`.

```
opal predict \
  --config <yaml> \
  [--model-path outputs/round_<r>/model.joblib | --round <r>] \
  [--model-name <registry_name> --model-params <params.json>] \
  [--in <csv|parquet>] \
  [--id-col <name> --sequence-col <name>] \
  [--generate-id-from-sequence] \
  [--out <csv|parquet>]
```

* Scores your input table (defaults to `records.parquet`).
* Writes to stdout as CSV by default; or to `--out` (Parquet keeps vectors as list<float>).
* Validates the X column exists.
* Requires `model_meta.json` next to the model; use `--model-name/--model-params` if missing.
* If your input lacks an id column, use `--generate-id-from-sequence` (requires a sequence column).

### `record-show`

Per-record history report (ground truth + per-round predictions/rank/selected).

```
opal record-show \
  --config <yaml> \
  [<ID-or-SEQ> | --id <ID> | --sequence <SEQ>] \
  [--with-sequence] \
  [--legacy] \
  [--json]
```

### `model-show`

Inspect a saved model; optionally dump full feature importances.

```
opal model-show \
  --model-path outputs/round_<r>/model.joblib \
  [--model-name <registry_name> --model-params <params.json>] \
  [--out-dir <dir>]
```

* Always prints model type and params.
* If `--out-dir` is passed, writes full `feature_importance_full.csv` and prints top-20 in JSON.

### `objective-meta`

List objective metadata and diagnostic keys for a round.

```
opal objective-meta --config <yaml> --round <k|latest> [--legacy]
```

* Reads from `outputs/ledger.runs.parquet` and `outputs/ledger.predictions/`.
* `--legacy` allows older sinks (deprecated).

### `explain`

Dry-run planner for a round: counts, plan, warnings. **No writes.**

```
opal explain --config <yaml> --round <k>
```

Prints: number of training labels, candidate universe size, transforms/models/selection used, vector dimension, and any preflight warnings.

### `status`

Dashboard from `state.json`.

```
opal status --config <yaml> [--round <k> | --all] [--json]
```

* Default: latest round summary.
* `--round k`: specific round details.
* `--all`: dump every round (JSON-friendly).

### `validate`

End-to-end table checks (essentials present; X present).

```
opal validate --config <yaml>
```

* Verifies **USR essentials** exist in `records.parquet`.
* Verifies the configured **X** column exists.
* If Y is present, validates vector length & numeric/finite cells.

### `label-hist`

Validate or repair the label history column (explicit, no silent fixes).

```
opal label-hist validate --config <yaml>
opal label-hist repair --config <yaml> [--apply]
```

* `repair` is **dry-run** by default; use `--apply` to write.

### `plot`

Generate plots declared in the campaign’s `plots:` block. Plots are plugin-driven and campaign-scoped.

```
opal plot --config <yaml-or-dir> [--round <selector>] [--name <plot-name>]
```

* `--round <selector>`: `latest | all | 3 | 1,3,7 | 2-5` (plugin may define defaults).
* `--name <plot-name>`: run a single plot by name; omit to run **all**.
* Overwrites files by default; continues on error; exit code **1** if any plot failed.

**Campaign YAML (example)**

```yaml
plots:
  - name: score_vs_rank_latest
    kind: scatter_score_vs_rank     # plot plugin id
    params:
      score_field: "pred__y_obj_scalar"  # field from run_pred rows
      hue: null                           # or "round"
      highlight_selected: false
    output:
      format: "png"                       # png/svg/pdf
      dpi: 600
      dir: "{campaign}/plots/{kind}/{name}"
      filename: "{name}{round_suffix}.png"
```

**Data sources**
Plot plugins typically read from the campaign’s **ledger sinks** under `outputs/` and/or **`records.parquet`**. You may add extra sources per plot entry via:

```yaml
data:
  - name: extra_csv
    path: ./extras/scores.csv
```

---

## Typical workflows

### Initialize a campaign

```bash
opal init --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml
```

Creates `outputs/` and writes/updates `state.json`.

### Ingest labels observed in round *r*


```bash
opal ingest-y \
  --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml \
  --observed-round 0 \
  --csv data/my_new_data_with_labels.csv
```

Appends to label history (`opal__<slug>__label_hist`) and emits `label` events.

### Train–score–select with labels as of round *r*

```bash
opal run \
  --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml \
  --labels-as-of 0 \
  --k 12
```

You’ll get per-round artifacts, appended `run_pred`/`run_meta` events, and updated caches.

### Ephemeral predictions

```bash
opal predict \
  --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml \
  --model-path src/dnadesign/opal/campaigns/my_campaign/outputs/round_0/model.joblib \
  --in new_candidates.parquet \
  --out preds.csv
```

### Generate plots

```bash
opal plot --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml
opal plot -c . --name score_vs_rank_latest --round latest
```

---

## Typing less

You can often omit `--config` thanks to **auto-discovery**. The CLI tries, in order:

1. **Explicit flag** (`--config`)
2. **Environment variable** `OPAL_CONFIG`
3. **Workspace marker** `.opal/config` in current or parent folders
4. **Nearest campaign.yaml** in current or parent folders
5. **Single fallback** under `src/dnadesign/opal/campaigns/`

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

## Extending the CLI

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

## CLI directory map

```bash
opal/src/cli/
  app.py            # builds Typer app; root callback & Ctrl-C handling
  registry.py       # @cli_command; discovery; install into app
  commands/
    _common.py      # resolve_config_path, store_from_cfg, json_out, internal_error
    init.py
    ingest_y.py
    run.py
    explain.py
    predict.py
    model_show.py
    objective-meta.py
    record_show.py
    status.py
    validate.py
    label_hist.py
    plot.py
```

*One command = one job.* Business logic stays in application modules.

---

@e-south
