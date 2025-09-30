
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
* **New IDs** allowed if your CSV includes essentials: `sequence`, `bio_type`, `alphabet`, and the configured X column.
* Appends to `opal__<slug>__label_hist` and writes the current Y column.
* Emits `label` events into `outputs/events.parquet`.

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

* Pulls effective labels with `observed_round ≤ R`, then trains on the current Y column.
* Predicts in batches (`scoring.score_batch_size` or `--score-batch-size`).
* Applies your **objective** to produce a scalar **selection score**.
* Selects with the configured strategy + tie handling.  
  * If `selection.params.exclude_already_labeled: true` (default), designs already labeled at or before `--round` are **excluded from scoring/selection**.


**Artifacts written** (`outputs/round_<r>/`):

* `model.joblib`
* `selection_top_k.csv`
* `round_ctx.json`
* `objective_meta.json`
* `round.log.jsonl` — compact JSONL with stage events and prediction batch progress

**Events appended** to **`outputs/events.parquet`**:

* `run_pred` — one row per candidate with **`pred__y_hat_model`** and **`pred__y_obj_scalar`**, selection rank/flag, and diagnostics.
* `run_meta` — one row per run with model/config/selection snapshot and artifact checksums.

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
  [--in <csv|parquet>] \
  [--out <csv|parquet>]
```

* Scores your input table (defaults to `records.parquet`).
* Writes to stdout as CSV by default; or to `--out`.
* Validates the X column exists.

### `record-show`

Per-record history report (ground truth + per-round predictions/rank/selected).

```
opal record-show \
  --config <yaml> \
  (--id <ID> | --sequence <SEQ>) \
  [--with-sequence] \
  [--json]
```

### `model-show`

Inspect a saved model; optionally dump full feature importances.

```
opal model-show \
  --model-path outputs/round_<r>/model.joblib \
  [--out-dir <dir>]
```

* Always prints model type and params.
* If `--out-dir` is passed, writes full `feature_importance_full.csv` and prints top-20 in JSON.

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
Plot plugins typically read from the campaign’s **`outputs/events.parquet`** and/or **`records.parquet`**. You may add extra sources per plot entry via:

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
3. **Nearest campaign.yaml** in current or parent folders
4. **Single fallback** under `src/dnadesign/opal/campaigns/`

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
    record_show.py
    status.py
    validate.py
    plot.py
```

*One command = one job.* Business logic stays in application modules.

---

@e-south