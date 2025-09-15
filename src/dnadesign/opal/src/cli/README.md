# OPAL -- Command Line Interface

The OPAL CLI is a thin layer over OPAL’s application modules. It lets you initialize a campaign, ingest new labeled samples, train/score/select for a round, inspect records and models, and validate your dataset.


This document is **CLI-focused**:


* [Quick start](#quick-start)
* [Command overview](#command-overview)
* [Typical workflows](#typical-workflows)
  * [Initialize a campaign](#initialize-a-campaign)
  * [Ingest labels for round *k* (CSV → Y)](#ingest-labels-for-round-k)
  * [Train, score, and select for a round](#train-score-select-for-a-round)
  * [Inspect the campaign state](#inspect-the-campaign-state)
  * [Ephemeral predictions](#ephemeral-predictions)
* [Typing less](#typing-less)
* [Extending the CLI](#extending-the-cli)
* [CLI directory map](#cli-directory-map)

---

## Quick start

See all available commands and flags:

```bash
opal --help
```

---

## Command overview

Command modules are thin wrappers; they call into OPAL’s application layer and are designed to fail fast given errors. Each command is designed to do one thing. 

> Tip: Most commands accept `--config /path/to/campaign.yaml`. You can often omit it—see the [Typing less](#typing-less) section below.

### `init`

Initialize/validate a campaign workspace and write `state.json`.

```
opal init --config <yaml>
```

* Ensures the campaign `workdir` exists with `outputs/`.
* Writes/updates `state.json` with campaign identity, data location, and settings.

### `ingest-y`

Transform a tidy CSV/Parquet to model-ready **Y**, preview, confirm, and append to label history.

```
opal ingest-y \
  --config <yaml> \
  --round <k> \
  --csv <path> \
  [--transform <name>] \
  [--params <path.json>] \
  [--yes]
```

Behavior & checks:

* Uses the `transforms_y` from your YAML (overridable via flags).
* **Strict preflights**: schema expectations, completeness, etc.
* **Preview is always printed** (counts + sample) before any write.
* **New IDs** are allowed if your CSV includes **essentials**: `sequence`, `bio_type`, `alphabet`, and the configured **X** column (representation).
* Appends to `opal__<slug>__label_hist` and writes the current Y column.
* Idempotent per `(id, round)`: refuses to change label history for the same `(id, r)`.

### `run`

Train on all labels **≤ round k**, score the broader candidate sequence population, evaluate the objective, select top-k, write artifacts, and write back per-round results to `records.parquet`.

```
opal run \
  --config <yaml> \
  --round <k> \
  [--k <n>] \
  [--resume] \
  [--force] \
  [--score-batch-size <n>]
```

* Pulls **effective labels** up to round `k` per campaign policy.
* Fits the configured model (e.g., RandomForest).
* Predicts in batches; computes uncertainty (e.g., mean of per-tree std).
* Applies your **objective** to produce a scalar **selection score**.
* Selects with the configured **selection strategy** and **tie handling**.
* **Artifacts** to `outputs/round_<k>/`:

  * `model.joblib`, `selection_top_k.csv`, `feature_importance.csv`,
  * `predictions_with_uncertainty.csv`, `round_model_metrics.json`,
  * `round_ctx.json` (RoundContext), `round.log.jsonl`, `objective_meta.json`
* **Write-backs** into `records.parquet` (per round `k`):

  * `opal__<slug>__r<k>__pred_y`
  * `opal__<slug>__r<k>__selection_score__<objective>`
  * `opal__<slug>__r<k>__rank_competition`
  * `opal__<slug>__r<k>__selected_top_k_bool`
  * `opal__<slug>__r<k>__uncertainty__mean_all_std` *(optional)*
  * `opal__<slug>__r<k>__flags`
  * `opal__<slug>__r<k>__fingerprint`
  * `opal__<campaign>__label_hist`

Flags to know:

* `--resume`/`--force` allow overwriting existing per-round columns if you re-run a round.
* `--k` overrides the default top-k from YAML.
* `--score-batch-size` overrides YAML for one run.

### `predict`

Run **ephemeral** predictions from a frozen model. No writes to `records.parquet`.

```
opal predict \
  --config <yaml> \
  --model-path outputs/round_<k>/model.joblib \
  [--in <csv|parquet>] \
  [--out <csv|parquet>]
```

* Scores your input table (defaults to `records.parquet` if `--in` not provided).
* Writes to stdout as CSV by default; or to `--out` file.
* Validates that the representation column **X** exists.

### `record-show`

Compact per-record history report (ground truth + per-round predictions/rank/selected).

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
  --model-path outputs/round_<k>/model.joblib \
  [--out-dir <dir>]
```

* Always prints model type and params.
* If `--out-dir` is passed, writes full `feature_importance_full.csv` and prints top-20 in JSON.

### `explain`

Dry-run planner for a round: prints counts, plan, and warnings. **No writes.**

```
opal explain --config <yaml> --round <k>
```

* Shows number of training labels, candidate universe size, transforms/models/selection used, vector dimension, and any preflight warnings.

### `status`

View a dashboard from `state.json`.

```
opal status --config <yaml> [--round <k> | --all] [--json]
```

* By default: latest round summary.
* `--round k`: specific round details.
* `--all`: dump every round (JSON-friendly).

### `validate`

End-to-end table checks (essentials present; X column present).

```
opal validate --config <yaml>
```

* Verifies required **USR essentials** exist in `records.parquet`.
* Verifies the configured representation column **X** exists.

---

## Typical workflows

Below are some usage examples. Adjust paths and names to your campaign. Also see the [**hands-on demo**](../../campaigns/demo/README.md) for a complete, runnable example, which covers generating demo records, initializing the campaign, ingesting labels, running round 0, and inspecting outputs step-by-step.

### Initialize a campaign

```bash
opal init --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml
```

What this does:

* Validates your config and data references.
* Creates `outputs/` and writes/updates `state.json`.

### Ingest labels for round *k*

Prepare a tidy file (CSV/Parquet) with:

* Required identifiers: typically `id` (or `design_id`) and any fields your `transforms_y` needs.
* For **new** records not yet in `records.parquet`, include the **essentials**:

  * `sequence`, `bio_type`, `alphabet`, and the configured **X** column.

```bash
opal ingest-y \
  --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml \
  --round 0 \
  --csv data/my_new_data_with_labels.csv
# OPAL prints a preview; confirm to proceed.
```

This appends to label history (`opal__<slug>__label_hist`) and writes the current Y column.

### Train-score-select for a round

```bash
opal run \
  --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml \
  --round 0 \
  --k 12
```

You’ll get:

* **Artifacts** in `outputs/round_0/`:

  * `model.joblib`, `selection_top_k.csv`, `predictions_with_uncertainty.csv`,
  * `feature_importance.csv`, `round_model_metrics.json`,
  * `round_ctx.json`, `objective_meta.json`, `round.log.jsonl`
* **Write-backs** to `records.parquet` for round 0 (see command overview).

### Inspect the campaign state

```bash
# Campaign status
opal status --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml

# Dry-run planner for next round
opal explain --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml --round 1

# Per-record card
opal record-show --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml --id e153ebc...

# Inspect saved model params & optional feature importances
opal model-show --model-path src/dnadesign/opal/campaigns/my_campaign/outputs/round_0/model.joblib
```

### Ephemeral predictions

Use a frozen model to score a new table:

```bash
opal predict \
  --config src/dnadesign/opal/campaigns/my_campaign/campaign.yaml \
  --model-path src/dnadesign/opal/campaigns/my_campaign/outputs/round_0/model.joblib \
  --in new_candidates.parquet \
  --out preds.csv
```

---

## Typing less

You can often omit `--config` thanks to **auto-discovery**. The CLI tries, in order:

1. **Explicit flag**: If you pass `--config`, it uses that path.
2. **Environment variable (`OPAL_CONFIG`)**:

   * An *environment variable* is a simple key-value set in your shell session.
   * Setting it like

     ```bash
     export OPAL_CONFIG=/absolute/path/to/campaign.yaml
     ```

     tells the CLI where your config lives for **this terminal session**.
   * Close the terminal and it resets. If you want it to persist, add the export to your shell rc file or use a tool like `direnv`.
3. **Current or parent folders**: If your current working directory (or any parent) contains a `campaign.yaml`, OPAL auto-uses it.

   ```bash
   # from campaigns/my_campaign/
   opal status
   opal run -r 0
   opal explain -r 1
   ```
4. **Single fallback**: If there is **exactly one** `campaign.yaml` under `src/dnadesign/opal/campaigns/`, it will be used.

#### Shell completions (press TAB for suggestions)

Install once per shell:

```bash
opal --install-completion zsh   # or: bash / fish / powershell
exec $SHELL -l                  # reload your shell; or source your rc file
```

Then try:

```bash
opal <TAB>
opal run --<TAB>
```

#### Debugging tip

Set `OPAL_DEBUG=1` to print full tracebacks on internal errors:

```bash
export OPAL_DEBUG=1
```

(Otherwise, the CLI prints a concise message and a hint.)

---

### Extending the CLI

Add your own command in three steps:

1. Create a new module under `src/dnadesign/opal/src/cli/commands/`, e.g. `my_cmd.py`.
2. Decorate your function:

```python
from ..registry import cli_command

@cli_command("my-cmd", help="What it does.")
def cmd_my_cmd(...):
    ...
```

3. The CLI auto-discovers it via `discover_commands()` and mounts it with `install_registered_commands()`.

#### Guidelines

* Keep the command thin: parse flags, load config/store, call application code.
* Reuse `_common.py` helpers for config resolution, stores, and JSON output.
* Raise `OpalError` for user-correctable issues; let the CLI handle messaging/exit codes.

---

## CLI directory map
```text
opal/src/cli/
  app.py            # builds Typer app; root callback & Ctrl-C handling
  registry.py       # @cli_command decorator; discovery; install into app
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
```

*One command = one job.* Business logic stays in application modules.

---



@e-south