## OPAL — Optimization with Active Learning

**OPAL** is an [EVOLVEpro-style](https://www.science.org/doi/10.1126/science.adr6006) active-learning engine for DNA/protein sequence design. It fits a top-layer regressor (e.g., `RandomForestRegressor`) on a chosen representation column **X** and a label column **Y**, predicts **Ŷ** for a broader sequence population, and **selects the top-k** per round based on **Ŷ**.

* Reads a **records.parquet** (either from a [**USR**](../usr/README.md) dataset or a local path).
* Writes per-round outputs back to the **campaign state** and artifacts to `outputs/round_<k>/`.
* Uses registries for **transforms**, **models**, **objectives**, and **selection strategies** to stay extensible.
* **Built for iterative use**: as new experimental labels arrive, OPAL lets you add labels to new or existing sequences, train a fresh top-layer model, and surface new top candidates—while keeping a reproducible history of past models, predictions, and label events.

---

## Contents

* [Quick install](#quick-install)
* [Repo & campaign layout](#repo--campaign-layout)
* [Core concepts](#core-concepts)
* [Configuration (campaign.yaml)](#configuration-campaignyaml)
* [CLI overview](#cli-overview)
* [Typical workflows](#typical-workflows)
* [Data contracts & write-backs](#data-contracts--write-backs)
* [Determinism, performance, locks](#determinism-performance-locks)
* [Exit codes](#exit-codes)

---

## OPAL Layout

```bash
src/dnadesign/opal/src/
├─ cli/
│  ├─ app.py
│  ├─ registry.py
│  └─ commands/
│     ├─ _common.py
│     ├─ init.py
│     ├─ ingest_y.py
│     ├─ run.py
│     ├─ explain.py
│     ├─ predict.py
│     ├─ model_show.py
│     ├─ record_show.py
|     ├─ validate.py
│     └─ status.py
├─ config/
│  ├─ types.py
│  ├─ plugin_schemas.py
│  └─ loader.py
├─ registries/
│  ├─ transforms_x.py
│  ├─ transforms_y.py
│  ├─ models.py
│  ├─ objectives.py
│  └─ selections.py
├─ transforms_x/…     
├─ transforms_y/…
├─ models/…   
├─ objectives/…               
├─ selection/…                  
├─ artifacts.py, data_access.py, explain.py, ingest.py, …
└─ …
```

---

## Quick install

OPAL lives inside the `dnadesign` repo, with a CLI shortcut inside `pyproject.toml`:

```toml
[project.scripts]
opal = "dnadesign.opal.src.cli.app:main"
```

Install in editable mode (from the `dnadesign` repo root):

```bash
uv pip install -e .
# or: pip install -e .
```

---

## Repo and campaign layout

```
src/dnadesign/opal/
├─ src/                        # OPAL code (modules)
├─ campaigns/                  # all OPAL campaigns live here
│   └─ <campaign_name>/
│       ├─ campaign.yaml
│       ├─ state.json
│       ├─ campaign.log.jsonl
│       ├─ outputs/
│       │   └─ round_<k>/
│       │       ├─ model.joblib
│       │       ├─ selection_top_k.csv
│       │       ├─ feature_importance.csv
│       │       ├─ predictions_with_uncertainty.csv
│       │       ├─ round_model_metrics.json
│       │       └─ round.log.jsonl
│       └─ records.parquet     # where X and Y live; OPAL writes back round columns here
└─ README.md

```

**USR datasets** live elsewhere (usually `src/dnadesign/usr/datasets/<dataset>/records.parquet`) and are **not copied** into campaigns.

---

## Core concepts

* **records.parquet**: single source of truth for sequences and derived columns (**X**, **Y**, **Ŷ**, selection scores, etc.).
* **Representation (X)**: explicitly named column (e.g., `infer__...__logits_mean`), accepted as **Arrow `list<float>`** or **JSON array string**, coerced to `float32`. Fixed dimension across all used rows is required.
* **Label (Y)**: explicitly named column; may be **scalar** or **vector**.
* **Label history (per campaign)**: `opal__<slug>__label_hist` stores append-only events `{"r": <round>, "y": "<json array or number>", "shape": [optional], "ts": "<iso8601>"}`.
* **Rounds**: `--round k` trains on labels from rounds `≤ k`, scores the candidate universe, ranks, and writes selection metadata for the next lab round.
* **Per-round write-backs (records.parquet)** include:

  * Vector predictions: `opal__<slug>__r{k}__pred_y`
  * **Persisted selection score**: `opal__<slug>__r{k}__selection_score__<objective>`
  * Competition rank: `opal__<slug>__r{k}__rank_competition`
  * Selection flag: `opal__<slug>__r{k}__selected_top_k_bool`
  * **Uncertainty (scalar)**: `opal__<slug>__r{k}__uncertainty__mean_all_std` (mean of per-output stds)

---

## Configuration (`campaign.yaml`)

Below is a template that shows the key blocks you’ll typically configure for an OPAL
campaign. Keep comments for guidance; trim fields that don’t apply to you.

```yaml

```

**Standalone**: set `data.location.kind: "local"` and `path: "./src/dnadesign/opal/campaigns/<slug>/records.parquet"`.

---

## CLI overview

```
opal --help
```

**Commands**

* `init --config <yaml>`
  Initialize/validate the campaign workspace; write `state.json`.

* `ingest-y --config <yaml> --round <k> --csv <path> [--transform <name>] [--params <path.json>]`
  Ingest external data via a configured transform → preview → interactive → write Y to `records.parquet` and append `label_hist`. Strict checks (essentials present, X present).

* `run|fit --config <yaml> --round <k> [--k <n>] [--resume|--force] [--score-batch-size <n>]`
  Train on labels ≤k, score the universe, write round artifacts, and write back predictions/ranks/selection/uncertainty.

* `predict --config <yaml> --model-path <outputs/round_k/model.joblib> [--in <csv|parquet>] [--out <csv|parquet>]`
  Ephemeral inference with a frozen model; no write-backs.

* `record-show --config <yaml> (--id <ID> | --sequence <SEQ> --bio-type <dna|protein> --alphabet <...>) [--with-sequence] [--json]`
  Per-record report: ground truth & history; per-round predictions, ranks, selection flag.

* `model-show --model-path <outputs/round_k/model.joblib> [--out-dir <dir>]`
  Show model params; optionally dump full feature importances.

* `explain --config <yaml> --round <k>`
  Dry-run planner (counts, dedup policy, model config, seeds, universe size). **No writes**.

* `status --config <yaml> [--round <k> | --all] [--json]`
  Dashboard from `state.json` (latest round by default).

* `validate --config <yaml>`
  End-to-end table checks (essentials present; X column present).

### Typing less with config auto-discovery

You can often omit `--config` entirely:

* **Inside a campaign folder:** If your CWD (or any parent) contains `campaign.yaml`, `opal` will auto-use it.

  ```bash
  # from campaigns/t7_polymerase/
  opal status
  opal run -r 0
  opal explain -r 1
  ```
* **Anywhere with an env var:** Set once per shell (or via direnv) and forget `-c`.

  ```bash
  export OPAL_CONFIG=/abs/path/to/campaign.yaml
  opal status
  opal explain -r 2
  ```
* **Single fallback:** If there’s exactly one `campaign.yaml` under
  `src/dnadesign/opal/campaigns/`, `opal` will use that.

If you want completions:

```bash
# one-time per shell
opal --install-completion zsh   # or bash/fish/powershell
exec $SHELL -l                  # or: source your rc file
```

Then try:

```bash
opal <TAB>
opal run --<TAB>
```

---

## Typical workflows

### 1) Initialize a campaign (USR)

```bash
opal init --config src/dnadesign/opal/campaigns/prom60-etoh-cipro-andgate/campaign.yaml
```

### 2) Ingest labels for a round (CSV → Y)

`my_new_data_with_labels.csv` should include **at minimum** `design_id`, `experiment_id`, and any **transform-specific** columns required by your ingestion transform to compute model-ready Y. If a design is **not already present** in `records.parquet`, the file must also supply the **essentials** to create it: `sequence`, `bio_type`, `alphabet`, and the configured **X representation** column. OPAL validates the schema and transform preconditions, prints a preview, and then prompts you to confirm before writing.

```bash
opal ingest-y \
  --campaign my_campaign_name \
  --round 0 \
  --csv data/my_new_data_with_labels.csv
# OPAL prints a preview + warnings (if any), then prompts:
# Proceed to write labels for round 0? (y/N)
```

### 3) Train and score a round (produce selections for round n+1)

```bash
opal run \
  --config src/dnadesign/opal/campaigns/my_campaign_name/campaign.yaml \
  --round 0 --k 12
```

Artifacts appear at:

```
campaigns/my_campaign_name/outputs/round_0/
  model.joblib
  selection_top_k.csv
  feature_importance.csv
  predictions_with_uncertainty.csv
  round_model_metrics.json
  round.log.jsonl
```

Write-backs in `records.parquet`:

* `opal__my_campaign_name__r0__pred_vec5`
* `opal__my_campaign_name__r0__selection_score__logic_plus_effect_v1`
* `opal__my_campaign_name__r0__rank_competition`
* `opal__my_campaign_name__r0__selected_top_k_bool`
* `opal__my_campaign_name__r0__uncertainty__mean_all_std`

### 4) Status / explain / model / record reports

```bash
opal status --config .../campaign.yaml
opal explain --config .../campaign.yaml --round 0
opal model-show --model-path .../round_0/model.joblib
opal record-show --config .../campaign.yaml --id e153ebc...
```

### 5) Ephemeral predictions on a frozen model

```bash
opal predict \
  --config .../campaign.yaml \
  --model-path campaigns/my_campaign_name/outputs/round_0/model.joblib \
  --in new_candidates.parquet \
  --out preds.csv
```

---

## Data contracts & write-backs

### Essentials (USR schema)

| column       | type            | notes                  | 
| ------------ | --------------- | ---------------------- | 
| `id`         | string          | sha1 `bio_type`        | 
| `bio_type`   | string          | `"dna"` or `"protein"` | 
| `sequence`   | string          | case-insensitive       | 
| `alphabet`   | string          | e.g. `dna_4`           | 
| `length`     | int32           | `len(sequence)`        | 
| `source`     | string          | provenance             | 
| `created_at` | timestamp (UTC) | ingest time            | 

**Namespacing rule**: Secondary columns are `<tool>__<field>`. OPAL writes only:

* `opal__<slug>__label_hist` (append-only per id)
* `opal__<slug>__r<k>__pred_y`
* `opal__<slug>__r{k}__selection_score__<objective>`
* `opal__<slug>__r<k>__rank_competition`
* `opal__<slug>__r<k>__selected_top_k_bool`
* `opal__<slug>__r{k}__uncertainty__mean_all_std`

**X (representation)**: Arrow `list<float>` or JSON array string. **Fixed dim** across used rows.
*Tensor convention*: store flattened vector in the main column and an optional shape sidecar.

**Y (labels)**: Arrow `list<float>` for main vector columns; **history `y` stored as a compact JSON string** inside each event to preserve shape flexibly.

---

## Predictions & uncertainty artifacts

**File**: `campaigns/<slug>/outputs/round_<k>/predictions_with_uncertainty.csv`

**Columns**

* `id`
* `round`
* `y_pred_vec` — JSON string of the length-5 vector
* `y_pred_std_vec` — JSON string (per-output std across RF trees, length-5)
* `logic_fidelity_l2_norm01` — `[0,1]` (from `v` vs setpoint)
* `effect_scaled_p95` — `[0,1]` (from `e` and per-round denom)
* `selection_score_logic_x_effect_v1` — scalar used for ranking
* `selection_score_std` — std across **tree-level scalar scores** (evaluate objective per tree)
* `model_params_sha256` — hash of RF + scaler params
* `generated_at` — ISO8601 timestamp

**Uncertainty in records.parquet** (scalar):
`opal__<slug>__r{k}__uncertainty__mean_all_std = mean(y_pred_std_vec)`.

---

## Notes

#### Training defaults & per-target scaling

* **Model**: `RandomForestRegressor` multi-output, fixed hyperparams (Evolve-style): `n_estimators=100`, `criterion="friedman_mse"`, `bootstrap=True`, `oob_score=True`, `max_features=1.0`, `random_state=7`, `n_jobs=-1`.
* **Per-target scaling (fit-time only)**: enabled by default (`robust_iqr_per_target`).

  * Center: median; Scale: `IQR/1.349`; **skip** if `n_labels < 5` or scale ≈ 0.
  * Predictions are **inverse-transformed** back to original units before scoring/writes.

---

@e-south



