## OPAL — Optimization with Active Learning

**OPAL** is an [EVOLVEpro-style](https://www.science.org/doi/10.1126/science.adr6006) active-learning engine for DNA/protein sequence design. It fits a top-layer regressor (e.g., `RandomForestRegressor`) on a chosen representation column **X** and a label column **Y**, predicts **Ŷ** for a broader sequence population, and **selects the top-k** per round based on **Ŷ**.

* Reads a **records.parquet** (either from a [**USR**](../usr/README.md) dataset or a local path).
* Writes per-round outputs back to a campaign state and artifacts to `outputs/round_<k>/`.
* Uses registries for **transforms**, **models**, **objectives**, and **selection strategies** to stay extensible. Campaigns swap plugins without touching core code.
* **Built for iterative use**: as new experimental labels arrive, OPAL lets you add labels to new or existing sequences, train a fresh top-layer model, and surface new top candidates—while keeping a reproducible history of past models, predictions, and label events.

---

## Contents

* [Quick install](#quick-install)
* [Repo & campaign layout](#repo--campaign-layout)
* [Demo campaign](#demo-campaign)
* [Core concepts](#core-concepts)
* [Configuration (campaign.yaml)](#configuration-campaignyaml)
* [CLI overview](#cli-overview)
* [Typical workflows](#typical-workflows)
* [Data contracts & write-backs](#data-contracts--write-backs)
* [Determinism, performance, locks](#determinism-performance-locks)

---

## OPAL Layout

```bash
src/dnadesign/opal/src/
├─ cli/                     # CLI app + command registry
│  ├─ app.py               # Typer app entrypoint
│  ├─ registry.py          # auto-discovers and installs commands
│  └─ commands/            # add new CLI commands here (plug-and-play)
│     ├─ run.py            # core pipeline: fit → predict → objective → selection
│     ├─ ingest_y.py       # CSV → labels via transforms_y
│     ├─ explain.py        # dry-run planner (counts, configs)
│     ├─ predict.py        # ephemeral predictions from a frozen model
│     ├─ record_show.py    # per-record history and per-round results
│     ├─ init.py           # scaffold/validate campaign workspace
│     ├─ status.py         # dashboard from state.json
│     └─ validate.py       # table checks (essentials present)
├─ config/                 # YAML loader + plugin param schemas
│  ├─ types.py
│  ├─ plugin_schemas.py
│  └─ loader.py
├─ registries/             # plugin registries (transforms_x/y, models, objectives, selections)
│  ├─ transforms_x.py
│  ├─ transforms_y.py
│  ├─ models.py
│  ├─ objectives.py
│  └─ selections.py
├─ transforms_x/           # concrete X transforms (import triggers registration)
├─ transforms_y/           # concrete Y ingests (import triggers registration)
├─ models/                 # concrete model wrappers
├─ objectives/             # concrete objectives (+ docs)
├─ selection/              # concrete selection strategies
├─ artifacts.py            # artifact writers (selection CSV, round ctx, logs, metrics)
├─ data_access.py          # RecordsStore: IO, label history, fixed-width X
├─ round_context.py        # RoundContext and fingerprinting helpers
├─ writebacks.py           # minimal per-row columns writer
└─ …
```

---

## Quick install

OPAL lives inside the `dnadesign` repo, with a CLI shortcut inside `pyproject.toml`:

```toml
[project.scripts]
opal = "dnadesign.opal.src.cli.app:main"
```

---

## Demo campaign

Run a self-contained example:

1) Generate demo records (requires pandas + pyarrow or fastparquet):
   python src/dnadesign/opal/campaigns/demo/make_records.py

2) Initialize:
   opal init -c src/dnadesign/opal/campaigns/demo/campaign.yaml

3) Ingest labels (round 0):
   opal ingest-y -c src/dnadesign/opal/campaigns/demo/campaign.yaml \
     --round 0 \
     --csv src/dnadesign/opal/campaigns/demo/data/mock_ingest.csv \
     --yes

4) Train + score + select (round 0):
   opal run -c src/dnadesign/opal/campaigns/demo/campaign.yaml -r 0 -k 5

See the walkthrough at:
src/dnadesign/opal/campaigns/demo/README.md

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
│   └─ <campaign>/             # CAMPAIGN is a short slug identifying the run space
│       ├─ campaign.yaml       # configuration (plugin refs + policies)
│       ├─ state.json          # append-only campaign state across rounds
│       ├─ campaign.log.jsonl  # high-level events (ingest, fit, predict, objective, selection)
│       ├─ outputs/
│       │   └─ round_<k>/
│       │       ├─ model.joblib                    # frozen model incl. scaler
│       │       ├─ selection_top_k.csv             # lab handoff (id, sequence, score)
│       │       ├─ feature_importance.csv          # model-dependent (e.g., RF importances)
│       │       ├─ predictions_with_uncertainty.csv# scored universe (+uncertainty, ranks, flags)
│       │       ├─ round_model_metrics.json        # fit/predict timing, OOB metrics, etc.
│       │       ├─ round_ctx.json                  # RoundContext: setpoint, pool, params, fingerprint
│       │       ├─ objective_meta.json             # denominator used, β/γ, other objective metadata
│       │       └─ round.log.jsonl                 # fine-grained events for this round
│       └─ records.parquet     # single source of truth; OPAL writes round columns here
└─ README.md

```

**USR datasets** live elsewhere (usually `src/dnadesign/usr/datasets/<dataset>/records.parquet`) and are **not copied** into campaigns.

---

## Core concepts

* **records.parquet**: single source of truth for sequences and derived columns (**X**, **Y**, **Ŷ**, selection scores, etc.).
* **Representation (X)**: explicitly named column (e.g., `infer__...__logits_mean`), accepted as **Arrow `list<float>`** or **JSON array string**, coerced to `float32`. Fixed dimension across all used rows is required.
* **Label (Y)**: explicitly named column; shape and semantics are campaign-specific and defined by your `transforms_y` and objective.
* **Label history (per campaign)**: `opal__<slug>__label_hist` stores append-only events `{"r": <round>, "y": "<json array or number>", "shape": [optional], "ts": "<iso8601>"}`.
* **Rounds**: `--round k` trains on labels from rounds `≤ k`, scores the candidate universe, ranks, and writes selection metadata for the next lab round.
* **Per-round write-backs (records.parquet)** include:

  * Vector predictions: `opal__<campaign>__r{k}__pred_y`
  * **Persisted selection score**: `opal__<campaign>__r{k}__selection_score__<objective>`
  * Competition rank: `opal__<campaign>__r{k}__rank_competition`
  * Selection flag: `opal__<campaign>__r{k}__selected_top_k_bool`
  * **Uncertainty (optional, scalar)**: e.g., `opal__<campaign>__r{k}__uncertainty__mean_all_std` for tree-based models
  * **Flags**: `opal__<campaign>__r{k}__flags` (compact QC: e.g., `clip_effect|nan_pred`)
  * **Fingerprint**: `opal__<campaign>__r{k}__fingerprint` (short digest for reproducibility)

---

## Configuration (`campaign.yaml`)

Campaigns configure plugin refs; OPAL remains agnostic to specifics.

Key blocks:

- `campaign`: `name`, `slug`, `workdir`
- `data`: `location` (USR/local), `representation_column_name`, `label_source_column_name`, `y_expected_length`
- `transforms_x`: `{ name, params }` (X→matrix)
- `transforms_y`: `{ name, params }` (CSV→labels)
- `models`: `{ name, params }`
- `objectives`: `{ name, params }`
- `selection`: `{ name, params }`
- `training.target_scaler` and `scoring.sort_stability`

Example:
```yaml
campaign: { name: My Campaign, slug: my_campaign, workdir: src/dnadesign/opal/campaigns/my_campaign }
data:
  location: { kind: local, path: campaigns/my_campaign/records.parquet }
  representation_column_name: rep__vec
  label_source_column_name: y
  y_expected_length: 8
transforms_x: { name: identity, params: {} }
transforms_y: { name: logic5_from_tidy_v1, params: { /* ... */ } }
models: { name: random_forest, params: { n_estimators: 100, random_state: 7 } }
objectives: { name: sfxi_v1, params: { setpoint_vector: [0,0,0,1] } }
selection: { name: top_n, params: { top_k_default: 12, tie_handling: competition_rank } }
training:
  target_scaler: { enable: true, minimum_labels_required: 5 }
scoring:
  score_batch_size: 10000
  sort_stability: "(-opal__{slug}__r{round}__selection_score__{objective}, id)"
```

Note: the `{objective}` token in `sort_stability` is filled with your chosen objective name.

---

## CLI overview

```
opal --help
```

**Commands**

* `init --config <yaml>`
  - Initialize/validate the campaign workspace; write `state.json`.

* `ingest-y --config <yaml> --round <k> --csv <path> [--transform <name>] [--params <path.json>]`
  - Ingest external data via a configured transform → preview → interactive → write Y to `records.parquet`.
  - Append `label_hist`.
  - Strict checks (essentials present, X present).

* `run --config <yaml> --round <k> [--k <n>] [--resume|--force] [--score-batch-size <n>]`
  - Train on labels ≤k, score the universe.
  - Evaluate your configured objective.
  - Write artifacts + per-row results.

* `predict --config <yaml> --model-path <outputs/round_k/model.joblib> [--in <csv|parquet>] [--out <csv|parquet>]`
  - Ephemeral inference with a frozen model; no write-backs.

* `record-show --config <yaml> (--id <ID> | --sequence <SEQ> --bio-type <dna|protein> --alphabet <...>) [--with-sequence] [--json]`
  - Per-record report: ground truth & history; per-round predictions, ranks, selection flag.

* `model-show --model-path <outputs/round_k/model.joblib> [--out-dir <dir>]`
  - Show model params; optionally dump full feature importances.

* `explain --config <yaml> --round <k>`
  - Dry-run planner (counts, dedup policy, model config, seeds, universe size). **No writes**.

* `status --config <yaml> [--round <k> | --all] [--json]`
  - Dashboard from `state.json` (latest round by default).

* `validate --config <yaml>`
  - End-to-end table checks (essentials present; X column present).

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

Write-backs in `records.parquet` (example):

* `opal__<campaign>__r<k>__pred_y`
* `opal__<campaign>__r<k>__selection_score__<objective>`
* `opal__<campaign>__r<k>__rank_competition`
* `opal__<campaign>__r<k>__selected_top_k_bool`
* `opal__<campaign>__r<k>__uncertainty__mean_all_std` (optional)
* `opal__<campaign>__r<k>__flags`
* `opal__<campaign>__r<k>__fingerprint`

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
  --model-path campaigns/<campaign_name>/outputs/round_0/model.joblib \
  --in new_candidates.parquet \
  --out preds.csv
```

---

## Architecture & data flow

Handoffs are strict and plugin-driven:

```
tidy.csv ──► transforms_y ──► labels[id,y]
records.parquet[X] ──► transforms_x ──► X
X + labels ──► model.fit
model + RoundContext ──► predict Ŷ ──► objective ──► score ──► selection
```

### RoundContext (recorded per round)
- `slug`, `round_index`, `run_id`, `code_version`
- `setpoint`, `label_ids`, `training_label_count`
- `effect_pool_for_scaling` (if needed by objective)
- `percentile_cfg` (objective scaling config)
- model/transform plugin names+params, `y_expected_length`
- `fingerprint` (short, full)

### Extending the CLI
- Create a new module under `src/dnadesign/opal/src/cli/commands/` (e.g., `my_cmd.py`).
- Decorate the entry function with `@cli_command("my-cmd", help="...")`.
- The command is auto-discovered by `cli.registry.discover_commands()` and added to the app via `install_registered_commands()`.
- Keep commands thin: parse flags, load config/store, and call into core modules.

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

* `opal__<campaign>__label_hist` (append-only per id)
* `opal__<campaign>__r<k>__pred_y`
* `opal__<campaign>__r{k}__selection_score__<objective>`
* `opal__<campaign>__r<k>__rank_competition`
* `opal__<campaign>__r<k>__selected_top_k_bool`
* `opal__<campaign>__r{k}__uncertainty__mean_all_std`

**X (representation)**: Arrow `list<float>` or JSON array string. **Fixed dim** across used rows.
*Tensor convention*: store flattened vector in the main column and an optional shape sidecar.

**Y (labels)**: Arrow `list<float>` for main vector columns; **history `y` stored as a compact JSON string** inside each event to preserve shape flexibly.

---

## Notes

#### Training defaults & per-target scaling

* **Model**: default `RandomForestRegressor` multi-output (swappable via registry).
* **Per-target scaling (fit-time only)**: enabled by default (`robust_iqr_per_target`).

  * Center: median; Scale: `IQR/1.349`; **skip** if `n_labels < 5` or scale ≈ 0.
  * Predictions are **inverse-transformed** back to original units before scoring/writes.

---

@e-south
