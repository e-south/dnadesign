## OPAL — Optimization with Active Learning

**OPAL** is an [EVOLVEpro-style](https://www.science.org/doi/10.1126/science.adr6006) active-learning engine for DNA/protein sequence design. It fits a top-layer regressor (e.g., `RandomForestRegressor`) on a chosen representation column **X** and a label column **Y**, predicts **Ŷ** for the wider candidate set, and **selects the top-k** per round based on **Ŷ**.

* Works with [**USR**](../usr/README.md) datasets or a local `records.parquet` (standalone).
* Writes per-round predictions/ranks/selection flags back to the **source table**; models and selection files live in the **campaign folder**.
* Uses registries for **models**, **transforms**, and **selection strategies** to stay extensible.
* Built for iterative use: as new experimental labels arrive, OPAL lets you import labels, train a new top-layer model, and surface new top candidates—while keeping a reproducible history of past models, predictions, and label events.

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

## Quick install

OPAL lives inside the `dnadesign` repo, with a CLI shortcut inside `pyproject.toml`:

```toml
[project.scripts]
opal = "dnadesign.opal.src.cli:main"
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
│       ├─ campaign.yaml       # campaign initialization criteria lives here 
│       ├─ state.json
│       ├─ campaign.log.jsonl
│       ├─ outputs/
│       │   └─ round_<k>/
│       │       ├─ model.joblib
│       │       ├─ selection_top_k.csv
│       │       ├─ feature_importance.csv
│       │       ├─ round_model_metrics.json
│       │       └─ round.log.jsonl
│       └─ records.parquet     # where X and y live
└─ README.md
```

**USR datasets** live elsewhere (usually `src/dnadesign/usr/datasets/<dataset>/records.parquet`) and are **not copied** into campaigns.

---

## Core concepts

* **records.parquet**: single source of truth for sequences and derived columns (X, y, ŷ).
* **Representation (X)**: explicitly named column (e.g., `infer__...__logits_mean`), accepted as **Arrow list<float>** or **JSON array string**, coerced to `float32`. 
  - Fixed dimension across all used rows is required.
* **Label (Y)**: explicitly named numeric column (e.g., `exp__two_factor_output_activity_pdual10`).
* **Label history (per campaign)**: `opal__<campaign>__label_hist` stores append-only events `{"r": <round>, "y": <float>, "ts": <iso8601>}` (for reproducibility).
* **Rounds**: When you `run --round k`, OPAL trains on labels from rounds `≤ k`, scores the candidate universe, ranks by predicted Y, and writes selection for the **next** lab round.
* **Write-backs** (per round):
  `opal__<slug>__r<k>__pred`, `opal__<slug>__r<k>__rank_competition`, `opal__<slug>__r<k>__selected_top_k_bool`.

---

## Configuration (`campaign.yaml`)

Minimal USR-backed example:

```yaml
campaign:
  name: "T7 Polymerase"
  slug: "t7_polymerase"
  workdir: "./src/dnadesign/opal/campaigns/t7_polymerase"

data:
  location:
    kind: "usr"
    dataset: "60bp_dual_promoter_cpxR_LexA"
    usr_root: "./src/dnadesign/usr/datasets"
  representation_column_name: "infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean"
  label_source_column_name: "exp__two_factor_output_activity_pdual10"
  transform:
    name: "identity"
    params: {}

training:
  model:
    name: "random_forest"
    params:
      n_estimators: 100
      criterion: "friedman_mse"
      bootstrap: true
      oob_score: true
      random_state: 7
      n_jobs: -1
  policy:
    cumulative_training: true
    label_cross_round_deduplication_policy: "latest_only"
    allow_resuggesting_candidates_until_labeled: true

selection:
  strategy: "top_n"
  top_k_default: 12
  tie_handling: "competition_rank"

scoring:
  score_batch_size: 10000
  sort_stability: "(-y_pred, id)"

safety:
  fail_on_mixed_biotype_or_alphabet: true
  require_biotype_and_alphabet_on_init: true
  conflict_policy_on_duplicate_ids: "error"
  write_back_requires_columns_present: true
  accept_x_mismatch: false

metadata:
  objective: "maximize"
  notes: ""
```

**Standalone**: set `data.location.kind: "local"` and `path: "./src/dnadesign/opal/campaigns/<slug>/records.parquet"`.

---

## CLI overview

```
opal --help
```

**Commands**

* `init --config <yaml>`
  Create/validate the campaign workspace and write initial `state.json`.

* `run|fit --config <yaml> --round <k> [--k <n>] [--resume|--force] [--score-batch-size <n>]`
  Train on labels ≤k, score the universe, write round artifacts, write-back predictions/ranks/selection.

* `labels-import --config <yaml> --round <k> --path <csv|parquet> [--allow-overwrite-meta]`
  Import labels (id,y,\[sequence,bio\_type,alphabet,…]); update label column and `label_hist`. Fail fast if any labeled id lacks X.

* `predict --config <yaml> --model-path <outputs/round_k/model.joblib> [--in <csv|parquet>] [--out <csv|parquet>]`
  Ephemeral inference with an explicit model path; no write-backs.

* `record-show --config <yaml> (--id <ID> | --sequence <SEQ> --bio-type <dna|protein> --alphabet <...>) [--with-sequence] [--json]`
  Compact per-record report: ground truth + src round from history; per-round y\_pred/rank/selected if present.

* `model-show --model-path <outputs/round_k/model.joblib> [--out-dir <dir>]`
  Show model type/params; optionally dump full feature importances.

* `explain --config <yaml> --round <k>`
  Dry-run planner: counts, dedup policy effects, model config, seeds, universe size. **No writes**.

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
opal init --config src/dnadesign/opal/campaigns/t7_polymerase/campaign.yaml
```

### 2) Import labels for round 0

`labels.csv` must have: `id,y` (optionally `sequence,bio_type,alphabet` and **namespaced** meta).
Labeled new rows must also include the configured **X column**.

```bash
opal labels-import \
  --config src/dnadesign/opal/campaigns/t7_polymerase/campaign.yaml \
  --round 0 \
  --path data/R0_labels.csv
```

### 3) Train & score round 0 (produce selections for round 1)

```bash
opal run \
  --config src/dnadesign/opal/campaigns/t7_polymerase/campaign.yaml \
  --round 0 --k 12
```

Artifacts appear at:

```
campaigns/t7_polymerase/outputs/round_0/
  model.joblib
  selection_top_k.csv
  feature_importance.csv
  round_model_metrics.json
```

Predictions/ranks/flags are written back to `records.parquet`:

* `opal__t7_polymerase__r0__pred`
* `opal__t7_polymerase__r0__rank_competition`
* `opal__t7_polymerase__r0__selected_top_k_bool`

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
  --model-path campaigns/t7_polymerase/outputs/round_0/model.joblib \
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
* `opal__<slug>__r<k>__pred`
* `opal__<slug>__r<k>__rank_competition`
* `opal__<slug>__r<k>__selected_top_k_bool`

**Representation column (X)**

* Explicit name (`data.representation_column_name`)
* Accepts Arrow `list<float>` or JSON array string like `"[0.1, 0.2]"`
* Coerced to `np.float32`; **fixed dimension** across all used rows; ragged/non-numeric → **fail fast**.

**Label column (Y)**

* Explicit name (`data.label_source_column_name`)
* Numeric; training dedup across rounds ≤k: **latest\_only**.

**Uniformity**

* Mixed `bio_type`/`alphabet` across training or candidate rows → **fail fast** with a concise summary and sample IDs.

---

## Determinism, performance, locks

* **Deterministic ordering**: sort by `(-y_pred, id)`.
* **Ranking**: competition ranking (`1,2,3,3,5`) and **include all ties** at the k-th boundary.
* **Performance**: `n_jobs=-1` for RF; scoring in batches (default **10,000**, configurable).
* **Locking**: campaign-level file lock; rely on USR’s atomic writes for `records.parquet`.

---

## Exit codes

* `0` success (no warnings)
* `2` success with warnings
* `3` contract/pre-flight violation (missing X, ragged dims, mixed bio/alphabet, no labels ≤k, etc.)
* `4` not found / path errors
* `5` invalid CLI usage / bad arguments
* `6` artifact exists; refused without `--resume/--force`
* `7` lock acquisition failed
* `8` checksum mismatch / corruption detected
* `9` uncaught/internal error

---

@e-south