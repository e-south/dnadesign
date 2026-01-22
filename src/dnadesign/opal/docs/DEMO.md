## OPAL Demo Campaign -- SFXI (setpoint x intensity)

This demo walks a **complete OPAL loop** on a small dataset with the SFXI ingest + plotting stack.

**What you'll learn (and see):**
- a full `run` (train -> score -> select)
- ledger-backed inspection (`status`, `runs`, `log`)
- plots (quick + configured)
- where artifacts and ledgers live

This demo is **self-contained**: it ships with a local `records.parquet` and label CSVs under `inputs/`, so no USR setup is required.

---

### TL;DR

From the repo root:

```bash
cd src/dnadesign/opal/campaigns/demo/

# 1) Initialize & validate
uv run opal init     -c campaign.yaml
uv run opal validate -c campaign.yaml

# 2) Ingest round-0 labels
uv run opal ingest-y -c campaign.yaml --round 0 \
  --csv inputs/r0/demo_y_sfxi.csv

# 3) Train, score, select (round 0)
uv run opal run -c campaign.yaml --round 0 --resume

# 4) Inspect
uv run opal status -c campaign.yaml
uv run opal runs list -c campaign.yaml
uv run opal log -c campaign.yaml --round latest
uv run opal verify-outputs -c campaign.yaml --round latest

# 5) Plot
uv run opal plot -c campaign.yaml --quick
uv run opal plot -c campaign.yaml
```

**Notes:**
- Use `uv run opal ...` to ensure the correct environment.
- If `outputs/round_0/` already exists (this repo ships with demo outputs), `opal run` will refuse to overwrite
  unless you pass `--resume` or delete the existing artifacts first.

---

### Data used here

- **Local dataset**: `src/dnadesign/opal/campaigns/demo/records.parquet`
  - contains `sequence`, `mock__X_value`, and a placeholder label column.
- **Experimental labels**: `src/dnadesign/opal/campaigns/demo/inputs/r0/demo_y_sfxi.csv`
  - includes `intensity_log2_offset_delta`.

**8-vector label convention**

```
Y = [v00, v10, v01, v11, y00*, y10*, y01*, y11*]
```

The demo CSV includes **`intensity_log2_offset_delta`** (constant) so the
`SFXI` transform can enforce a strict match between data and objective params.

---

### Optional: swap to a USR dataset

If you want to exercise the same workflow against a USR dataset later, update
`campaign.yaml`:

```yaml
data:
  location: { kind: usr, path: ../../../usr/datasets, dataset: demo }
```

The rest of the demo stays the same.

---

### What to expect (captured outputs, trimmed)

#### `opal status` (after a run)

```
Campaign
  name           : Demo (vec8)
  slug           : demo
  workdir        : <repo>/src/dnadesign/opal/campaigns/demo
  X column       : mock__X_value
  Y column       : mock__y_label
  num_rounds     : 1

Latest round
  r              : 0
  run_id         : r0-<timestamp>
  n_train        : 6
  n_scored       : 9
  top_k requested: 5
  top_k effective: 5
  round_dir      : <repo>/src/dnadesign/opal/campaigns/demo/outputs/round_0
```

#### `opal runs list`

```
Runs
  - r=0, run_id=r0-<timestamp>, model=random_forest, objective=sfxi_v1,
    selection=top_n, n_train=6, n_scored=9
  - (more runs appear here if you re-run with --resume)
```

#### `opal log --round latest`

```
Round log
  round            : 0
  path             : <repo>/.../outputs/round_0/round.log.jsonl
  events           : <count>
  predict_batches  : <count>
  predict_rows     : <count>
  duration_total_s : <seconds>
  duration_fit_s   : <seconds>

Stages
  - done
  - fit
  - predict_batch
  - selection
  - yops_fit_transform
  - yops_inverse_done
```

---

### Where outputs go

**Per-round artifacts** (for audit + reuse):

```
outputs/round_<k>/
  model.joblib
  model_meta.json
  selection_top_k.csv
  selection_top_k.parquet
  selection_top_k__run_<run_id>.csv
  selection_top_k__run_<run_id>.parquet
  labels_used.parquet
  round_ctx.json
  objective_meta.json
  round.log.jsonl
```

**Ledger sinks (append-only)**:

```
outputs/ledger.runs.parquet
outputs/ledger.labels.parquet
outputs/ledger.predictions/part-*.parquet
```

---

### Interactive notebook (marimo)

Generate a campaign-tied notebook and open it in marimo:

```bash
uv run opal notebook generate -c campaign.yaml --round latest
uv run opal notebook run -c campaign.yaml
```

The notebook loads ledger artifacts (runs, predictions, labels) and gives you
interactive filtering and plots for the selected run.

Canonical vs cache vs transient (notebook):

- **Canonical (ledger)**: append-only, run-aware sources under `outputs/ledger.*`.
- **Cache (records)**: `latest_pred_*` columns in `records.parquet` (convenience only).
- **Overlay (notebook)**: in-memory overlays (ephemeral) for exploration, never persisted.
- **Y-ops gating**: SFXI scoring runs only when predictions are in objective space (Y-ops inverse applied).

---

### Demo `campaign.yaml`

```yaml
# OPAL demo campaign configuration (local)

campaign:
  name: "Demo (vec8)"
  slug: "demo"
  workdir: "."  # resolved relative to this file

data:
  location: { kind: local, path: records.parquet }
  x_column_name: "mock__X_value"
  y_column_name: "mock__y_label"
  y_expected_length: 8

ingest:
  duplicate_policy: "error"

training:
  policy:
    cumulative_training: true
    label_cross_round_deduplication_policy: "latest_only"
    allow_resuggesting_candidates_until_labeled: true
  y_ops:
    - name: intensity_median_iqr
      params:
        min_labels: 5
        center: median
        scale: iqr
        eps: 1e-8

transforms_x: { name: identity, params: {} }

transforms_y:
  name: sfxi_vec8_from_table_v1
  params:
    sequence_column: sequence
    logic_columns: ["v00","v10","v01","v11"]
    intensity_columns: ["y00_star","y10_star","y01_star","y11_star"]
    enforce_log2_offset_match: true
    expected_log2_offset_delta: 0.0

model:
  name: "random_forest"
  params:
    n_estimators: 100
    criterion: "friedman_mse"
    bootstrap: true
    oob_score: true
    random_state: 7
    n_jobs: -1
    emit_feature_importance: true

objective:
  name: "sfxi_v1"
  params:
    setpoint_vector: [0, 0, 0, 1]
    logic_exponent_beta: 1.0
    intensity_exponent_gamma: 1.0
    intensity_log2_offset_delta: 0.0
    scaling: { percentile: 95, min_n: 5, eps: 1.0e-8 }

selection:
  name: "top_n"
  params:
    top_k: 5
    tie_handling: "competition_rank"
    objective_mode: "maximize"

scoring:
  score_batch_size: 1000

safety:
  fail_on_mixed_biotype_or_alphabet: true
  require_biotype_and_alphabet_on_init: true
  conflict_policy_on_duplicate_ids: "error"
  write_back_requires_columns_present: true
  accept_x_mismatch: false

metadata:
  notes: "Demo campaign for OPAL using SFXI (vec8) and identity X."

plot_config: plots.yaml
```

---

### Troubleshooting

- **Delta mismatch** during `ingest-y`:
  - Ensure `intensity_log2_offset_delta` in the CSV equals
    `objective.params.intensity_log2_offset_delta` in `campaign.yaml`.
- **Matplotlib cache warning**:
  - OPAL sets a writable Matplotlib cache dir automatically during `opal plot`.
    If you explicitly set `MPLCONFIGDIR` and it points to an unwritable path,
    unset it or point it at a writable directory.
- **Old outputs layout**:
  - Remove `outputs/` and `state.json`, then re-run `opal init`.

---

@e-south
