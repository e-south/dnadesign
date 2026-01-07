## OPAL Demo Campaign -- SFXI (setpoint x intensity)

This demo walks a **complete OPAL loop** on a small dataset with the modern, strict
SFXI ingest + plotting stack. It is designed to be copy/pasteable, deterministic,
and easy to extend.

**What you'll learn (and see):**
- strict label ingestion with delta checks
- a full `run` (train -> score -> select)
- ledger-backed inspection (`status`, `runs`, `log`)
- plots (quick + configured)
- where artifacts and ledgers live

This demo is **self-contained**: it ships with a local `records.parquet` and
label CSVs under `inputs/`, so no USR setup is required.

---

### TL;DR (happy path)

From the repo root:

```bash
cd src/dnadesign/opal/campaigns/demo/

# (Optional) suppress PyArrow sysctl warnings in non-TTY contexts
export OPAL_SUPPRESS_PYARROW_SYSCTL=1

# 1) Initialize & validate
uv run opal init     -c campaign.yaml
uv run opal validate -c campaign.yaml

# 2) Ingest round-0 labels
uv run opal ingest-y -c campaign.yaml --round 0 \
  --csv inputs/r0/demo_y_sfxi.csv

# 3) Train, score, select (round 0)
uv run opal run -c campaign.yaml --round 0

# 4) Inspect
uv run opal status -c campaign.yaml
uv run opal runs list -c campaign.yaml
uv run opal log -c campaign.yaml --round latest

# 5) Plot
MPLCONFIGDIR=$PWD/.tmp/mpl OPAL_SUPPRESS_PYARROW_SYSCTL=1 \
  uv run opal plot -c campaign.yaml --quick
MPLCONFIGDIR=$PWD/.tmp/mpl OPAL_SUPPRESS_PYARROW_SYSCTL=1 \
  uv run opal plot -c campaign.yaml
```

**Notes:**
- Use `uv run opal ...` to ensure the correct environment.
- On macOS, `OPAL_SUPPRESS_PYARROW_SYSCTL=1` suppresses PyArrow sysctl warnings (helpful in CI/non-TTY runs).
- `MPLCONFIGDIR` avoids Matplotlib cache warnings and speeds imports.

---

### Data used here (demo)

- **Local dataset**: `src/dnadesign/opal/campaigns/demo/records.parquet`
  - contains `sequence`, `mock__X_value`, and a placeholder label column.
- **Experimental labels**: `src/dnadesign/opal/campaigns/demo/inputs/r0/demo_y_sfxi.csv`
  - includes `intensity_log2_offset_delta` (strict delta match).

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
  run_id         : r0-2026-01-02T18:48:11+00:00
  n_train        : 9
  n_scored       : 15
  top_k requested: 5
  top_k effective: 5
  round_dir      : <repo>/src/dnadesign/opal/campaigns/demo/outputs/round_0
```

#### `opal runs list`

```
Runs
  - r=0, run_id=r0-2026-01-01T23:..., model=random_forest, objective=sfxi_v1,
    selection=top_n, n_train=9, n_scored=15
  - r=0, run_id=r0-2026-01-02T18:..., model=random_forest, objective=sfxi_v1,
    selection=top_n, n_train=9, n_scored=15
```

#### `opal log --round latest`

```
Round log
  round            : 0
  path             : <repo>/.../outputs/round_0/round.log.jsonl
  events           : 25
  predict_batches  : 2
  predict_rows     : 30
  duration_total_s : 68998.0
  duration_fit_s   : 68684.0

Stages
  - done
  - fit
  - predict_batch
  - selection
  - yops_fit_transform
  - yops_inverse_done
```

#### `opal plot --quick`

```
[ok] quick_score_vs_rank (scatter_score_vs_rank) -> .../outputs/plots/quick_score_vs_rank.png
[ok] quick_percent_high (percent_high_activity_over_rounds) -> .../outputs/plots/quick_percent_high.png
[ok] quick_sfxi_logic_fidelity (sfxi_logic_fidelity_closeness) -> .../outputs/plots/quick_sfxi_logic_fidelity.png
[ok] quick_fold_change_vs_logic_fidelity (fold_change_vs_logic_fidelity) -> .../outputs/plots/quick_fold_change_vs_logic_fidelity.png
[ok] quick_feature_importance (feature_importance_bars) -> .../outputs/plots/quick_feature_importance.png
```

#### `opal plot` (full config)

```
[ok] score_vs_rank_latest (scatter_score_vs_rank) -> .../outputs/plots/score_vs_rank_latest.png
[ok] percent_high_activity (percent_high_activity_over_rounds) -> .../outputs/plots/percent_high_activity.png
[ok] sfxi_logic_closeness (sfxi_logic_fidelity_closeness) -> .../outputs/plots/sfxi_logic_closeness.png
[ok] fold_change_vs_logic (fold_change_vs_logic_fidelity) -> .../outputs/plots/fold_change_vs_logic.png
```

---

### Plots: why `on_violin_invalid: line`

The SFXI closeness plot draws violins by default. The demo dataset is small, so
we **explicitly** set:

```yaml
on_violin_invalid: line
```

This makes the fallback intentional for the demo: for small sample sizes, the
plot uses a mean-line summary instead of failing. The **default** behavior is
now strict (`on_violin_invalid: error`), so production configs should choose
explicitly.

---

### Where outputs go

**Per-round artifacts** (for audit + reuse):

```
outputs/round_<k>/
  model.joblib
  model_meta.json
  selection_top_k.csv
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

---

### Demo `campaign.yaml` (current, canonical)

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
  - Set `MPLCONFIGDIR=$PWD/.tmp/mpl`.
- **Old outputs layout**:
  - Remove `outputs/` and `state.json`, then re-run `opal init`.

---

@e-south
