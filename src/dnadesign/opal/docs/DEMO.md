## OPAL Demo Campaign -- SFXI (setpoint x intensity)

This demo walks a **complete OPAL loop** on a small dataset with the SFXI ingest + plotting stack.

**What you'll learn (and see):**
- a full `run` (train -> score -> select)
- ledger-backed inspection (`status`, `runs`, `log`)
- plots (quick + configured)
- where artifacts and ledgers live

This demo is **self-contained**: it ships with a local `records.parquet` and label inputs under `inputs/`, so no USR setup is required.

---

### TL;DR

From the repo root:

```bash
cd src/dnadesign/opal/campaigns/demo/

# 1) Initialize & validate
uv run opal init     -c configs/campaign.yaml
uv run opal validate -c configs/campaign.yaml

# 2) Ingest round-0 labels
uv run opal ingest-y -c configs/campaign.yaml --round 0 \
  --csv inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop

# 3) Train, score, select (round 0)
uv run opal run -c configs/campaign.yaml --round 0

# 4) Inspect
uv run opal status -c configs/campaign.yaml
uv run opal runs list -c configs/campaign.yaml
uv run opal log -c configs/campaign.yaml --round latest
uv run opal verify-outputs -c configs/campaign.yaml --round latest

# 5) Plot
uv run opal plot -c configs/campaign.yaml --quick
uv run opal plot -c configs/campaign.yaml
```

**Notes:**
- Use `uv run opal ...` to ensure the correct environment.
- If `outputs/rounds/round_0/` already exists from a prior run, `opal run` will refuse to overwrite
  unless you pass `--resume` (which wipes the round directory) or delete the existing artifacts first.

---

### Clean-slate reset (demo)

From the campaign root:

```bash
# Preferred: reset campaign state + remove OPAL columns
uv run opal campaign-reset -c configs/campaign.yaml --yes --no-backup

# Manual fallback (if you want to keep outputs/)
uv run opal prune-source -c configs/campaign.yaml --scope any --yes --no-backup
```

---

### Data used here

- **Local dataset**: `src/dnadesign/opal/campaigns/demo/records.parquet`
  - contains `sequence` and `infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean` (X).
  - the Y column (`sfxi_8_vector_y_label`) is added during label ingestion.
- **Experimental labels**: `src/dnadesign/opal/campaigns/demo/inputs/r0/vec8-b0.xlsx`
  - includes `intensity_log2_offset_delta`.

**8-vector label convention**

```
Y = [v00, v10, v01, v11, y00*, y10*, y01*, y11*]
```

The demo XLSX includes **`intensity_log2_offset_delta`** (constant) so the
`SFXI` transform can enforce a strict match between data and objective params.

---

### Optional: swap to a USR dataset

If you want to exercise the same workflow against a USR dataset later, update
`configs/campaign.yaml`:

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
  X column       : infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean
  Y column       : sfxi_8_vector_y_label
  num_rounds     : 1

Latest round
  r              : 0
  run_id         : r0-<timestamp>
  n_train        : 6
  n_scored       : 9
  top_k requested: 5
  top_k effective: 5
  round_dir      : <repo>/src/dnadesign/opal/campaigns/demo/outputs/rounds/round_0
```

#### `opal runs list`

```
Runs
  - r=0, run_id=r0-<timestamp>, model=random_forest, objective=sfxi_v1,
    selection=top_n, n_train=6, n_scored=9
  - (more runs appear here if you re-run with --resume; the round dir is wiped before the re-run)
```

#### `opal log --round latest`

```
Round log
  round            : 0
  path             : <repo>/.../outputs/rounds/round_0/round.log.jsonl
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
outputs/rounds/round_<k>/
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
outputs/ledger/runs.parquet
outputs/ledger/labels.parquet
outputs/ledger/predictions/part-*.parquet
```

---

### Interactive notebook (marimo)

Generate a campaign-tied notebook and open it in marimo:

```bash
uv run opal notebook generate -c configs/campaign.yaml --round latest
uv run opal notebook run -c configs/campaign.yaml
```

The notebook loads campaign artifacts and label history from `records.parquet`,
then gives you interactive filtering and plots for the selected run.

Canonical vs ledger vs overlay (notebook):

- **Canonical (dashboard)**: `records.parquet` label history (`opal__<slug>__label_hist`) plus campaign artifacts/state.
- **Ledger (audit)**: append-only run metadata and predictions under `outputs/ledger/` (optional for audit).
- **Overlay (notebook)**: in-memory rescoring from stored predictions for exploration only; never persisted.
- **Y-ops gating**: SFXI scoring runs only when predictions are in objective space (Y-ops inverse applied).

---

### Demo config (`configs/campaign.yaml`)

```yaml
# OPAL demo campaign configuration (local)

campaign:
  name: "Demo (vec8)"
  slug: "demo"
  workdir: "."  # resolved relative to the campaign root

data:
  location: { kind: local, path: records.parquet }
  x_column_name: "infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean"
  y_column_name: "sfxi_8_vector_y_label"
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
    `objective.params.intensity_log2_offset_delta` in `configs/campaign.yaml`.
- **Matplotlib cache warning**:
  - OPAL sets a writable Matplotlib cache dir automatically during `opal plot`.
    If you explicitly set `MPLCONFIGDIR` and it points to an unwritable path,
    unset it or point it at a writable directory.
- **Old outputs layout**:
  - Remove `outputs/` and `state.json`, then re-run `opal init` (or use the demo reset command below).

---

### Campaign reset command (hidden)

`campaign-reset` is a hidden helper (not listed in `opal --help`) that wipes the demo state:
it prunes OPAL columns from `records.parquet`, deletes `outputs/`, and removes `state.json`.

From the demo campaign root:

```bash
uv run opal campaign-reset -c configs/campaign.yaml

# Non-interactive (skip prompt)
uv run opal campaign-reset -c configs/campaign.yaml --yes
```

For non-demo campaigns, you must pass `--allow-non-demo` and confirm the slug.

---

@e-south
