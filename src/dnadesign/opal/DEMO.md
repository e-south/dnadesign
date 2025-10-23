## Demo Campaign — SFXI (setpoint × intensity)

This demo runs the full OPAL pipeline on mock **X** and **Y** values while exercising real plugins:

- Y-ingest: `sfxi_vec8_from_table_v1`
- Objective: `sfxi_v1` (setpoint fidelity × intensity)
- Model: Random Forest

See the SFXI objective details [**here**](./src/objectives/DOCS/setpoint_fidelity_x_intensity.md) (optional).

---

### What this demo does

1. **Ingest** a tidy CSV containing logic and intensity (log2*) columns → build an 8-vector label per sequence.  
2. **Train** a Random Forest regressor on your chosen X and these 8-vector labels.  
3. **Predict** Ŷ for the unlabeled candidate pool.  
4. **Score** each candidate with the SFXI-derived scalar.  
5. **Rank & select** top-k by that scalar, write per-round artifacts, and append canonical events.

---

### Data used here

- **USR dataset**: `usr/datasets/demo/records.parquet`  
  Contains `sequence`, `mock__X_value`, and a placeholder label column.
- **Experimental Y labels (SFXI)**: `usr/demo_material/demo_y_sfxi.csv`

**8-vector label convention**:

````
Y = [v00, v10, v01, v11, y00*, y10*, y01*, y11*]
````

---

### Run the demo

From the demo workdir:

```bash
cd ../campaigns/demo/

# 1) Initialize & validate
opal init     -c campaign.yaml
opal validate -c campaign.yaml

# 2) Provide labels for round 0
mkdir -p inputs/r0/
cp ../../../usr/demo_material/demo_y_sfxi.csv inputs/r0/

# 3) Ingest round 0 labels
opal ingest-y -c campaign.yaml --observed-round 0 --csv inputs/r0/r0_y_sfxi.csv

# 4) Train, score, select for round 0
opal run -c campaign.yaml --labels-as-of 0

# 5) Lists objective‑level info per round for row‑level diagnostics
opal objective-meta -c campaign.yaml --round latest
````

Artifacts appear in `outputs/round_0/`:

* `model.joblib`
* `selection_top_k.csv`
* `round_ctx.json`
* `objective_meta.json`
* `round.log.jsonl`

Canonical events are appended to **`outputs/events.parquet`**:

* `label` — rows emitted by `ingest-y`
* `run_pred` — one per candidate scored
* `run_meta` — one per run with config and artifact checksums

**Ergonomic caches** written to `records.parquet`:

* `opal__<slug>__latest_as_of_round`
* `opal__<slug>__latest_pred_scalar`

Inspect:

```bash
opal status
opal record-show -c campaign.yaml --sequence ACCTG...
opal explain     -c campaign.yaml --round 1
```

---

### IDs: how they’re resolved here

If the label CSV has **no** `id`, OPAL will:

1. Match by `sequence` to existing rows and append a new round label.
2. **Fail fast** if that sequence already has labels for this campaign (immutability).
3. Create a new row (essentials + generated `id`) if the sequence is new.

---

### Demo `campaign.yaml` (local)

```yaml
# OPAL demo campaign configuration (local)

campaign:
  name: "Demo (vec8)"
  slug: "demo"
  workdir: "/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/opal/campaigns/demo"

data:
  # Use the "demo" usr records.parquet file included in the repo.
  location: { kind: usr, path: ../../../usr/datasets, dataset: demo }
  x_column_name: "mock__X_value"
  y_column_name: "mock__y_label"
  y_expected_length: 8

metadata:
  notes: "Demo campaign for OPAL using SFXI (vec8) and identity X."

safety:
  fail_on_mixed_biotype_or_alphabet: true
  require_biotype_and_alphabet_on_init: true
  conflict_policy_on_duplicate_ids: "error"
  write_back_requires_columns_present: true
  accept_x_mismatch: false

# --------------------------
# Plugin blocks (registry-first)
# --------------------------

# X transform (raw -> model-ready X)
transforms_x: { name: identity, params: {} }

# Y ingestion transform (raw -> model-ready y)
transforms_y:
  name: sfxi_vec8_from_table_v1
  params:
    sequence_column: sequence
    logic_columns: ["v00","v10","v01","v11"]
    intensity_columns: ["y00_star","y10_star","y01_star","y11_star"]

model:
  name: "random_forest"
  params:
    n_estimators: 100
    criterion: "friedman_mse"
    bootstrap: true
    oob_score: true
    random_state: 7
    n_jobs: -1

training:
  policy:
    cumulative_training: true
    label_cross_round_deduplication_policy: "latest_only"
    allow_resuggesting_candidates_until_labeled: true
  # Ephemeral Y-ops used at fit-time and inverted at predict-time
  y_ops:
    - name: intensity_median_iqr     # scales indices 4:8 only; no log2 changes here
      params:
        min_labels: 5
        center: median
        scale: iqr
        eps: 1e-8

objective:
  name: "sfxi_v1"
  params:
    setpoint_vector: [0, 0, 0, 1]
    logic_exponent_beta: 1.0
    intensity_exponent_gamma: 1.0
    intensity_log2_offset_delta: 0.0
    scaling: { percentile: 95, min_n: 5, fallback_p: 75, eps: 1.0e-8 }

selection:
  name: "top_n"
  params:
    top_k: 5
    tie_handling: "competition_rank"
    objective_mode: "maximize"

scoring:
  score_batch_size: 1000
```

> Next: read about command surface in the **[CLI Manual](./src/cli/README.md)**, or jump back to the **[Top-level README](./README.md)**.

---

@e-south