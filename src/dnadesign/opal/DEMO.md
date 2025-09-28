## Demo Campaign — SFXI (setpoint × intensity)

This demo runs the full OPAL pipeline on mock **X** and **Y** while exercising real plugins:

- Y-ingest: `sfxi_vec8_from_table_v1`
- Objective: `sfxi_v1` (setpoint fidelity × intensity)
- Model: Random Forest

Read [**here**](../../src/objectives/DOCS/setpoint_fidelity_x_intensity.md) for more details on `sfxi` (optional).

### What this demo does

1. **Ingest** tidy CSV containing logic and intensity (log2*) columns → build an 8-vector label per sequence.  
2. **Train** a Random Forest regressor on your chosen X and these 8-vector labels.  
3. **Predict** Ŷ for the unlabeled candidate pool.  
4. **Score** each candidate with the SFXI-derived scalar.
5. **Rank & select** top-k by that scalar, write minimal per-round columns, and snapshot predictions to history.

---

### Data used here

- **USR dataset**: `usr/datasets/demo/records.parquet`  
  Contains `sequence`, `mock__X_value`, and `mock__y_label` prepared for SFXI.
- **Experimental Y labels (SFXI)**: `usr/demo_material/demo_y_sfxi.csv`  


**8-vector label convention** in this demo:

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
opal ingest-y -c campaign.yaml --round 0 --csv inputs/r0/demo_y_sfxi.csv

# 4) Train, score, select for round 0
opal run -c campaign.yaml -r 0
````

Artifacts will appear in `outputs/round_0/`: 

* `model.joblib`
* `selection_top_k.csv`
* `round_ctx.json`
* `objective_meta.json`
* `round.log.jsonl`
* and an append-only table at `outputs/events.parquet`.

Per-round write-backs to `records.parquet`:

* `opal__<slug>__latest_round` (int)
* `opal__<slug>__latest_score` (float map per id)


Inspect:

```bash
opal status
opal record-show -c campaign.yaml --sequence ACCTG...
opal explain     -c campaign.yaml --round 1
```

---

### IDs: how they’re resolved here

If the label CSV has **no** `id`, OPAL:

1. Matches by `sequence` to existing rows and appends a new round label,
2. **Fails fast** if that sequence already has labels for this campaign,
3. Creates a new row (essentials + generated `id`) if the sequence is new.

---

### What SFXI does here (scoring at selection time)

1. **Convert intensities to linear:**
   `y_lin = max(0, 2**(y*) − δ)` (δ is a log2* offset; default 0)
2. **Setpoint weighting:**
   Normalize `setpoint_vector` to get weights `w = p / sum(p)` (or zeros if `sum(p)=0`).
3. **Raw effect:**
   `E_raw = Σ_i w_i * y_lin_i`
4. **Round-specific scaling denominator:**
   Compute the **p-th percentile** (default p=95; fallback=75; min_n=5; eps) of `E_raw` **on this round’s labeled designs** under the same setpoint.
5. **Scaled effect:**
   `E_scaled = clip01(E_raw / denom)`
6. **Logic fidelity (0..1):**
   `F_logic = 1 − ||v̂ − p||₂ / D` where `D = max_corner_distance(setpoint)`
7. **Final SFXI score:**
   `score = (F_logic)^β * (E_scaled)^γ` (β=γ=1 by default)

**Diagnostics** persisted per candidate:

* `logic_fidelity_l2_norm01`, `effect_scaled`
* `selection_score` (the SFXI scalar)
* `uncertainty_mean_all_std` (RF per-tree std averaged across targets)

---

### Demo `campaign.yaml`

```yaml
campaign:
  name: "Demo (vec8)"
  slug: "demo"
  workdir: "/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/opal/campaigns/demo"

data:
  # USR demo dataset included in the repo
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
# Plugins (registry-first)
# --------------------------

# X transform (raw -> model-ready X)
transforms_x: { name: identity, params: {} }

# Y ingest (tidy -> vec8 Y)
transforms_y:
  name: sfxi_vec8_from_table_v1
  params:
    sequence_column: sequence
    logic_columns: ["v00","v10","v01","v11"]
    intensity_columns: ["y00_star","y10_star","y01_star","y11_star"]

models:
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
  target_scaler:
    enable: true
    minimum_labels_required: 5
    center_statistic: "median"
    scale_statistic: "iqr"

objectives:
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
    top_k_default: 5
    tie_handling: "competition_rank"
    objective: "maximize"

# Scoring performance (batch sizing)
scoring:
  score_batch_size: 10000
```

---

### Why the SFXI denominator is per-round

The per-round percentile scaling makes brightness comparable **within a round** and robust to day-to-day drift. OPAL snapshots the exact denominator and label pool in `round_ctx.json` and `objective_meta.json` so scores are auditable and reproducible later.
