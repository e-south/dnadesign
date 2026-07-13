## Deterministic OPAL rounds (RF + SFXI + Top-n)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-13


This workflow ingests round-0 labels, fits a `random_forest` model, scores
candidates with `sfxi_v1`, selects candidates with `top_n`, and verifies the
resulting audit trail.

Campaign: `src/dnadesign/opal/campaigns/demo_rf_sfxi_topn/`

**Reference docs:**

- [Configuration](../reference/configuration.md)
- [SFXI behavior and math](../plugins/objectives/sfxi.md)
- [Selection plugins](../plugins/selection/README.md)
- [CLI reference](../reference/cli.md)
- [Architecture and data flow](../concepts/architecture.md)
- [RoundCtx and contract auditing](../concepts/roundctx.md)

**Outcome**

- Run a complete OPAL round from a clean workspace.
- Inspect how `configs/campaign.yaml` controls transforms, model, objective,
  and selection.
- Locate round artifacts and append-only ledgers.
- Exercise the inspection surface (`status`, `runs list`, `ctx audit`,
  `record-show`, plus optional `predict` and `plot`).

**Prerequisites and assumptions**

- Run commands from the repository root; listed paths are root-relative.
- OPAL is runnable as `uv run opal ...` in this repo.
- Demo inputs exist:
  - design space: `campaigns/demo_rf_sfxi_topn/records.parquet`
  - labels: `campaigns/demo_rf_sfxi_topn/inputs/r0/vec8-b0.xlsx`

**Expected outputs**

After a successful round-0 run:

- Round artifacts: `outputs/rounds/round_0/...`
  - selected rows: `outputs/rounds/round_0/selection/selections.parquet`
  - logical union: `outputs/rounds/round_0/selection/selection_batch.parquet`
- Append-only ledgers:
  - labels: `outputs/ledger/labels.parquet`
  - predictions: `outputs/ledger/predictions/`
  - runs: `outputs/ledger/runs.parquet`

> `opal campaign-reset --apply` deletes generated state and outputs for the campaign. It is intended for demo resets.

---

### Configuration surface

This demo is “config-driven on purpose”: changing these blocks changes behavior without changing runtime code.

```yaml
training:                           # Training-time target transforms and policies
  y_ops:                            # Per-round Y operations fit on labels, then inverted before objectives
    - name: intensity_median_iqr    # Robustly center/scale intensity targets.
      params:                       # Y-op hyperparameters
        min_labels: 5               # Require enough labels before enabling this transform
        center: median              # Robust center statistic
        scale: iqr                  # Robust spread statistic
        eps: 1e-8                   # Numerical floor for near-zero spread

model:                              # Surrogate model used for candidate prediction
  name: random_forest               # Deterministic tree ensemble baseline
  params:                           # Model hyperparameters
    n_estimators: 100               # Number of trees
    random_state: 7                 # Seed for reproducibility
    n_jobs: -1                      # Use all available CPU cores
    emit_feature_importance: true   # Persist feature-importance artifact

selection_views:                       # Declare target-specific scoring and selection views
  - id: primary                        # Give the only view a stable public identifier
    objective:                         # Configure this view's objective plugin
      name: sfxi_v1                    # Score canonical SFXI v1
      params:                          # Declare the SFXI target and scaling contract
        setpoint_vector: [0, 0, 0, 1]  # Target the AND response pattern
        logic_exponent_beta: 1.0       # Weight logic fidelity linearly
        intensity_exponent_gamma: 1.0  # Weight scaled intensity linearly
        intensity_log2_offset_delta: 0.0 # Require matching intensity offset semantics
        scaling: { percentile: 95, min_n: 5, eps: 1.0e-8 } # Calibrate intensity scaling
    selection:                         # Configure this view's selector
      name: top_n                      # Use deterministic greedy ranking
      params:                          # Bind top-N to the objective channel
        top_k: 5                       # Request five candidates
        score_ref: sfxi                # Use the local SFXI score channel
        objective_mode: maximize       # Prefer larger SFXI values
        tie_handling: competition_rank # Preserve score ties

selection_batch:                       # Build the logical union of all view selections
  deduplicate_by: id                   # Keep one logical row per candidate ID
```

How this maps to runtime:

* `training.y_ops` is applied at fit time and inverted before scoring so predictions are in objective units.
* `model` produces predicted vec8 per candidate.
* The `primary` objective converts vec8 to scalar channel `sfxi` plus diagnostics.
* The `primary` selector ranks by its configured local `score_ref`.

---


### Round 0 end-to-end

#### 1. Prepare a clean workspace

```bash
# Enter the demo campaign directory.
cd src/dnadesign/opal/campaigns/demo_rf_sfxi_topn
# Copy the shared demo design-space records into this campaign.
test -f records.parquet

# Reset generated outputs and state for a fresh demo run.
uv run opal campaign-reset -c configs/campaign.yaml --apply --no-backup
# Initialize campaign state and workspace outputs.
uv run opal init     -c configs/campaign.yaml
# Validate config, plugin wiring, and core data contracts.
uv run opal validate -c configs/campaign.yaml
```

Checkpoint:

* `state.json` exists.
* `validate` returns `OK: validation passed`.

#### 2. (Optional) Ask OPAL what it expects next

These helpers are read-only; they surface the typical next step.

```bash
# Render a config-specific runbook for this campaign.
uv run opal guide -c configs/campaign.yaml --format markdown
# Ask OPAL for the next recommended step at labels-as-of round 0.
uv run opal guide next -c configs/campaign.yaml --round 0
# Preview what a round-0 run will require and emit.
uv run opal explain -c configs/campaign.yaml --round 0
```

#### 3. Ingest observed labels for round 0

```bash
# Ingest measured labels and stamp them as observed in round 0.
uv run opal ingest-y -c configs/campaign.yaml --round 0 \
  --csv inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --apply
```
> `ingest-y --round R` records when a label was measured. `run --round R`
> uses labels observed through round `R`.

Checkpoint:

* label events appended to `outputs/ledger/labels.parquet`
* record-level label history updated in `records.parquet`

#### 4. Train, score, and select (round 0)

```bash
# Train, score, and select candidates using labels visible through round 0.
uv run opal run -c configs/campaign.yaml --round 0
```

Checkpoint:

* `outputs/rounds/round_0/selection/selections.parquet`
* `outputs/rounds/round_0/selection/selection_batch.parquet`
* `outputs/ledger/runs.parquet`
* `outputs/ledger/predictions/`

#### 5. Verify and inspect the run

```bash
# Check selection and ledger consistency for the latest round.
uv run opal verify-outputs -c configs/campaign.yaml --view primary --round latest
# Print campaign status and latest round pointers.
uv run opal status   -c configs/campaign.yaml
# List recorded runs for this campaign.
uv run opal runs list -c configs/campaign.yaml
# Audit RoundCtx contract payloads for the latest round.
uv run opal ctx audit -c configs/campaign.yaml --round latest
```

Expected result: `verify-outputs` reports `mismatches: 0`.

Preview the selection through its public contract:

```bash
# Show the primary view's resolved selected rows.
uv run opal selection-set show -c configs/campaign.yaml --view primary --round latest
```

Inspect a selected record:

```bash
# Show the top selected record (competition rank 1) from the latest round.
uv run opal record-show -c configs/campaign.yaml --view primary --selected-rank 1 --round latest --run-id latest
```

#### 6. Optional read-only analysis and plots

```bash
# Export round-level predictions for downstream analysis.
uv run opal predict -c configs/campaign.yaml --round latest --out outputs/predict_r0.parquet
# Render the score-vs-rank plot for the latest round.
uv run opal plot   -c configs/campaign.yaml --view primary --name score_vs_rank_latest --round latest
# Render feature-importance bars from the RF model artifact.
uv run opal plot   -c configs/campaign.yaml --view primary --name feature_importance_bars_latest --round latest
```

> If a plot name is unavailable, run `uv run opal plot --help` and inspect
> `configs/plots.yaml`.

---

### Continue to round 1, etc.

`sfxi_v1` uses within-round label statistics for scaling. Each `run --round R`
expects enough labels observed in round `R` (see `scaling.min_n`).

```bash
# Ingest the next batch and stamp labels as observed in round 1.
uv run opal ingest-y -c configs/campaign.yaml --round 1 \
  --csv inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --apply

# Re-run training/selection with labels visible through round 1.
uv run opal run -c configs/campaign.yaml --round 1 --resume
# Re-check ledger and artifact consistency after the resume run.
uv run opal verify-outputs -c configs/campaign.yaml --view primary --round latest
```

---

### If a step fails

* `SFXI min_n` failure: ingest more labels for the round supplied to `opal run`.
* Unknown sequences during ingest: keep `--unknown-sequences drop` for demo data or ensure input IDs match `records.parquet`.
* `verify-outputs` mismatch: rerun `opal run ... --resume`, then inspect
  `selections.parquet` and the named view in `outputs/ledger/predictions/`.
