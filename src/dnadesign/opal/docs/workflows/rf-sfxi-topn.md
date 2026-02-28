## Deterministic OPAL rounds (RF + SFXI + Top-n)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This is the baseline OPAL “design-build-test-learn loop” walkthrough: ingest round-0 labels, fit a `random_forest` model, score candidates with `sfxi_v1`, select the next batch with `top_n`, and verify that the audit trail is consistent.

Campaign: `src/dnadesign/opal/campaigns/demo_rf_sfxi_topn/`

**Reference docs:**

- [Configuration](../reference/configuration.md)
- [SFXI behavior and math](../plugins/objective-sfxi.md)
- [Selection plugins](../plugins/selection.md)
- [CLI reference](../reference/cli.md)
- [Architecture and data flow](../concepts/architecture.md)
- [RoundCtx and contract auditing](../concepts/roundctx.md)

**What this doc is meant to accomplish**

- Run a complete OPAL round from a clean workspace.
- See how `configs/campaign.yaml` controls transforms, model, objective, and selection.
- Know where OPAL writes round artifacts and append-only ledgers.
- Exercise the “inspection” surface (`status`, `runs list`, `ctx audit`, `record-show`, plus optional `predict` + `plot`).

**Prerequisites and assumptions**

- Run commands from the repository root (paths in this doc assume that).
- OPAL is runnable as `uv run opal ...` in this repo.
- Demo inputs exist:
  - design space: `campaigns/demo/records.parquet`
  - labels: `campaigns/demo_rf_sfxi_topn/inputs/r0/vec8-b0.xlsx`

**Outputs you should expect**

After a successful round-0 run:

- Round artifacts: `outputs/rounds/round_0/...`
  - selection CSV: `outputs/rounds/round_0/selection/selection_top_k.csv`
- Append-only ledgers:
  - labels: `outputs/ledger/labels.parquet`
  - predictions: `outputs/ledger/predictions/`
  - runs: `outputs/ledger/runs.parquet`

> `opal campaign-reset --apply` deletes generated state and outputs for the campaign. Use it to reset demos.

---

### The knobs in the config that this demo is exercising

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

objectives:                             # Objective plugins that emit score/uncertainty channels
  - name: sfxi_v1                       # Setpoint fidelity × intensity scalar objective
    params:                             # SFXI objective parameters
      setpoint_vector: [0, 0, 0, 1]     # Target logic state order
      logic_exponent_beta: 1.0          # Weight on logic-fidelity term
      intensity_exponent_gamma: 1.0     # Weight on intensity-effect term
      intensity_log2_offset_delta: 0.0  # Log2 offset used when recovering linear intensity
      scaling: { percentile: 95, min_n: 5, eps: 1.0e-8 }  # Round-local effect scaling config

selection:                              # Selection strategy over objective channels
  name: top_n                           # Deterministic rank-by-score selector
  params:                               # Selection contract fields
    top_k: 5                            # Number of candidates to select
    score_ref: sfxi_v1/sfxi             # Objective score channel used for ranking
    objective_mode: maximize  # Sets `objective_mode` for this example configuration.
    tie_handling: competition_rank      # Tie policy for ranking output
```

How this maps to runtime:

* `training.y_ops` is applied at fit time and inverted before scoring so predictions are in objective units.
* `model` produces predicted vec8 per candidate.
* `objectives.sfxi_v1` converts vec8 → scalar score channel `sfxi` + diagnostics.
* `selection.top_n` ranks purely by the configured `score_ref`.

---


### Round 0 end-to-end

#### 1. Prepare a clean workspace

```bash
# Enter the demo campaign directory.
cd src/dnadesign/opal/campaigns/demo_rf_sfxi_topn
# Copy the shared demo design-space records into this campaign.
cp ../demo/records.parquet ./records.parquet

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
uv run opal guide next -c configs/campaign.yaml --labels-as-of 0
# Preview what a round-0 run will require and emit.
uv run opal explain -c configs/campaign.yaml --labels-as-of 0
```

#### 3. Ingest observed labels for round 0

```bash
# Ingest measured labels and stamp them as observed in round 0.
uv run opal ingest-y -c configs/campaign.yaml --observed-round 0 \
  --in inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --apply
```
> `--observed-round R` stamps when a label was *measured* (stored in the label ledger).
> `--labels-as-of R` controls which labels are *visible to training + selection* during `opal run`.
> For a clean loop: ingest labels with `--observed-round R`, then run with `--labels-as-of R`.

Checkpoint:

* label events appended to `outputs/ledger/labels.parquet`
* record-level label history updated in `records.parquet`

#### 4. Train, score, and select (round 0)

```bash
# Train, score, and select candidates using labels visible through round 0.
uv run opal run -c configs/campaign.yaml --labels-as-of 0
```

Checkpoint:

* `outputs/rounds/round_0/selection/selection_top_k.csv`
* `outputs/ledger/runs.parquet`
* `outputs/ledger/predictions/`

#### 5. Verify and inspect the run

```bash
# Check selection and ledger consistency for the latest round.
uv run opal verify-outputs -c configs/campaign.yaml --round latest
# Print campaign status and latest round pointers.
uv run opal status   -c configs/campaign.yaml
# List recorded runs for this campaign.
uv run opal runs list -c configs/campaign.yaml
# Audit RoundCtx contract payloads for the latest round.
uv run opal ctx audit -c configs/campaign.yaml --round latest
```

Expected result: `verify-outputs` reports `mismatches: 0`.

Preview the selection CSV:

```bash
# Preview selected candidates and ranking columns.
head -n 6 outputs/rounds/round_0/selection/selection_top_k.csv
```

Inspect a selected record:

```bash
# Show the top selected record (competition rank 1) from the latest round.
uv run opal record-show -c configs/campaign.yaml --selected-rank 1 --round latest --run-id latest
```

#### 6. Optional read-only analysis and plots

```bash
# Export round-level predictions for downstream analysis.
uv run opal predict -c configs/campaign.yaml --round latest --out outputs/predict_r0.parquet
# Render the score-vs-rank plot for the latest round.
uv run opal plot   -c configs/campaign.yaml --name score_vs_rank_latest --round latest
# Render feature-importance bars from the RF model artifact.
uv run opal plot   -c configs/campaign.yaml --name feature_importance_bars_latest --round latest
```

> If a plot name isn’t available in your build, run `uv run opal plot --help` and/or check your campaign’s `plot_presets` configuration.

---

### Continue to round 1, etc.

`sfxi_v1` uses within-round label statistics for scaling. Each `--labels-as-of R` run expects enough labels in observed round `R` (see `scaling.min_n`).

```bash
# Ingest the next batch and stamp labels as observed in round 1.
uv run opal ingest-y -c configs/campaign.yaml --observed-round 1 \
  --in inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --apply

# Re-run training/selection with labels visible through round 1.
uv run opal run -c configs/campaign.yaml --labels-as-of 1 --resume
# Re-check ledger and artifact consistency after the resume run.
uv run opal verify-outputs -c configs/campaign.yaml --round latest
```

---

### If a step fails

* `SFXI min_n` failure: ingest more labels for the same `--observed-round` you’re running as `--labels-as-of`.
* Unknown sequences during ingest: keep `--unknown-sequences drop` for demo data or ensure input IDs match `records.parquet`.
* `verify-outputs` mismatch: rerun `opal run ... --resume`, then compare `selection_top_k.csv` to the latest run row in `outputs/ledger/runs.parquet`.
