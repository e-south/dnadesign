## Score-driven OPAL rounds (GP + SFXI + Top-n)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-13


This workflow changes the surrogate model from `random_forest` to
`gaussian_process` while retaining deterministic `top_n` selection. It
isolates model behavior from acquisition-policy behavior.

Campaign: `src/dnadesign/opal/campaigns/demo_gp_topn/`

**Reference docs:**

* [Configuration](../reference/configuration.md)
* [Gaussian Process behavior and math](../plugins/models/gaussian-process.md)
* [SFXI behavior and math](../plugins/objectives/sfxi.md)
* [Selection plugins](../plugins/selection/README.md)
* [CLI reference](../reference/cli.md)

**Outcome**

* Run a full round with GP predictions.
* Locate GP uncertainty even when selection ignores it.
* Apply the deterministic workflow's verification and audit contracts.

---

### Configuration difference from the RF baseline

The key difference is the `model` block; SFXI + `top_n` are unchanged.

```yaml
model:                              # Surrogate model used for candidate prediction
  name: gaussian_process            # Probabilistic regressor with predictive uncertainty
  params:                           # GP hyperparameters
    alpha: 1.0e-6                   # Observation-noise regularization term
    normalize_y: true               # Normalize targets before GP fit
    n_restarts_optimizer: 2         # Kernel optimizer restart count
    kernel:                         # Kernel family and shape parameters
      name: matern                  # Matern kernel for smoothness control
      length_scale: 0.5             # Characteristic input distance scale
      nu: 1.5                       # Matern smoothness parameter
      with_white_noise: true        # Add WhiteKernel noise component

selection_views:                       # Declare target-specific scoring and selection views
  - id: primary                        # Give the only view a stable public identifier
    objective: {name: sfxi_v1, params: {...}} # Emit the local SFXI score channel
    selection:                          # Configure this view's selector
      name: top_n                       # Use deterministic greedy ranking
      params:                           # Bind top-N to the objective channel
        top_k: 5                        # Request five candidates
        score_ref: sfxi                 # Use the local SFXI score channel
        objective_mode: maximize        # Prefer larger SFXI values
        tie_handling: competition_rank  # Preserve score ties
```

Runtime behavior:

* GP produces predictive uncertainty (`sigma`) internally.
* `top_n` ranks only by `score_ref`, so uncertainty does not affect which candidates are selected.

### Round 0 end-to-end

#### 1. Prepare the workspace

```bash
# Enter the GP Top-N demo campaign directory.
cd src/dnadesign/opal/campaigns/demo_gp_topn
# Copy the shared demo design-space records into this campaign.
cp ../demo_rf_sfxi_topn/records.parquet ./records.parquet

# Reset generated outputs and state for a fresh demo run.
uv run opal campaign-reset -c configs/campaign.yaml --apply --no-backup
# Initialize campaign state and workspace outputs.
uv run opal init     -c configs/campaign.yaml
# Validate config, plugin wiring, and core data contracts.
uv run opal validate -c configs/campaign.yaml
```

#### 2. Ingest labels (observed round 0)

```bash
# Ingest measured labels and stamp them as observed in round 0.
uv run opal ingest-y -c configs/campaign.yaml --round 0 \
  --csv inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --apply
```

#### 3. Run round 0 (labels visible through round 0)

```bash
# Train, score, and select candidates using labels visible through round 0.
uv run opal run -c configs/campaign.yaml --round 0
```
> `ingest-y --round R` records the measurement round. `run --round R` uses
> labels observed through round `R`.

Checkpoint outputs:

* `outputs/rounds/round_0/selection/selections.parquet`
* `outputs/rounds/round_0/selection/selection_batch.parquet`
* `outputs/ledger/runs.parquet`
* `outputs/ledger/predictions/`

#### 4. Verify + inspect

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

#### 5. Optional: confirm uncertainty exists (even though selection ignores it)

Inspect a selected record and look for the selected score/uncertainty fields.

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
```

### Continue to round 1, etc.

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

* sklearn GP `ConvergenceWarning`: common on small demo data; treat as informational if `validate` and `verify-outputs` pass.
* `SFXI min_n` failure: ingest enough labels for the round supplied to `opal run`.
* Unknown sequences during ingest: ensure input IDs exist in `records.parquet` (or keep `--unknown-sequences drop`).
