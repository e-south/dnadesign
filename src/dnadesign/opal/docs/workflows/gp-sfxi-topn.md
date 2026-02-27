## Score-driven OPAL rounds (GP + SFXI + Top-n)

This demo swaps the surrogate model from `random_forest` to `gaussian_process` but keeps deterministic `top_n` selection. It’s the “model changed, selection unchanged” bridge between the random-forest baseline and Expected Improvement.

Campaign: `src/dnadesign/opal/campaigns/demo_gp_topn/`

**Reference docs:**

* [Configuration](../reference/configuration.md)
* [Gaussian Process behavior and math](../plugins/model-gaussian-process.md)
* [SFXI behavior and math](../plugins/objective-sfxi.md)
* [Selection plugins](../plugins/selection.md)
* [CLI reference](../reference/cli.md)

**What this doc is meant to accomplish**

* Run a full round with GP predictions.
* See where GP uncertainty is stored even when selection ignores it.
* Keep the same verification and audit habits as the deterministic workflow.

---

### What changed from the RF baseline in the config

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

selection:                          # Selection strategy over objective channels
  name: top_n                       # Deterministic rank-by-score selector
  params:                           # Selection contract fields
    top_k: 5                        # Number of candidates to select
    score_ref: sfxi_v1/sfxi         # Objective score channel used for ranking
    objective_mode: maximize  # Sets `objective_mode` for this example configuration.
    tie_handling: competition_rank  # Tie policy for ranking output
```

What to expect at runtime:

* GP produces predictive uncertainty (`sigma`) internally.
* `top_n` ranks only by `score_ref`, so uncertainty does not affect which candidates are selected.

### Round 0 end-to-end

#### 1. Prepare the workspace

```bash
# Enter the GP Top-N demo campaign directory.
cd src/dnadesign/opal/campaigns/demo_gp_topn
# Copy the shared demo design-space records into this campaign.
cp ../demo/records.parquet ./records.parquet

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
uv run opal ingest-y -c configs/campaign.yaml --observed-round 0 \
  --in inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --apply
```

#### 3. Run round 0 (labels visible through round 0)

```bash
# Train, score, and select candidates using labels visible through round 0.
uv run opal run -c configs/campaign.yaml --labels-as-of 0
```
> * `--observed-round R`: measurement stamp for ingest.
> * `--labels-as-of R`: training/selection visibility cutoff for `opal run`.

Checkpoint outputs:

* `outputs/rounds/round_0/selection/selection_top_k.csv`
* `outputs/ledger/runs.parquet`
* `outputs/ledger/predictions/`

#### 4. Verify + inspect

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

#### 5. Optional: confirm uncertainty exists (even though selection ignores it)

Inspect a selected record and look for the selected score/uncertainty fields.

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
```

### Continue to round 1, etc.

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

* sklearn GP `ConvergenceWarning`: common on small demo data; treat as informational if `validate` and `verify-outputs` pass.
* `SFXI min_n` failure: ingest enough labels for the same observed round you run as `--labels-as-of`.
* Unknown sequences during ingest: ensure input IDs exist in `records.parquet` (or keep `--unknown-sequences drop`).
