## Uncertainty-aware OPAL rounds (GP + SFXI + EI)

This demo involves a `gaussian_process` model that produces both predictions and predictive uncertainty, `sfxi_v1` turns those into a scalar score (and score uncertainty), and `expected_improvement` selects the next batch by balancing exploitation and exploration.

Campaign: `src/dnadesign/opal/campaigns/demo_gp_ei/`

**Reference docs:**

* [Gaussian Process behavior and math](../plugins/model-gaussian-process.md)
* [Expected Improvement behavior and math](../plugins/selection-expected-improvement.md)
* [SFXI behavior and math](../plugins/objective-sfxi.md)
* [Selection plugins](../plugins/selection.md)
* [CLI reference](../reference/cli.md)

**What this doc is meant to accomplish**

* Run a round where selection is driven by *both* predicted score and predicted uncertainty.
* Make channel wiring explicit (`score_ref` + `uncertainty_ref`).
* Show EI-specific failure modes (missing uncertainty, invalid sigma, non-positive sigma).

---

### The EI wiring that matters in the config

The distinguishing feature of this workflow is the selection block. EI requires uncertainty.

```yaml
selection:                              # Uncertainty-aware acquisition strategy
  name: expected_improvement            # EI balances exploitation and exploration
  params:                               # Selection contract + EI weights
    top_k: 5                            # Number of candidates to select
    score_ref: sfxi_v1/sfxi             # Objective score channel for improvement term
    uncertainty_ref: sfxi_v1/sfxi       # Objective uncertainty channel (std-dev) for exploration term
    objective_mode: maximize            # Higher score is better
    tie_handling: competition_rank      # Tie policy for ranking output
    alpha: 1.0                          # Weight on exploitation component
    beta: 1.0                           # Weight on exploration component
```

Two important notes about refs:

1. `score_ref` always identifies a score channel key produced by the objective.
2. `uncertainty_ref` identifies an uncertainty channel key. Some objectives publish uncertainty under the same key as the score (SFXI does this for `sfxi`), so it can be valid for `score_ref` and `uncertainty_ref` to be identical.

EI contract reminder: OPAL fails fast if uncertainty is missing/invalid.

---

### Round 0 end-to-end

#### 1. Prepare the workspace

```bash
# Enter the GP+EI demo campaign directory.
cd src/dnadesign/opal/campaigns/demo_gp_ei
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

#### 3. Run round 0 with Expected Improvement

```bash
# Train, score, and select with Expected Improvement at labels-as-of round 0.
uv run opal run -c configs/campaign.yaml --labels-as-of 0
```
> * `--observed-round R`: measurement stamp for ingest.
> * `--labels-as-of R`: training/selection visibility cutoff for `opal run`.

Checkpoint:

* `outputs/rounds/round_0/selection/selection_top_k.csv`
* `outputs/ledger/runs.parquet` (includes `selection__score_ref` and `selection__uncertainty_ref`)
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

Preview the selection:

```bash
# Preview selected candidates and ranking columns.
head -n 6 outputs/rounds/round_0/selection/selection_top_k.csv
```

Optional: inspect objective channel metadata for the latest run.

```bash
# Show score/uncertainty channel refs and objective diagnostics for the latest round.
uv run opal objective-meta -c configs/campaign.yaml --round latest
```

Inspect a selected record:

```bash
# Show the top selected record (competition rank 1) from the latest round.
uv run opal record-show -c configs/campaign.yaml --selected-rank 1 --round latest --run-id latest
```

#### 5. Optional read-only analysis and plots

```bash
# Export round-level predictions for downstream analysis.
uv run opal predict -c configs/campaign.yaml --round latest --out outputs/predict_r0.parquet
# Render the score-vs-rank plot for the latest round.
uv run opal plot   -c configs/campaign.yaml --name score_vs_rank_latest --round latest
```

---

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

* Missing/invalid EI uncertainty:

  * confirm `selection.params.uncertainty_ref` matches an uncertainty channel emitted by the objective
  * confirm the model is producing non-negative predictive std
  * confirm `training.y_ops` supports inverse-transforming standard deviation (units consistency)
* Any non-positive uncertainty value: EI errors; confirm GP std is being emitted and propagated.
* `SFXI min_n` failure: ingest enough labels for the same observed round you run as `--labels-as-of`.
