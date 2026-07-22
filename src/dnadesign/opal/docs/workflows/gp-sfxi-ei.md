## Uncertainty-aware OPAL rounds (GP + SFXI + EI)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-13


This workflow uses a `gaussian_process` model for predictions and predictive
uncertainty. `sfxi_v1` emits a scalar objective and corresponding uncertainty;
`expected_improvement` ranks candidates by its acquisition value.

Campaign: `src/dnadesign/opal/campaigns/demo_gp_ei/`

**Reference docs:**

* [Gaussian Process behavior and math](../plugins/models/gaussian-process.md)
* [Expected Improvement behavior and math](../plugins/selection/expected-improvement.md)
* [SFXI behavior and math](../plugins/objectives/sfxi.md)
* [Selection plugins](../plugins/selection/README.md)
* [CLI reference](../reference/cli.md)

**Outcome**

* Run a round where selection is driven by *both* predicted score and predicted uncertainty.
* Make channel wiring explicit (`score_ref` + `uncertainty_ref`).
* Exercise EI-specific failure modes: missing uncertainty, invalid sigma, and
  non-positive sigma.

---

### EI configuration

The distinguishing feature of this workflow is the selection block. EI requires uncertainty.

```yaml
selection_views: # Declare target-specific scoring and selection views.
  - id: primary # Give the only view a stable public identifier.
    objective: {name: sfxi_v1, params: {...}} # Emit SFXI score and uncertainty channels.
    selection: # Configure this view's acquisition policy.
      name: expected_improvement # Select by expected improvement.
      params: # Bind EI to declared objective channels.
        top_k: 5 # Request five candidates.
        score_ref: sfxi # Use the local SFXI score channel.
        uncertainty_ref: sfxi # Use the matching uncertainty channel.
        objective_mode: maximize # Prefer larger SFXI values.
        tie_handling: competition_rank # Preserve score ties.
        alpha: 1.0 # Weight predicted improvement.
        beta: 1.0 # Weight predictive uncertainty.
```

Two important notes about refs:

1. `score_ref` always identifies a score channel key produced by the objective.
2. `uncertainty_ref` identifies an uncertainty channel key. Some objectives publish uncertainty under the same key as the score (SFXI does this for `sfxi`), so it can be valid for `score_ref` and `uncertainty_ref` to be identical.
3. EI ranking uses normalized acquisition as the primary key; ties are broken by predicted score (higher in `maximize`, lower in `minimize`), then by `id`.

EI contract reminder: OPAL fails fast if uncertainty is missing/invalid.

---

### Round 0 end-to-end

#### 1. Prepare the workspace

```bash
# Enter the GP+EI demo campaign directory.
cd src/dnadesign/opal/campaigns/demo_gp_ei
# Copy the shared demo design-space records into this campaign.
cp ../demo_rf_sfxi_topn/records.parquet ./records.parquet

# Reset generated outputs and state for a fresh demo run.
uv run opal campaign-reset -c configs/campaign.yaml --apply --no-backup
# Initialize campaign state and workspace outputs.
uv run opal init     -c configs/campaign.yaml
# Validate config, plugin wiring, and core data contracts.
uv run opal validate -c configs/campaign.yaml
```

For a copied campaign directory outside the repository tree, invoke OPAL
through the project root:

```bash
# Run OPAL from the repo root when executing outside the project tree.
uv run --project /path/to/dnadesign opal <command> ...
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

#### 3. Run round 0 with Expected Improvement

```bash
# Train, score, and select with Expected Improvement at labels-as-of round 0.
uv run opal run -c configs/campaign.yaml --round 0
```
> `ingest-y --round R` records the measurement round. `run --round R` uses
> labels observed through round `R`.

Checkpoint:

* `outputs/rounds/round_0/selection/selections.parquet`
* `outputs/rounds/round_0/selection/selection_batch.parquet`
* `outputs/ledger/runs.parquet` (includes view-indexed selector definitions)
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

Preview the selection:

```bash
# Show the primary view's resolved selected rows.
uv run opal selection-set show -c configs/campaign.yaml --view primary --round latest
```

Optional: inspect objective channel metadata for the latest run.

```bash
# Show score/uncertainty channel refs and objective diagnostics for the latest round.
uv run opal objective-meta -c configs/campaign.yaml --view primary --round latest
```

Inspect a selected record:

```bash
# Show the top selected record (competition rank 1) from the latest round.
uv run opal record-show -c configs/campaign.yaml --view primary --selected-rank 1 --round latest --run-id latest
```

#### 5. Optional read-only analysis and plots

```bash
# Export round-level predictions for downstream analysis.
uv run opal predict -c configs/campaign.yaml --round latest --out outputs/predict_r0.parquet
# Render the score-vs-rank plot for the latest round.
uv run opal plot   -c configs/campaign.yaml --view primary --name score_vs_rank_latest --round latest
```

---

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
* Missing/invalid EI uncertainty:

  * confirm `selection.params.uncertainty_ref` matches an uncertainty channel emitted by the objective
  * confirm the selected uncertainty channel is strictly positive per candidate at selection time
  * confirm `training.y_ops` supports inverse-transforming standard deviation (units consistency)
* Any non-positive uncertainty value: EI errors; confirm GP std is being emitted and propagated.
* `SFXI min_n` failure: ingest enough labels for the round supplied to `opal run`.
