## Workflow: GP + SFXI + expected_improvement

### Intent

Use this flow for uncertainty-aware acquisition:

- model: `gaussian_process`
- objective: `sfxi_v1` (score + uncertainty channels)
- selection: `expected_improvement`

Reference docs:

- [Model plugins](../plugins/models.md)
- [Gaussian Process behavior and math](../plugins/model-gaussian-process.md)
- [Selection plugins](../plugins/selection.md)
- [Expected Improvement behavior and math](../plugins/selection-expected-improvement.md)
- [SFXI behavior and math](../plugins/objective-sfxi.md)

### Campaign

- `src/dnadesign/opal/campaigns/demo_gp_ei/`

### Guided runbook

Generate a campaign-specific guided runbook before executing commands:

```bash
cd src/dnadesign/opal/campaigns/demo_gp_ei
uv run opal guide -c configs/campaign.yaml --format markdown
uv run opal guide next -c configs/campaign.yaml --labels-as-of 0
```

### End-to-end commands (round 0 + inspection)

Run from repo root:

Round flag semantics used below:
- `ingest-y --observed-round` stamps when labels were observed.
- `run/explain --labels-as-of` selects the training cutoff used for training and selection.

```bash
cd src/dnadesign/opal/campaigns/demo_gp_ei

# 1) Create campaign-local records for this flow
cp ../demo/records.parquet ./records.parquet

# 2) Optional fresh rerun cleanup
uv run opal campaign-reset -c configs/campaign.yaml --apply --no-backup

# 3) Initialize + validate
uv run opal init -c configs/campaign.yaml
uv run opal validate -c configs/campaign.yaml

# 4) Ingest round-0 labels
uv run opal ingest-y -c configs/campaign.yaml --observed-round 0 \
  --in inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --apply

# 5) Train/score/select with EI
uv run opal run -c configs/campaign.yaml --labels-as-of 0

# 6) Verify and inspect
uv run opal verify-outputs -c configs/campaign.yaml --round latest
uv run opal status -c configs/campaign.yaml
uv run opal runs list -c configs/campaign.yaml

# 7) Inspect runtime carriers and next-round plan
uv run opal ctx audit -c configs/campaign.yaml --round latest
uv run opal explain -c configs/campaign.yaml --labels-as-of 1

# 8) Inspect one selected record
head -n 6 outputs/rounds/round_0/selection/selection_top_k.csv
selected_id="$(tail -n +2 outputs/rounds/round_0/selection/selection_top_k.csv | head -n 1 | cut -d, -f1)"
uv run opal record-show -c configs/campaign.yaml --id "${selected_id}" --run-id latest

# 9) Ephemeral predictions and plots
uv run opal predict -c configs/campaign.yaml --round latest --out outputs/predict_r0.parquet
uv run opal plot -c configs/campaign.yaml --name score_vs_rank_latest --round latest
```

### Expected outcome

- `verify-outputs` reports `mismatches: 0`
- latest run shows `selection=expected_improvement`
- run metadata includes `selection__score_ref` and `selection__uncertainty_ref`

### What to check after run

- selection CSV: `outputs/rounds/round_0/selection/selection_top_k.csv`
- run ledger: `outputs/ledger/runs.parquet`
- prediction ledger: `outputs/ledger/predictions/`
- round context: `outputs/rounds/round_0/metadata/round_ctx.json`

### Strict EI behavior

EI does not degrade to top_n when uncertainty is missing or invalid. It fails fast.

### Round progression (round 1)

```bash
uv run opal ingest-y -c configs/campaign.yaml --observed-round 1 \
  --in inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --apply

uv run opal run -c configs/campaign.yaml --labels-as-of 1 --resume
uv run opal verify-outputs -c configs/campaign.yaml --round latest
```
