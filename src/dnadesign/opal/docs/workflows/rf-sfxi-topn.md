## Workflow: RF + SFXI + top_n

### Intent

Use this flow for the deterministic baseline:

- model: `random_forest`
- objective: `sfxi_v1`
- selection: `top_n`

Reference docs:

- [Configuration](../reference/configuration.md)
- [Selection plugins](../plugins/selection.md)
- [SFXI behavior and math](../plugins/objective-sfxi.md)
- [CLI reference](../reference/cli.md)

### Campaign

- `src/dnadesign/opal/campaigns/demo_rf_sfxi_topn/`

### Guided runbook

Generate a campaign-specific guided runbook before executing commands:

```bash
cd src/dnadesign/opal/campaigns/demo_rf_sfxi_topn
uv run opal guide -c configs/campaign.yaml --format markdown
uv run opal guide next -c configs/campaign.yaml --labels-as-of 0
```

### End-to-end commands (round 0 + inspection)

Run from repo root:

Round flag semantics used below:
- `ingest-y --observed-round` stamps when labels were observed.
- `run/explain --labels-as-of` selects the training cutoff used for training and selection.

```bash
cd src/dnadesign/opal/campaigns/demo_rf_sfxi_topn

# 1) Create campaign-local records for this flow
cp ../demo/records.parquet ./records.parquet

# 2) Optional fresh rerun cleanup
uv run opal campaign-reset -c configs/campaign.yaml --apply --no-backup

# 3) Initialize + validate config/data contracts
uv run opal init -c configs/campaign.yaml
uv run opal validate -c configs/campaign.yaml

# 4) Ingest observed labels for round 0
uv run opal ingest-y -c configs/campaign.yaml --observed-round 0 \
  --in inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --apply

# 5) Train/score/select
uv run opal run -c configs/campaign.yaml --labels-as-of 0

# 6) Verify core outputs
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
uv run opal plot -c configs/campaign.yaml --name feature_importance_bars_latest --round latest
```

### Expected outcome

- `verify-outputs` reports `mismatches: 0`
- latest run shows `selection=top_n`

### What to check after run

- selection CSV: `outputs/rounds/round_0/selection/selection_top_k.csv`
- run ledger: `outputs/ledger/runs.parquet`
- prediction ledger: `outputs/ledger/predictions/`
- round context: `outputs/rounds/round_0/metadata/round_ctx.json`

### Round progression (round 1)

`sfxi_v1` enforces current-round scaling minimums. Ingest labels for each round before running that round:

```bash
uv run opal ingest-y -c configs/campaign.yaml --observed-round 1 \
  --in inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --apply

uv run opal run -c configs/campaign.yaml --labels-as-of 1 --resume
uv run opal verify-outputs -c configs/campaign.yaml --round latest
```
