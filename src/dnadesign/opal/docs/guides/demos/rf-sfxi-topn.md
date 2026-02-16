## Demo Flow: RF + SFXI + top_n

### Intent

Use this flow when you want the simplest production-like OPAL path:

- model: `random_forest`
- objective: `sfxi_v1`
- selection: `top_n`

This is the baseline for validating ingestion, scoring, and selection contracts.

### Campaign

- `src/dnadesign/opal/campaigns/demo_rf_sfxi_topn/`

### End-to-end commands

Run from repo root:

```bash
cd src/dnadesign/opal/campaigns/demo_rf_sfxi_topn

# 1) Create campaign-local demo records so this flow is isolated
cp ../demo/records.parquet ./records.parquet

# 2) Initialize + validate config/data contracts
uv run opal init -c configs/campaign.yaml
uv run opal validate -c configs/campaign.yaml

# 3) Ingest observed labels for round 0
uv run opal ingest-y -c configs/campaign.yaml --round 0 \
  --csv inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --yes

# 4) Train/score/select
uv run opal run -c configs/campaign.yaml --round 0

# 5) Validate output consistency
uv run opal verify-outputs -c configs/campaign.yaml --round latest
uv run opal status -c configs/campaign.yaml
uv run opal runs list -c configs/campaign.yaml
```

### Expected outcome

- `verify-outputs` reports `mismatches: 0`
- latest run shows `selection=top_n`

### Round progression (r >= 1)

`sfxi_v1` enforces current-round scaling minimums. Ingest labels for each round before running that round:

```bash
uv run opal ingest-y -c configs/campaign.yaml --round 1 \
  --csv inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --yes

uv run opal run -c configs/campaign.yaml --round 1 --resume
uv run opal verify-outputs -c configs/campaign.yaml --round latest
```
