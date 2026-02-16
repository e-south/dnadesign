## Demo Flow: GP + SFXI + expected_improvement

### Intent

Use this flow for uncertainty-aware candidate acquisition:

- model: `gaussian_process`
- objective: `sfxi_v1` (score + uncertainty channels)
- selection: `expected_improvement`

This is the canonical OPAL UQ path for exploration/exploitation behavior.

### Campaign

- `src/dnadesign/opal/campaigns/demo_gp_ei/`

### End-to-end commands

Run from repo root:

```bash
cd src/dnadesign/opal/campaigns/demo_gp_ei

# 1) Create campaign-local demo records so this flow is isolated
cp ../demo/records.parquet ./records.parquet

# 2) Initialize + validate
uv run opal init -c configs/campaign.yaml
uv run opal validate -c configs/campaign.yaml

# 3) Ingest round-0 labels
uv run opal ingest-y -c configs/campaign.yaml --round 0 \
  --csv inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --yes

# 4) Train/score/select with EI
uv run opal run -c configs/campaign.yaml --round 0

# 5) Verify and inspect
uv run opal verify-outputs -c configs/campaign.yaml --round latest
uv run opal status -c configs/campaign.yaml
uv run opal runs list -c configs/campaign.yaml
```

### Expected outcome

- `verify-outputs` reports `mismatches: 0`
- latest run shows `selection=expected_improvement`
- run metadata includes `selection__score_ref` and `selection__uncertainty_ref`

### Strict EI behavior

EI does not degrade to top_n when uncertainty is missing or invalid. It fails fast.

### Round progression (r >= 1)

```bash
uv run opal ingest-y -c configs/campaign.yaml --round 1 \
  --csv inputs/r0/vec8-b0.xlsx \
  --unknown-sequences drop \
  --if-exists replace \
  --yes

uv run opal run -c configs/campaign.yaml --round 1 --resume
uv run opal verify-outputs -c configs/campaign.yaml --round latest
```
