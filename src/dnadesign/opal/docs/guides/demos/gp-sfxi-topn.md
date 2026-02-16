## Demo Flow: GP + SFXI + top_n

### Intent

Use this flow when you want Gaussian Process modeling but deterministic top-k selection:

- model: `gaussian_process`
- objective: `sfxi_v1`
- selection: `top_n`

This isolates model-side uncertainty estimation while keeping selection behavior simple.

### Campaign

- `src/dnadesign/opal/campaigns/demo_gp_topn/`

### End-to-end commands

Run from repo root:

```bash
cd src/dnadesign/opal/campaigns/demo_gp_topn

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

# 4) Train/score/select
uv run opal run -c configs/campaign.yaml --round 0

# 5) Verify and inspect
uv run opal verify-outputs -c configs/campaign.yaml --round latest
uv run opal status -c configs/campaign.yaml
uv run opal runs list -c configs/campaign.yaml
```

### Expected outcome

- `verify-outputs` reports `mismatches: 0`
- latest run shows `selection=top_n`
- GP fitting may emit sklearn `ConvergenceWarning` on demo data; contracts still pass

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
