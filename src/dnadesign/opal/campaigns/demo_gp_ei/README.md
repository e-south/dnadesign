## Demo Campaign: GP + SFXI + expected_improvement

### What this demo covers

Uncertainty-aware flow with `gaussian_process` model and `expected_improvement` selection.

### Run from this directory

```bash
cp ../demo/records.parquet ./records.parquet
uv run opal campaign-reset -c configs/campaign.yaml --apply --no-backup
uv run opal init -c configs/campaign.yaml
uv run opal validate -c configs/campaign.yaml
uv run opal ingest-y -c configs/campaign.yaml --round 0 --csv inputs/r0/vec8-b0.xlsx --unknown-sequences drop --if-exists replace --apply
uv run opal run -c configs/campaign.yaml --round 0
uv run opal verify-outputs -c configs/campaign.yaml --round latest
```

### Full guide

- `src/dnadesign/opal/docs/workflows/gp-sfxi-ei.md`
