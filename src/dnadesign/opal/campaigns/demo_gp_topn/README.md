## Demo Campaign: GP + SFXI + top_n

### What this demo covers

Gaussian process model with deterministic selection (`top_n`) on the configured score channel.

### Run from this directory

```bash
# Copy baseline artifacts into the workspace-local location.
cp ../demo/records.parquet ./records.parquet
# Reset campaign state to a clean baseline before rerunning.
uv run opal campaign-reset -c configs/campaign.yaml --apply --no-backup
# Initialize campaign state for this OPAL demo workflow.
uv run opal init -c configs/campaign.yaml
# Validate campaign config and plugin contracts.
uv run opal validate -c configs/campaign.yaml
# Ingest observed labels for the selected campaign round.
uv run opal ingest-y -c configs/campaign.yaml --round 0 --csv inputs/r0/vec8-b0.xlsx --unknown-sequences drop --if-exists replace --apply
# Execute OPAL training, scoring, and selection for the next round.
uv run opal run -c configs/campaign.yaml --round 0
# Verify selection outputs and ledger consistency for the chosen round.
uv run opal verify-outputs -c configs/campaign.yaml --round latest
```

### Full guide

- `src/dnadesign/opal/docs/workflows/gp-sfxi-topn.md`
