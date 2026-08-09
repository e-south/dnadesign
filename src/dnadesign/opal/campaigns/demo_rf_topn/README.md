---
id: opal-campaign-demo-rf-topn
title: Random forest top N demo
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-08
surface: opal_campaign
campaign_slug: demo_rf_topn
campaign_kind: demo
runtime_status: runnable
---

## Random forest with top-N selection

This portable demo fits one synthetic scalar response with a random forest,
ranks the remaining candidates, and selects five rows. It demonstrates OPAL's
generic lifecycle; it does not encode a study objective.

From this directory:

```bash
# Copy the packaged candidate fixture into this campaign.
cp ../_fixtures/scalar-regression/records.parquet records.parquet
# Create the round-zero input directory.
mkdir -p inputs/r0
# Copy the packaged round-zero labels.
cp ../_fixtures/scalar-regression/labels.csv inputs/r0/labels.csv
# Clear prior demo state if this directory has already been run.
uv run opal campaign-reset -c configs/campaign.yaml --apply --no-backup
# Initialize the campaign ledger.
uv run opal init -c configs/campaign.yaml
# Validate the campaign and its declared inputs.
uv run opal validate -c configs/campaign.yaml
# Ingest round-zero labels into the campaign ledger.
uv run opal ingest-y -c configs/campaign.yaml --round 0 --csv inputs/r0/labels.csv --unknown-sequences drop --if-exists replace --apply
# Fit, score, and select the next batch.
uv run opal run -c configs/campaign.yaml --round 0
# Replay and verify the published outputs.
uv run opal verify-outputs -c configs/campaign.yaml --view primary --round latest
```

See the [campaign round guide](../../docs/workflows/campaign-round.md) for the
contract behind each command.
