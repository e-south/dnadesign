---
id: opal-campaign-demo-gp-ei
title: Gaussian process uncertainty demo
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-08
surface: opal_campaign
campaign_slug: demo_gp_ei
campaign_kind: demo
runtime_status: runnable
---

## Gaussian process with uncertainty-aware selection

This portable demo fits one synthetic scalar response and passes the Gaussian
process predictive standard deviation through the scalar objective. The
selector then balances the predicted response and uncertainty. The stable
registry name is `expected_improvement`; its documented behavior is a
pool-relative acquisition heuristic, not classical expected improvement.

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

See the [selection contract](../../docs/plugins/selection/expected-improvement.md).
