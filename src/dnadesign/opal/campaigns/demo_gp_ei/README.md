---
id: opal-campaign-demo-gp-ei
title: Demo campaign GP SFXI expected improvement
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-13
surface: opal_campaign
campaign_slug: demo_gp_ei
campaign_kind: demo
runtime_status: runnable
---

## Demo Campaign: GP + SFXI + expected_improvement

**Owner:** OPAL
**Lifecycle:** portable demo
**Last verified:** 2026-07-13

### Purpose

Uncertainty-aware flow with `gaussian_process` model and `expected_improvement` selection.

### Run from this directory

```bash
# Reuse the canonical demo candidate records.
cp ../demo_rf_sfxi_topn/records.parquet ./records.parquet
# Remove generated state from an earlier local run.
uv run opal campaign-reset -c configs/campaign.yaml --apply --no-backup
# Initialize the campaign workspace.
uv run opal init -c configs/campaign.yaml
# Validate config, records, and plugin contracts.
uv run opal validate -c configs/campaign.yaml
# Ingest the round-0 demo labels.
uv run opal ingest-y -c configs/campaign.yaml --round 0 --csv inputs/r0/vec8-b0.xlsx --unknown-sequences drop --if-exists replace --apply
# Fit, score, and select round 0.
uv run opal run -c configs/campaign.yaml --round 0
# Verify the primary selection view against its ledgers.
uv run opal verify-outputs -c configs/campaign.yaml --view primary --round latest
```

### Full guide

- [Campaign round](../../docs/workflows/campaign-round.md)
- [Expected Improvement contract](../../docs/plugins/selection/expected-improvement.md)
- [SFXI objective contract](../../docs/plugins/objectives/sfxi.md)
