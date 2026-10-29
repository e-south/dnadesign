<!--
---
id: opal-campaign-demo-rf-topn
title: Random forest top N demo
owner: dnadesign-maintainers
status: active
last_verified: 2026-10-25
surface: opal_campaign
campaign_slug: demo_rf_topn
campaign_kind: demo
runtime_status: runnable
---
-->

# Random forest with top-N selection

This portable demo fits one synthetic scalar response with a random forest,
ranks the remaining candidates, and selects five rows.

Run the [isolated demo](../../docs/index.md#try-a-complete-selection-round) from
the repository root. It copies the examples into a temporary directory and keeps
the resulting campaigns for inspection:

```bash
# Run the three examples in fresh temporary campaign copies.
uv run opal demo-matrix --rounds 0 --keep
```

The printed `tmp_root` contains this example under `demo_rf_topn`. Use its config
to inspect the [selection batch](../../docs/reference/cli.md#selection-batch).
The [campaign round guide](../../docs/workflows/campaign-round.md) explains each
step and how to adapt the [configuration](configs/campaign.yaml) for your data.
