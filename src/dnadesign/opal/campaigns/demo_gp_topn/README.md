---
id: opal-campaign-demo-gp-topn
title: Gaussian process top N demo
owner: dnadesign-maintainers
status: active
last_verified: 2026-10-25
surface: opal_campaign
campaign_slug: demo_gp_topn
campaign_kind: demo
runtime_status: runnable
---

## Gaussian process with top-N selection

This portable demo fits one synthetic scalar response with a Gaussian process
and ranks candidates by the predicted value. It uses the same fixture as the
other demos so the model and selector are the only meaningful differences.

Run the [isolated demo](../../docs/index.md#try-a-complete-selection-round) from
the repository root. It copies the examples into a temporary directory:

```bash
# Run the three examples in fresh temporary campaign copies.
uv run opal demo-matrix --rounds 0 --keep
```

The printed `tmp_root` contains this example under `demo_gp_topn`. Use its config
to inspect the [selection batch](../../docs/reference/cli.md#selection-batch).
For your own data, adapt the [configuration](configs/campaign.yaml) using the
[campaign round guide](../../docs/workflows/campaign-round.md).

See the [campaign round guide](../../docs/workflows/campaign-round.md).
