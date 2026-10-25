---
id: opal-campaign-demo-gp-ei
title: Gaussian process uncertainty demo
owner: dnadesign-maintainers
status: active
last_verified: 2026-10-25
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

Run the [isolated demo](../../docs/index.md#try-a-complete-selection-round) from
the repository root. It copies the examples into a temporary directory:

```bash
# Run the three examples in fresh temporary campaign copies.
uv run opal demo-matrix --rounds 0 --keep
```

The printed `tmp_root` contains this example under `demo_gp_ei`. Use its config
to inspect the [selection batch](../../docs/reference/cli.md#selection-batch).
For your own data, adapt the [configuration](configs/campaign.yaml) using the
[campaign round guide](../../docs/workflows/campaign-round.md).

See the [selection contract](../../docs/plugins/selection/expected-improvement.md).
