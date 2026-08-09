---
id: opal-campaign-routes
title: OPAL campaign routes
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-08
surface: opal_campaign_index
---

## OPAL Campaign Routes

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-08

An executable campaign has one `configs/campaign.yaml` using
`opal.campaign.v3` and an explicit `ownership` block. Repository discovery
loads only that path shape and rejects invalid or unowned configs.

### Portable demos

| Campaign | Kind | Purpose |
| --- | --- | --- |
| [`demo_rf_topn`](demo_rf_topn/README.md) | demo | Random-forest model with direct scalar ranking |
| [`demo_gp_topn`](demo_gp_topn/README.md) | demo | Gaussian-process model with direct score ranking |
| [`demo_gp_ei`](demo_gp_ei/README.md) | demo | Gaussian-process model with uncertainty-aware selection |

All three use the small synthetic scalar fixture under
[`_fixtures/scalar-regression/`](_fixtures/scalar-regression/README.md). The
demos exercise OPAL mechanics without choosing a scientific objective.

### External campaigns

Live campaigns belong to their owning study workspace. They use OPAL's public
configuration and CLI contracts but are not packaged with dnadesign. Runtime
state remains authoritative in each campaign's `state.json` and ledger.

### Placement Rules

- Use `owner_scope: opal_demo` for portable, tool-owned examples backed by
  local fixtures. Demo ownership cannot declare a study or dataset ID.
- Use `owner_scope: study_campaign` for routed study execution. Declare
  `study_id`, `dataset_id`, and `portable: false`.
- Keep one campaign when candidate data, X, Y, labels, transforms, model, and
  round history are shared. Represent target masks or setpoints as named
  selection views.
- Create another campaign only when the learning lifecycle differs.
- Never hand-edit `state.json`, `outputs/`, ledgers, or generated notebooks.
