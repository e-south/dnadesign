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
| [`demo_gp_topn`](demo_gp_topn/README.md) | demo | Gaussian-process model with direct score ranking |
| [`demo_gp_ei`](demo_gp_ei/README.md) | demo | Gaussian-process model with expected-improvement selection |

[`demo_rf_sfxi_topn`](demo_rf_sfxi_topn/README.md) is a portable regression
fixture for the retained SFXI objective plugin. It owns the shared demo records
file, but it is not OPAL's canonical workflow or a study route.

### Study campaign

[`secg_msrb_greedy`](secg_msrb_greedy/README.md) is the maintained
stress-promoter campaign. It applies named MSRB selection views to one shared
response-window phenotype. The stress study owns the scientific meaning,
source evidence, and synthesis gate.

Run state is not declared in this index. `state.json` and the run ledger are
the runtime sources of truth. The stress-study status and readiness routes are
documented under
[`docs/studies/stress_ethanol_cipro_growth/`](../../../../docs/studies/stress_ethanol_cipro_growth/).

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
