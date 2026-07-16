---
id: opal-campaign-routes
title: OPAL campaign routes
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-15
surface: opal_campaign_index
---

## OPAL Campaign Routes

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-15

An executable campaign has one `configs/campaign.yaml` using
`opal.campaign.v3` and an explicit `ownership` block. Repository discovery
loads only that path shape and rejects invalid or unowned configs.

### Executable Campaigns

| Campaign | Kind | Purpose |
| --- | --- | --- |
| [`demo_rf_sfxi_topn`](demo_rf_sfxi_topn/README.md) | demo | Canonical local RF, SFXI, and greedy-selection example; owns the shared demo records fixture |
| [`demo_gp_topn`](demo_gp_topn/README.md) | demo | Local Gaussian-process example with deterministic top-N selection |
| [`demo_gp_ei`](demo_gp_ei/README.md) | demo | Local Gaussian-process example with expected-improvement selection |
| [`secg_rmf_greedy`](secg_rmf_greedy/README.md) | study | Stress-study learning loop with ethanol, ciprofloxacin, and AND RMF selection views |

Run state is not declared in this index. `state.json` and the run ledger are
the runtime sources of truth. The stress-study status and readiness routes are
documented under
[`docs/studies/stress_ethanol_cipro_growth/`](../../../../docs/studies/stress_ethanol_cipro_growth/).

### Study Source Evidence

The campaign registry contains executable campaigns only. The stress study
owns its immutable SFXI round-0 run artifacts under
`workbench/source_evidence/opal_sfxi_round0/`; that evidence is not an OPAL
campaign route. Run IDs, artifact digests, and interpretation are recorded in
the study's [SFXI source-evidence record](../../../../docs/studies/stress_ethanol_cipro_growth/contexts/opal/sfxi-round0-source-evidence.md).

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
