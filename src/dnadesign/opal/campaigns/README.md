---
id: opal-campaign-routes
title: OPAL campaign routes
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-13
surface: opal_campaign_index
---

## OPAL Campaign Routes

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-13

An executable campaign has one `configs/campaign.yaml` using
`opal.campaign.v3` and an explicit `ownership` block. Repository discovery
loads only that path shape and rejects invalid or unowned configs.

### Executable Campaigns

| Campaign | Kind | Runtime status | Purpose |
| --- | --- | --- | --- |
| [`demo_rf_sfxi_topn`](demo_rf_sfxi_topn/README.md) | demo | runnable | Canonical local RF, SFXI, and greedy-selection example; owns the shared demo records fixture |
| [`demo_gp_topn`](demo_gp_topn/README.md) | demo | runnable | Local Gaussian-process example with deterministic top-N selection |
| [`demo_gp_ei`](demo_gp_ei/README.md) | demo | runnable | Local Gaussian-process example with expected-improvement selection |
| [`secg_rmf_greedy`](secg_rmf_greedy/README.md) | study | blocked on label promotion | Stress-study learning loop with ethanol, ciprofloxacin, and AND RMF selection views |

Run state is not declared in this index. `state.json` and the run ledger are
the runtime sources of truth. The stress-study status and readiness routes are
documented under
[`docs/studies/stress_ethanol_cipro_growth/`](../../../../docs/studies/stress_ethanol_cipro_growth/).

### Digest-Pinned SFXI Source Runs

These directories retain ignored run state and outputs for audit. They have no
executable config and are not discovered as campaign routes.

| Source directory | Round-0 run ID | Labels | Scored | Selected |
| --- | --- | ---: | ---: | ---: |
| `secg_ethanol_rf_sfxi_topn` | `r0-2026-07-09T18:37:10+00:00` | 35 | 154785 | 6 |
| `secg_cipro_rf_sfxi_topn` | `r0-2026-07-09T18:37:49+00:00` | 35 | 154785 | 6 |
| `secg_and_rf_sfxi_topn` | `r0-2026-07-09T18:38:31+00:00` | 35 | 154785 | 6 |

Run IDs and artifact digests are preserved in the stress-study record. No
executable configs belong in these source directories.

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
