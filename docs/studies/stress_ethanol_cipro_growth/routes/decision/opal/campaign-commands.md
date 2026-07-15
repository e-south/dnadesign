---
doc_id: study-stress-ethanol-cipro-growth-opal-campaign-commands
surface: study-runbook
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-07-15
parent_route: README.md
type: runbook
plane: control-plane
owner_boundary: opal
surface_role: execution
---

## OPAL Campaign Commands

The inactive `secg_rmf_greedy` campaign declares `ethanol`, `ciprofloxacin`,
and `and` views. It is the sole executable stress-study OPAL campaign config;
the digest-pinned SFXI source runs remain evidence in their declared y-space.

### Readiness

```bash
uv run ops catalog show opal.downstream.usr-infer-x-active-learning
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.candidate_table --config src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/sampling.yaml --validate-existing
uv run opal validate -c src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml --json
```

The campaign remains inactive; `opal validate` fails until the study publishes
typed response-window labels and its promotion manifest, then verifies all digests and views.

### Promotion and execution

The study-owned repeat-aggregation and `opal.observed_label_promotion.v1`
publisher are implemented and fail closed. The current policy has 12 unresolved
repeated candidates, nine otherwise included candidates with bounded primary
components, and no study approval, so no production observation or label bundle
exists. A finite censor bound is not an exact observed label. Generic
`opal ingest-y` cannot modify this manifest-pinned source. Preview the
label-truth gate without writing artifacts:

```bash
OBS=dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations
READER=../reader/outputs/reviews/stress_response_window/latest
BINDINGS=src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest
uv run python -m "$OBS" preview --reader-bundle "$READER" --candidate-bindings "$BINDINGS"
```

The package READMEs for `response_window_observations` and
`response_window_label_promotion` own materialization and verification details.
Only continue after preview reports `ready_to_materialize: true` and the three
published artifacts verify.

```bash
CONFIG=src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml
uv run opal validate -c "$CONFIG" --json
uv run opal init -c "$CONFIG" --json
uv run opal run -c "$CONFIG" --round 0 --json
```

### Notebook review and verification

```bash
uv run opal verify-outputs -c "$CONFIG" --view ethanol --round latest --json
uv run opal selection-set show -c "$CONFIG" --view ethanol --round latest --json
uv run opal selection-batch show -c "$CONFIG" --round latest --json
uv run opal status -c "$CONFIG" --with-ledger --json
uv run opal ctx audit -c "$CONFIG" --round latest --json
```

Required evidence: three six-row selection sets, one 18-row sequence-unique
selection batch, one model artifact, one prediction ledger, and zero mismatches.

```bash
uv run opal notebook generate -c "$CONFIG" --round latest --force --json
uv run opal notebook run -c "$CONFIG"
uv run opal review -c "$CONFIG" --view ethanol --round latest --json
```

### Synthesis boundary

The study-owned synthesis handoff must reference one run and explicit view
memberships. A passing OPAL run does not authorize synthesis.
