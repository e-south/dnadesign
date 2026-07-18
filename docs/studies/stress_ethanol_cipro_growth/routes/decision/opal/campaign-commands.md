---
doc_id: study-stress-ethanol-cipro-growth-opal-campaign-commands
surface: study-runbook
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-07-18
parent_route: README.md
type: runbook
plane: control-plane
owner_boundary: opal
surface_role: execution
---

## OPAL Campaign Commands

`secg_msrb_greedy` is the sole executable stress-study OPAL campaign config.
It predicts the neutral Reader response-window Y and applies Multistate
Response Behavior (MSRB) under three target masks. Digest-pinned SFXI runs and
the completed RMF round remain non-executable comparator evidence.

Canonical command records are split by purpose under
`operations/contract/surfaces/execution/commands/opal/`. Use the study package
READMEs for label publication and the OPAL reference docs for generic CLI
semantics.

### Readiness

```bash
uv run ops catalog show opal.downstream.usr-infer-x-active-learning
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.candidate_table --config src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/sampling.yaml --validate-existing
uv run opal validate -c src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml --json
```

The study publication supplies 27 exact labels and eight measured-candidate
exclusions. Validation binds those rows to their provenance, candidate snapshot,
exclusion projection, and campaign contract. Generic `opal ingest-y` cannot
modify this manifest-pinned source.

### Round 0 initialization

The MSRB campaign has its own state, run IDs, model, predictions, and ledgers.
Do not reuse the retired RMF campaign state or assign its run ID to this slug.
Round 0 is complete. The following sequence is the reproducible initialization
route for a clean workspace; do not rerun it against the current state without
an explicit reset decision:

```bash
CONFIG=src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml
uv run opal validate -c "$CONFIG" --json
uv run opal init -c "$CONFIG" --json
uv run opal run -c "$CONFIG" --round 0 --json
```

### Notebook review and verification

```bash
CONFIG=src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml
for VIEW in ethanol ciprofloxacin and; do
  uv run opal verify-outputs -c "$CONFIG" --view "$VIEW" --round latest --json \
    | jq -e '.summary.rows_compared == 6 and .summary.mismatch_count == 0'
  uv run opal selection-set show -c "$CONFIG" --view "$VIEW" --round latest --json \
    | jq -e '.selected_count == 6'
done
uv run opal selection-batch show -c "$CONFIG" --round latest --json \
  | jq -e '.unique_count == 18 and ([.rows[].selection_batch_key] | unique | length) == 18'
uv run opal status -c "$CONFIG" --with-ledger --json
uv run opal notebook generate -c "$CONFIG" --round latest --force --json
uv run opal notebook run -c "$CONFIG"
```

Required evidence is three six-row selection sets, one 18-row sequence-unique
selection batch, a model artifact, a prediction ledger, and zero mismatches. These checks
passed for round 0 on 2026-07-18; they establish integrity, not predictive validity.

### Synthesis boundary

The study-owned synthesis handoff must reference one run and explicit view
memberships. A passing OPAL run does not authorize synthesis. The RF remains the
campaign model; PLS4 and the frozen RMF run remain study diagnostics, and
`model_support_ready` is false.
