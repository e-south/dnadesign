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

`secg_rmf_greedy` declares `ethanol`, `ciprofloxacin`, and `and` views. It is
the sole executable stress-study OPAL campaign config; the digest-pinned SFXI
source runs remain evidence in their declared y-space.

Canonical command records are split by purpose under
`operations/contract/surfaces/execution/commands/opal/`. Use the study package
READMEs for label publication and the OPAL reference docs for generic CLI
semantics.

### Readiness

```bash
uv run ops catalog show opal.downstream.usr-infer-x-active-learning
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.candidate_table --config src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/sampling.yaml --validate-existing
uv run opal validate -c src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml --json
```

The study publication supplies 27 exact labels and eight measured-candidate
exclusions. Validation binds those rows to their provenance, candidate snapshot,
exclusion projection, and campaign contract. Generic `opal ingest-y` cannot
modify this manifest-pinned source.

### Round 0 state

Round 0 completed once as run `r0-2026-07-16T01:32:16+00:00`. The following
sequence is for a clean, explicitly authorized campaign initialization; do not
rerun it against the completed state:

```bash
CONFIG=src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml
uv run opal validate -c "$CONFIG" --json
uv run opal init -c "$CONFIG" --json
uv run opal run -c "$CONFIG" --round 0 --json
```

### Notebook review and verification

```bash
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
selection batch, one model artifact, one prediction ledger, and zero mismatches. These
checks establish artifact integrity, not predictive validity.

### Synthesis boundary

The study-owned synthesis handoff must reference one run and explicit view
memberships. A passing OPAL run does not authorize synthesis. The RF remains the
campaign model; PLS4 remains a study diagnostic, and `model_support_ready` is
false.
