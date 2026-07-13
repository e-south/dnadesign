---
doc_id: study-stress-ethanol-cipro-growth-opal-campaign-commands
surface: study-runbook
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-07-13
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

Validation must report the eligibility rule and all three views; it does not
establish that the typed RMF label sidecar exists.

### Promotion and execution

Use these only after the study freezes the Reader reduction, repeat aggregation,
typed sidecar, RMF calibration, model, and greedy policy.

```bash
CONFIG=src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml
uv run opal init -c "$CONFIG" --json
uv run opal ingest-y -c "$CONFIG" --round 0 \
  --csv <typed-rmf-labels.parquet-or-csv> \
  --unknown-sequences error --if-exists fail --apply --json
uv run opal run -c "$CONFIG" --round 0 --json
```

### Verification

```bash
for view in ethanol ciprofloxacin and; do
  uv run opal verify-outputs -c "$CONFIG" --view "$view" --round latest --json
  uv run opal selection-set show -c "$CONFIG" --view "$view" --round latest --json
  uv run opal objective-meta -c "$CONFIG" --view "$view" --round latest --json
  uv run opal plot -c "$CONFIG" --view "$view" --round latest
done

uv run opal selection-batch show -c "$CONFIG" --round latest --json
uv run opal status -c "$CONFIG" --with-ledger --json
uv run opal runs list -c "$CONFIG" --json
uv run opal ctx audit -c "$CONFIG" --round latest --json
```

Required evidence: three six-row selection sets, one 18-row sequence-unique
batch, one model artifact, one prediction ledger, and zero mismatches; under-capacity batches fail.

### Notebook review

```bash
uv run opal notebook generate -c "$CONFIG" --round latest --force --json
uv run opal notebook run -c "$CONFIG"
```

The notebook exposes named selection views, shared model diagnostics, and one selection batch handoff.

```bash
uv run opal review -c "$CONFIG" --view ethanol --round latest --json
```

### Synthesis boundary

The study-owned synthesis handoff must reference one run and explicit view
memberships. A passing OPAL run does not authorize synthesis.
