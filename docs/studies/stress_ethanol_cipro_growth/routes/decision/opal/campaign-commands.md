---
doc_id: study-stress-ethanol-cipro-growth-opal-campaign-commands
surface: study-runbook
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-07-20
parent_route: README.md
type: runbook
plane: control-plane
owner_boundary: opal
surface_role: execution
---

## OPAL Campaign Commands

`secg_msrb_greedy` is the sole executable stress-study OPAL campaign. It predicts
Reader response-window Y, then scores it with MSRB; SFXI and RMF are comparator evidence.

### Readiness

```bash
uv run ops catalog show opal.downstream.usr-infer-x-active-learning
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.candidate_table --config src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/sampling.yaml --validate-existing
uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/test_source_tree_contracts.py::test_msrb_activation_receipt_is_one_way_digest_bound_and_claim_scoped
uv run opal validate -c src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml --json
```

Validation binds 27 exact labels and eight exclusions to their provenance,
candidate snapshot, and campaign contract. `opal ingest-y` cannot modify it.

### Read-only campaign verification

```bash
CONFIG=src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml
for VIEW in ethanol ciprofloxacin and; do
  uv run opal objective-meta -c "$CONFIG" --view "$VIEW" --round latest --profile --json \
    | jq -e '
        .objective.objective_name == "multistate_response_behavior_v1" and
        .selection.objective_mode == "maximize" and
        (.diagnostic_keys | index("hard_bottleneck_clearance") != null) and
        (.diagnostic_keys | index("off_signal_suppression_family_score") != null)'
  uv run opal verify-outputs -c "$CONFIG" --view "$VIEW" --round latest --json \
    | jq -e '.summary.rows_compared == 6 and .summary.mismatch_count == 0'
  uv run opal selection-set show -c "$CONFIG" --view "$VIEW" --round latest --json \
    | jq -e '.selected_count == 6'
done
uv run opal selection-batch show -c "$CONFIG" --round latest --json \
  | jq -e '.unique_count == 18 and ([.rows[].selection_batch_key] | unique | length) == 18'
uv run opal status -c "$CONFIG" --with-ledger --json
```

Required evidence is three six-row sets, one 18-row sequence-unique batch, model
and prediction artifacts, declared MSRB diagnostics, and zero mismatches. The
profile exposes the mask, shared soft-min scale, score direction, and family
scores.
Passing these checks establishes artifact integrity, not biological validity.

### Notebook review

Regenerate the plot artifacts before regenerating the notebook. Notebook generation writes the notebook artifact
and binds existing plot manifests; it does not rerender figures.

```bash
CONFIG=src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml
for VIEW in ethanol ciprofloxacin and; do
  uv run opal plot -c "$CONFIG" --view "$VIEW" --json
done
uv run opal notebook generate -c "$CONFIG" --round latest --force --json
uv run opal notebook run -c "$CONFIG"
```

Do not pin `--run-id` in the general plot loop. The response-window history
plot requests all rounds; an explicit run ID would narrow that plot to one run.

### Reset and replay round 0
The campaign owns its state, model, predictions, and ledgers. Round 0 is
complete. Run these mutating commands only in a clean workspace after an
explicit reset decision:

```bash
CONFIG=src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml
uv run opal validate -c "$CONFIG" --json
uv run opal init -c "$CONFIG" --json
uv run opal run -c "$CONFIG" --round 0 --json
```

### Synthesis boundary
The synthesis handoff must name one run and its view memberships. Passing does not
authorize synthesis; the RF remains the campaign model and `model_support_ready` is false.
