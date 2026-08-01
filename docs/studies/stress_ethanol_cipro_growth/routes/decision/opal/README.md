---
doc_id: study-stress-ethanol-cipro-growth-route-decision-opal
surface: study-route-detail
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-07-29
parent_route: ../../README.md
type: route
plane: control-plane
owner_boundary: opal
surface_role: decision
current_state: opal_assay_b1_order_ready
entry_artifact: usr_prom_eth_cip_opal_candidates
exit_artifact: opal_campaign_records_and_ledgers
assay_entry: reader_catalog_v4_record_v6
identity_bridge: dnadesign.study.promoter_candidate_bindings.v1
observation_bridge: stress_ethanol_cipro_growth.response_window_observations.v3
label_bridge: opal.observed_y_publication.v2
campaign: secg_msrb_greedy
synthesis_handoff: stress-opal-assay-b1-r0-msrb-v1
canonical_flow:
  - reader_records
  - promoter_binding
  - response_observations
  - label_promotion
  - immutable_campaign
  - synthesis_handoff
---

## OPAL Route Detail

Use this only after `routes/README.md` selects the OPAL campaign surface.

### Canonical assay-to-order path

```text
Reader canonical records
  -> promoter candidate binding
  -> response-window observations
  -> observed-Y label promotion
  -> immutable secg_msrb_greedy round
  -> accepted synthesis handoff
```

Each arrow is a verified artifact boundary, not a second runtime lifecycle.
Reader owns assay records and plots. The study owns identity, observation
policy, and immutable label publication. OPAL verifies and consumes those
labels and owns model fitting, scoring, and campaign ledgers; the study owns
the physical handoff. The completed round remains digest-pinned and must not
be regenerated merely because the future Reader adapter changed.

### Route coordinates

- State: `opal_assay_b1_order_ready`
- Candidate table: `usr_prom_eth_cip_opal_candidates`
- Table role: `opal_candidate_feature_table`
- Selected X: `latentdna__evo2_7b__context_anchor_mean_bidir_concat`
- Study implementation: `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/`
- Generic OPAL workflow: `src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md`

### Campaign configs and commands

Use `campaign-commands.md` for exact read-only verification and status commands;
the unified notebook displays named selection views from frozen
`decision.opal.batch0.provenance` and never revises evidence.

### Continue by question

- Candidate and label semantics: `../../../contexts/opal/candidate-table.md`
- Response reduction and objective comparison: `../../../contexts/opal/response-metastudy.md`
- MSRB protocol and interpretation: `../../../contexts/opal/multistate-response-behavior.md`
- Frozen RMF comparator: `../../../contexts/opal/response-magnitude-feasibility.md`
- DenseGen probes: `../../../contexts/opal/densegen-tfbs-learnability-probe-v1.md`
  and `../../../contexts/opal/densegen-axis-probe-v0.md`
- Physical handoff: `../../../contexts/opal/synthesis-handoff.md`
- Exact operator commands: `campaign-commands.md`

The frozen round used one shared eight-output model and three selection views.
It selected six unique sequences per view, with 18 total, and replayed with zero
output mismatches. This is campaign execution evidence, not proof of predictive
support or prospective enrichment; `model_support_ready` remains false.

### Physical Synthesis Handoff

- Lifecycle record: [synthesis_handoffs.yaml](../../../record/synthesis_handoffs.yaml)
- Dev spec: `../../../contexts/opal/synthesis-handoff.md`
- Accepted handoff: `stress-opal-assay-b1-r0-msrb-v1`
- Verify the record and retained order bundle: `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --handoff-id stress-opal-assay-b1-r0-msrb-v1 --json`
- Lifecycle state: `accepted_for_order`; vendor submission is not recorded.

### Boundaries

- OPAL consumes the study-owned candidate table and label publication, then
  owns fitting, scoring, selection, and campaign ledgers. It does not own label
  truth, DenseGen/Construct/Infer lineage, or physical synthesis.
- The unified notebook renders registered manifests. Study evidence and
  per-record provenance remain study-owned and read-only.
- Prune shared USR data only by campaign scope. A selection view owns one
  target; a selection batch owns their logical union.
- DenseGen probes exercise mechanics in silico and never create campaign or
  synthesis authority.
