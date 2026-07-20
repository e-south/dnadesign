---
doc_id: study-stress-ethanol-cipro-growth-opal-synthesis-handoff
surface: study-context
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-07-19
type: contract
plane: data-plane
owner_boundary: stress_ethanol_cipro_growth
surface_role: physical-synthesis-handoff
entry_artifact: opal_selection_batch
exit_artifact: vendor_neutral_synthesis_manifest
---

## Synthesis Handoff

### Purpose

The synthesis handoff converts one verified OPAL logical selection batch into
order-review artifacts without changing candidate identity or moving cloning
semantics into OPAL.

### Ontology

- **Campaign**: one shared label, model, prediction, and round lifecycle.
- **Selection view**: one target-specific objective and selector evaluated from
  shared predictions.
- **Selection batch**: the deduplicated logical union of all view selections.
- **Synthesis batch**: the study-owned physical projection after cloning checks.
- **Selection membership**: the view, rank, score, and score channel that
  nominated a candidate.

For measured rounds, `secg_msrb_greedy` is the campaign. `ethanol`,
`ciprofloxacin`, and `and` are selection views, not separate model histories.

### Source Contract

The measured-round source resolves one immutable run, verifies each named
selection set, verifies the logical batch, requires ID/rank/score agreement,
and checks selected sequences against `records.parquet`. The handoff does not
infer a view, merge campaign ledgers, fill slots, or reconstruct a rerun.

The manifest stores `selection_view_ids` and structured
`selection_memberships`. A candidate selected by multiple views remains one
physical row with every nomination.

### Pre-Assay Batch Zero

The pre-assay batch-zero order is digest-pinned source evidence. Its SFXI source
run slugs map explicitly to `ethanol`, `ciprofloxacin`, and `and` selection-view
identities. It did not result from a fitted OPAL round and must not be described
as active MSRB selection.

### Measured-Round Record

Before generation, add one row to
`docs/studies/stress_ethanol_cipro_growth/record/synthesis_handoffs.yaml` with:

- `source_authority: opal_selection_batch`;
- one `campaign_slug` and immutable `run_id`;
- `model_as_of_round` and physical `assay_batch_index`;
- one expected membership count per selection view;
- one artifact set for the deduplicated batch.

The intended greedy round declares six memberships per view and 18 unique
candidates. OPAL fails if `selection_batch.expected_unique_count: 18` is not
met; the synthesis record rechecks that boundary.

### Physical Invariants

- Candidate `id` and promoter `sequence` are unchanged.
- Promoter cores are uppercase `ACGT` and match the strategy length.
- Flanks are lowercase `acgt`.
- Candidate IDs and synthesis aliases are unique.
- Restriction sites are checked on the final insert, including junctions.
- Manifest, workbook, GenBank records, and feature table agree on identity.
- DenseGen TFBS coordinates use `offset`.
- Sigma-70 fixed-element coordinates use `offset_raw`.
- Workbook and GenBank readback pass before lifecycle acceptance.

### Operator Path

Inspect all three views and the logical union with `opal selection-set show
--view <id>` and `opal selection-batch show`. The current lifecycle does not
authorize synthesis, so the operator surface is preview-only:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
  --handoff-id <measured_round_handoff_id> --json
```

Draft mode accepts one campaign and one run:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
  --source opal-round --round <as_of_round> --run-id <run_id> --json
```

After a lifecycle decision explicitly grants synthesis authorization, the
approved command can be rerun with `--write`. A completed OPAL round or valid
preview is not that authorization.

### Evidence Boundary

The handoff proves identity, membership, cloning, and export integrity. It does
not prove response-window prediction accuracy, prospective MSRB enrichment,
exact rank stability, or order authorization. Those require OPAL model
evidence, prospective Reader measurements, and an explicit lifecycle decision.

### Validation

```bash
uv run pytest -q \
  src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff

uv run ruff check \
  src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff \
  src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff
```
