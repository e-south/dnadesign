---
doc_id: study-stress-ethanol-cipro-growth-opal-synthesis-handoff
surface: study-context
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-07-21
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
- SHA-256 receipts for the campaign config, selection batch, candidate-record
  Parquet, study alias registry, and cloning strategy;
- one exact `(SECG-NNN, candidate_id, promoter-core SHA-256)` row per selected
  candidate;
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
--view <id>` and `opal selection-batch show`. The accepted assay-batch-1 record
can be reverified without rewriting it:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
  --handoff-id stress-opal-assay-b1-r0-msrb-v1 --json
```

Draft mode accepts one campaign and one run:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
  --source opal-round --round <model_as_of_round> --run-id <run_id> \
  --batch-id <physical_assay_batch_id> --json
```

After a lifecycle decision explicitly grants artifact generation, add the
measured-round row with `lifecycle_status: authorized_for_materialization`.
Only that exact `--handoff-id` may be run once with `--write`. After generation
and readback, change the status to `generated_pending_acceptance`; that state is
not writable. Raw `--source opal-round` remains preview-only, and accepted or
later artifact sets are immutable. A completed OPAL round or valid preview is
not authorization.

An accepted artifact set is retained with the repository so a clean checkout
can review the workbook, manifest, GenBank records, and feature table. Live
status requires every accepted file to be present and to match its recorded
digest and readback contract. Missing, partial, or modified accepted files are
an integrity failure.

The stable `SECG-NNN` registry records candidate identity and nomination
provenance. The numbering is cumulative across rounds and does not encode a
selection view or order batch. A candidate nominated but not physically used in
one batch keeps its alias and may be selected later; the registry rejects reuse
of that alias for another candidate or sequence. Alias assignment does not
prove physical ordering or measurement.
Before a future handoff is accepted for order, its exact aliases must be bound
to that lifecycle event and checked against aliases already committed by an
accepted, ordered, received, or assayed event. A pending preview does not make
an alias unavailable for a later handoff.

The frozen pre-assay SFXI exports retain their original digests and pass the
current workbook and GenBank readback checks. They are historical source
evidence, not current order files. Future reuse begins from the stable alias and
candidate identity, then generates a new lifecycle-bound handoff.
The legacy batch-level record cannot enter a committed state until actual
physical inclusion is adjudicated per alias; absence from Reader is not proof
that a sequence was never ordered.

Materialization is more restrictive than preview. `--write` must use the active
source checkout and its canonical lifecycle record, MSRB campaign config, alias
registry, and cloning strategy. The run-selected candidate-record Parquet is
also digest-bound because it supplies the annotations written into GenBank.
Each of these five inputs must match its recorded digest before any output is
created. Every input and generated artifact path must resolve inside the active
checkout, including through symlinks. Preview commands may use alternate
fixture paths because they cannot write artifacts.

### Evidence Boundary

The handoff proves identity, membership, cloning, and export integrity. Its
`accepted_for_order` state records the physical decision for this exact batch.
It does not prove response-window prediction accuracy, prospective MSRB
enrichment, exact rank stability, hill climbing, or vendor submission. Those
claims require prospective Reader measurements or a later lifecycle record.

### Validation

```bash
uv run pytest -q \
  src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff

uv run ruff check \
  src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff \
  src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff
```
