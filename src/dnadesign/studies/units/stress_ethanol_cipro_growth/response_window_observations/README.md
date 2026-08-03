---
id: stress-ethanol-cipro-growth-response-window-observations
title: Response-window observations
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-08-02
first_hop: config/reader_response_projection.yaml
---

# Response-window observations

This package turns verified Reader experiment reductions into candidate-level
response-window observations for the stress promoter study. It does not define
SFXI, RMF, OPAL models, or campaign selection.

The handoff is:

```text
Reader catalog-v4 / record-v6 reductions
  -> exact study candidate bindings
  -> explicit study label-source decisions
  -> candidate response-window observations
  -> optional OPAL label promotion
```

Reader owns wells, trajectories, event alignment, and within-experiment
reductions. The stress study owns candidate identity, repeat disposition, and
the exact experiment used as each candidate label. OPAL consumes a verified
label publication. It does not choose among Reader experiments.

## Open next

- `config/reader_response_projection.yaml`: the exact canonical Reader
  experiment, source set, analysis settings, record contracts, primary
  reduction, and study display vocabulary consumed by this package.
- `config/observation_policy.yaml`: source-manifest pins, exact label-source
  decisions, censoring policy, and study approval. The historical Reader-bundle
  digest supports replay only. Current authoring requires an approved exact
  Reader record receipt; that pin is presently unset.
- `config/evidence/repeat_adjudication_4_8h_v1.json`: typed comparison evidence
  for every repeated candidate.
- `reader_records.py`: thin study projection over the shared public Reader
  record resolver in `dnadesign.studies.core.reader_records`.
- `reader_config_attestation.py`: exact comparison of Reader's public authoring
  payload with the study projection, bracketed by Reader verification.
- `reader_record_structure.py` and `reader_record_relations.py`: dataframe
  shape, identity, coverage, and cross-record checks.
- `historical/reader_bundle_v5.py`: explicit decoder used only to verify the
  frozen pre-RecordStore evidence path. Active source loading and publication
  cannot import it.
- `sources.py`: exact Reader alias resolution through the study candidate
  binding public API.
- `label_sources.py`: one explicit label source per eligible candidate while
  retaining every experiment as evidence.
- `uncertainty.py`: joint Reader bootstrap propagation from the selected
  source.
- `repeat_diagnostics.py`: component ranges across every measured source.
- `censoring.py`: bound provenance and the exact-label gate.
- `artifact.py`: create-only publication and full bundle verification.

## Scientific contract

The response vector is:

```text
[r00, r10, r01, r11, b00, b10, b01, b11]
```

The primary reduction is the duration-weighted geometric log mean from 4 to 8
hours after the intervention. Reader defines the reduction and field meanings.
The study applies candidate identity and source decisions without recomputing
Reader values.

One Reader experiment is one evidence unit regardless of well count. Wells are
never pooled across experiments. A candidate measured once passes through
unchanged. A repeated candidate must have one explicit disposition:

- `label_source_selected`: use the declared Reader experiment as the label source;
- `label_source_excluded`: retain all evidence but publish no label;
- `remeasure_required`: retain all evidence and wait for a new experiment; or
- `review_required`: block publication until the study decides.

All measured experiments remain in `contributions.parquet`. A
`label_source_selected`
repeat has exactly one label-source contribution. This makes “use the most
recent reviewed experiment” an auditable study decision rather than an
inferred ordering rule. Reader experiment slugs are not treated as timestamps
by code.

The selected source supplies the point estimate, joint bootstrap draws,
alternate-reduction sensitivity, and event-time sensitivity. Cross-experiment
ranges remain a separate evidence lane. Selected-source bootstrap intervals are
descriptive and set `population_coverage_claimed=false`; they do not model a
population of future experiments or promoter-cell heterogeneity.

Every primary component must be exact. A selected source with a clipped or
overflow-bounded component remains visible but receives no candidate label.
The package never imputes a bounded value and never falls back to a different
experiment automatically.

The typed repeat-evidence file binds the historical Reader manifest digest, primary
reduction, candidate ID, exact experiment set, selected source or exclusion,
status, classification, and all eight component ranges. It validates evidence
identity but does not encode a universal disagreement cutoff.

The frozen repeat adjudications select a reviewed source for eight repeated
candidates. ES22, ES25, ES28, and ES30 are excluded because unresolved source
disagreement prevents one defensible label. ES26 selects a reviewed source, but
exact-only censoring excludes that bounded source from labels. These decisions
remain evidence; they do not approve the current Reader record set.
“Unresolved source disagreement” does not claim biological heterogeneity,
technical failure, or assay-context drift; the present bulk data do not
distinguish those causes.

## Failure conditions

Publication fails on any unresolved repeat, missing or ambiguous alias, source
digest drift, candidate or sequence mismatch, undeclared experiment, malformed
repeat evidence, incomplete bootstrap coverage, non-finite vector, or missing
study approval. It also fails if Reader's public channel mapping, state mapping,
random seed, reduction window, or pre-window differs from the projection. The
in-memory evidence is sealed at preview time, so coordinated dataframe changes
cannot be published under unchanged source receipts. Artifact verification
rechecks selected-source values, contribution flags, uncertainty, sensitivities,
and record digests.

Candidate bindings remain the study source of truth for Reader aliases,
candidate IDs, sequence identity, and BaseRender metadata. This package carries
the resolved identity needed for evidence; it does not create another sequence
metadata system.

## Operator surface

Verify the approved immutable observation bundle without writing:

```bash
uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations \
  verify \
  --bundle-dir src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/response_window_observations/4_8h_v1 \
  --allowed-root src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/response_window_observations
```

For new data, first complete Reader's one lifecycle in `reader/`:

```bash
READER_EXPERIMENT=experiments/2026/20260717_stress_response_window_aggregate
uv run reader inspect "$READER_EXPERIMENT" --section plan --format json
uv run reader validate "$READER_EXPERIMENT" --format json
uv run reader run "$READER_EXPERIMENT" --dry-run --format json
uv run reader run "$READER_EXPERIMENT"
uv run reader records "$READER_EXPERIMENT" --format json
uv run reader verify "$READER_EXPERIMENT" --format json
uv run reader notebook "$READER_EXPERIMENT" --mode none
```

Then preview the study projection from `dnadesign/` without writing:

```bash
uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations \
  preview \
  --reader-root ../reader \
  --reader-experiment ../reader/experiments/2026/20260717_stress_response_window_aggregate \
  --candidate-bindings src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest
```

The adapter accepts only `plate_reader/four_state_event_window`, catalog schema
v4, record schema v6, the five exact dataframe contracts in the projection,
and digest-verified bytes. It also reads Reader's public `authoring` inspection
and compares the complete analysis block with the projection. Read-only
comparison of the stored neutral records found the same scientific values as
the earlier bundle, but current `reader verify` reports
`build.identity_mismatch`. That comparison is not approval. Reader must rerun
and verify the aggregate before the study can pin its receipt. No fallback to
the historical bundle exists on the active path.

Publish an approved immutable bundle:

```bash
uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations \
  materialize \
  --reader-root ../reader \
  --reader-experiment ../reader/experiments/2026/20260717_stress_response_window_aggregate \
  --candidate-bindings src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest \
  --out-dir src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/response_window_observations/<new-version> \
  --allowed-output-root src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/response_window_observations
```

A new `stress_ethanol_cipro_growth.response_window_observations.v3` bundle
contains:

- `manifest.json`
- `observations.parquet`
- `contributions.parquet`
- `bootstrap_draws.parquet`
- `uncertainty.parquet`
- `repeat_diagnostics.parquet`
- `reduction_sensitivity.parquet`
- `event_time_sensitivity.parquet`

The manifest pins the policy, Reader config and authoring digests, catalog and
provenance epoch, exact record revisions and content digests, study projection,
candidate bindings, value order, primary reduction, and every output digest.
Publication is create-only; a new scientific decision receives a new named
bundle. The accepted v2 bundle remains frozen campaign evidence, but is not an
authoring input for future observations.
