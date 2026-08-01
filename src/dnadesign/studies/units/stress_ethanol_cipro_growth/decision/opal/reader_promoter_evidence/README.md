---
id: stress-reader-promoter-evidence
title: Reader promoter-response display
owner: stress_ethanol_cipro_growth
surface: study-display-adapter
status: active
last_verified: 2026-07-29
entry_artifact: reader_catalog_v4_record_v6
exit_artifact: stress_ethanol_cipro_growth.reader_promoter_evidence.v3
---

# Reader promoter-response display

This adapter stages one canonical Reader response-window diagnostic for the
OPAL notebook. It does not run Reader, recompute a plot, produce labels, or
score an objective.

The live path is:

`Reader records → study display pin → promoter binding → portable media → OPAL viewport`

Reader owns the `plate_reader/response_window` experiment, its catalog-v4
provenance, schema-v6 records, and `plot:response_window_diagnostic`. The study
owns the selected source experiment and design, the exact promoter binding,
and the statement that this media belongs in the stress-study review surface.
OPAL verifies and renders the portable projection.

This package also owns the notebook artifact descriptor registered through the
`dnadesign.opal.reader_evidence_artifacts` entry-point group. The descriptor
routes notebook verification back through this package's authoritative v3
manifest verifier and supplies the study-specific evidence details. Generic
OPAL code contains no promoter, response-window, or stress-study semantics.

The display pin records the source experiment, design, plot-config digest, and
exact output path. The adapter then verifies:

- `response_window/designs` and `response_window/traces`, including revisions,
  revision digests, content digests, and sizes;
- the diagnostic file-bundle revision and its exact input revisions;
- every diagnostic file digest and the selected PNG or PDF signature; and
- one exact `reader.design_id` candidate binding.

Media is staged under
`reader_evidence_media/<diagnostic-revision-digest>/`. The v3 manifest remains
display-only and objective-neutral. MSRB labels, promotion, campaign state,
and the completed round-0 artifacts use separate contracts.

The projection pins the verified catalog-v4 / record-v6 diagnostic for the
reviewed SpyP source. Existing `secg_msrb_greedy` campaign files remain
immutable historical evidence; activating this display surface does not alter
labels, objectives, selections, or synthesis handoffs.

Run from the repository root after the Reader evidence is ready:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence \
  preview \
  --reader-root ../reader \
  --experiment-root ../reader/experiments/2026/20260717_stress_response_window_aggregate \
  --projection src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/config/reader_response_projection.yaml \
  --bindings-bundle src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest

uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence \
  materialize \
  --reader-root ../reader \
  --experiment-root ../reader/experiments/2026/20260717_stress_response_window_aggregate \
  --projection src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/config/reader_response_projection.yaml \
  --bindings-bundle src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest \
  --out-dir <new-review-input-directory>
```

`preview` is read-only. `materialize` verifies all source bytes before creating
the destination and publishes the manifest only after staged media verifies.
It never writes labels or campaign state.
