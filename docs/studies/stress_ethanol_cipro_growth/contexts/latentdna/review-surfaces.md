## Stress LatentDNA Review Surfaces

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-14

This page holds durable LatentDNA review-surface and browser-control semantics
for `stress_ethanol_cipro_growth`. Keep current status and next actions in
`../../record/status.md`.

### Decision Surfaces

- `dataset_overview`
- `design_structure_summary`
- `sigma35_ordinal_audit`
- `context_robustness_summary`
- `candidate_decision_frontier`
- `candidate_x_selection_scorecard`

### Gate

- `representation_health_summary`

### Primary Review Path

- `dataset_overview`
- `design_structure_summary`
- `sigma35_ordinal_audit`
- `context_robustness_summary`
- `candidate_decision_frontier`
- `candidate_x_selection_scorecard`

### Companion Visuals

- `balanced_design_family_margin_gallery`
- `sigma35_margin_ladder_gallery`
- `sigma35_stress_margin_gallery`
- `context_pair_summary`
- `reference_to_plan_centroid_heatmap`
- `reference_standard_strength_audit`

### Appendix Support

- `sigma35_centroid_distance_gallery`
- `native_tf_axis_orientation_audit`
- `native_regulator_plan_margin_enrichment`
- `native_regulator_plan_rank_tests.parquet` side table within
  `native_regulator_plan_margin_enrichment`
- `native_regulator_go_bp_plan_margin_enrichment`
- `plan_margin_feature_rank_tests.parquet` side table within
  `native_regulator_go_bp_plan_margin_enrichment`
- `appendix_geometry_review`
- `appendix_umap_gallery`

The current local browser artifacts include the available 7B sequence-view feature
sidecars. The default browser geometries include the controlled
equal-block bidirectional forward/RC anchor-mean candidate. Appendix
deliverables remain secondary review material, not the evidence source for
selecting `X`.

UMAP coordinates are seeded but population-fit dependent. The current appendix
UMAPs were fit with explicit recipe seeds over the expanded `160460`-row
candidate population; adding the RegulonDB-native audit quota legitimately
changes the fitted 2D coordinates even when the underlying Infer sidecars
remain complete and non-stale. Treat UMAPs as orientation views only and
compare high-dimensional scalar/neighbor metrics for study decisions.

Browser reference overlay controls are cohort-gated. The main `Hue` menu
colors the population rows, while `Reference labels`, `Reference annotations`,
and the separate `Reference hue` menu control star overlays. SFXI-scored
archive rows expose `SFXI score`, `SFXI logic fidelity`, and `SFXI effect
scaled`; Anderson and W collection rows expose `Reference strength`; RegulonDB
native core60 and BaeR/CpxR/LexA TF-axis rows expose `Native TF bin`; spyP/sulAp
and native MG1655 GenBank panels currently remain label/highlight overlays
without numeric reference hues.

### Pooling Semantics Guardrail

Infer mean-pools over token positions. Because Evo2 token states are causal in
the emitted orientation, `anchor_mean` is a prefix-conditioned anchor-span mean
from a full-sequence pass. The forward/reverse-complement concat is best
described as an equal-block, two-orientation 1 kb context-anchor summary. It is
analogous to the standard forward-plus-reverse workaround for causal sequence
models, but it is not a native bidirectional Evo2 state or hidden state.
