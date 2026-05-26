## RT-lnRNA LatentDNA Review Surfaces

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-26

This page holds durable LatentDNA review-surface and browser-control semantics
for `rt_lnrna_sponging_construct_triage`. Current status and next actions live
in `../../record/status.md`.

### Decision Surfaces

- `rt_lnrna_dataset_overview`
- `rt_lnrna_source_structure_summary`
- `rt_lnrna_slot_context_robustness_summary`
- `rt_lnrna_candidate_decision_frontier`
- `rt_lnrna_candidate_x_scorecard`

### Gate

- `representation_health_summary`

### Primary Review Path

- `rt_lnrna_dataset_overview`
- `representation_health_summary`
- `rt_lnrna_source_structure_summary`
- `rt_lnrna_overlay_ordinal_audit`
- `rt_lnrna_abundance_margin_ladder_gallery`
- `rt_lnrna_abundance_margin_scatter_gallery`
- `rt_lnrna_slot_context_robustness_summary`
- `rt_lnrna_slot_geometry_scatter_gallery`
- `rt_lnrna_candidate_decision_frontier`
- `rt_lnrna_candidate_x_scorecard`

### Appendix Support

- `appendix_umap_gallery`
- `representation_scree_diagnostic`

UMAP coordinates are orientation/review surfaces only. Candidate-X selection
uses high-dimensional scalar summaries: representation health, RT-native
source/design structure, Khan/Crawford source-scoped abundance ordinal signal,
span/context robustness, and dimensional cost.

Reference overlays are row-identifiable. The first RT sets cover GenBank
catalog rows, source-family anchors, MSD compiler landmarks, Khan
abundance-affiliated rows, Crawford design-affiliated rows, and selected
Crawford high/low abundance examples. These overlays are provenance and review
aids; they are not pseudo-records and they do not merge Khan and Crawford
abundance scales.

SPOP remains a future label-readiness overlay until Reader-to-Construct labels
are materialized.
