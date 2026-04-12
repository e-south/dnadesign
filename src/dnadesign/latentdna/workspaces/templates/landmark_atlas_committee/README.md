# Landmark Atlas Committee Template

This starter keeps promoter-study meaning in config only. Fill in the real USR root, validate the workspace, then materialize views and compile alignments explicitly.

The checked-in template now includes:

- paired anchor/context views for the 7B and 20B intermediate spaces
- explicit `anchor_ctx_7b` and `anchor_ctx_20b` alignments
- `delta7` and `delta20` derived view declarations
- landmark declarations and first scalar definitions
- one cohort declaration for neighborhood enrichment slices
- export definitions for `x0_primary_20b`, `x1_primary_20b`, `x2_primary_20b`, and `x3_ablation_7b`
- checked-in recipes and deliverables for:
  - `atlas_2x2_intermediate`
  - `control_neighborhood_enrichment`
  - `control_distance_margins`
  - `context_shift_primary`
  - `drag_qc`
  - `agreement_7b_vs_20b`
  - `x0_primary_20b`
  - `x1_primary_20b`
  - `x2_primary_20b`
  - `x3_ablation_7b`
- read-only notebook scaffold declarations for atlas, agreement, export, and enrichment review

The package now also carries fixture-scale package boundary and performance gates. Real promoter-study reruns are still tracked separately in the development journal and should not be inferred from the template alone.
