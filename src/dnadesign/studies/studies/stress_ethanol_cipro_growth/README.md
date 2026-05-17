# Stress / Ethanol / Ciprofloxacin Growth Study

This directory holds the checked-in study snapshot for the stress / ethanol / ciprofloxacin promoter study. Use it to see the current study phase, the active LatentDNA review path, and the linked study notes.

- Binding file: [latentdna_binding.yaml](/Users/Shockwing/Dropbox/projects/phd/dnadesign/docs/studies/stress_ethanol_cipro_growth/latentdna_binding.yaml)
- Workspace snapshot consumer doc: [stress-ethanol-cipro-representation-comparison.md](/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/latentdna/docs/workflows/stress-ethanol-cipro-representation-comparison.md)
- Active deliverables: `dataset_overview`, `representation_health_summary`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`, `candidate_decision_frontier`, `candidate_x_selection_scorecard`
- Companion visuals: `balanced_design_family_margin_gallery`, `sigma35_margin_ladder_gallery`, `sigma35_stress_margin_gallery`, `context_pair_summary`, `reference_to_plan_centroid_heatmap`, `reference_standard_strength_audit`
- Appendix support: `sigma35_centroid_distance_gallery`, `design_centroid_margin_gallery`, `reference_alignment_summary`, `representation_scree_diagnostic`
- Appendix deliverables: `appendix_geometry_review`, `appendix_umap_gallery`
- Reference-view branch: `usr_promoter_references` -> `construct_prom_eth_cip_reference_core60` -> `construct_prom_eth_cip_reference_contexts` -> `infer_prom_eth_cip_reference_views_7b`
- Study notes: [notes/README.md](/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/notes/README.md)
- Bidirectional-context audit: [2026-05-09-bidirectional-context-anchor-mean-confidence-audit.md](/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/notes/2026-05-09-bidirectional-context-anchor-mean-confidence-audit.md)
- View-language prose audit: [2026-05-09-view-language-prose-audit.md](/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/notes/2026-05-09-view-language-prose-audit.md)
- Candidate-X rationale and story surfaces: [2026-05-10-candidate-x-rationale-and-story-surfaces.md](/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/notes/2026-05-10-candidate-x-rationale-and-story-surfaces.md)
- Native reference processing and ontology audit: [2026-05-10-native-reference-processing-and-ontology-audit.md](/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/notes/2026-05-10-native-reference-processing-and-ontology-audit.md)

## Source Orientation

- `status/service.py`: study status service orchestration and OPS contract binding.
- `status/snapshot.py`: record-backed snapshot assembly.
- `status/preflight.py`: study-owned preflight context and check coordination.
- `status/probes/`: bounded data/runtime probes for semantic completeness,
  sequence-view contracts, and host/runtime dependencies. Deeper Infer feature
  completion remains command-backed preflight behavior, not cheap status.
- `status/ops/`: OPS provider entrypoints and status registry metadata.
- `opal_batch0/`: OPAL candidate-table sampling for this study only.
