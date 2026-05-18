# Stress / Ethanol / Ciprofloxacin Growth Study

This directory holds the checked-in study snapshot for the stress / ethanol / ciprofloxacin promoter study. Use it to see the current study phase, the active LatentDNA review path, and the linked study notes.

- Binding file: [contexts/latentdna/binding.yaml](../../../../../docs/studies/stress_ethanol_cipro_growth/contexts/latentdna/binding.yaml)
- Workspace snapshot consumer doc: [stress-ethanol-cipro-representation-comparison.md](../../../../latentdna/docs/workflows/stress-ethanol-cipro-representation-comparison.md)
- Active deliverables: `dataset_overview`, `representation_health_summary`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`, `candidate_decision_frontier`, `candidate_x_selection_scorecard`
- Companion visuals: `balanced_design_family_margin_gallery`, `sigma35_margin_ladder_gallery`, `sigma35_stress_margin_gallery`, `context_pair_summary`, `reference_to_plan_centroid_heatmap`, `reference_standard_strength_audit`
- Appendix support: `sigma35_centroid_distance_gallery`, `design_centroid_margin_gallery`, `reference_alignment_summary`, `representation_scree_diagnostic`
- Appendix deliverables: `appendix_geometry_review`, `appendix_umap_gallery`
- Deliverable docs: [deliverables/README.md](deliverables/README.md)
- Reference-view branch: `usr_promoter_references` -> `construct_prom_eth_cip_reference_core60` -> `construct_prom_eth_cip_reference_contexts` -> `infer_prom_eth_cip_reference_views_7b`
- Study notes: [notes/README.md](notes/README.md)
- Bidirectional-context audit: [2026-05-09 bidirectional context-anchor mean confidence](notes/audits/2026-05-09-bidirectional-context-anchor-mean-confidence.md)
- View-language prose audit: [2026-05-09 view-language prose](notes/audits/2026-05-09-view-language-prose.md)
- Candidate-X rationale and story surfaces: [2026-05-10 candidate-X story surfaces](notes/rationale/2026-05-10-candidate-x-story-surfaces.md)
- Native reference processing and ontology audit: [2026-05-10 native reference processing and ontology](notes/audits/2026-05-10-native-reference-processing-and-ontology.md)

## Source Orientation

- `status/service.py`: study status service orchestration and OPS contract binding.
- `status/snapshot.py`: record-backed snapshot assembly.
- `status/preflight.py`: study-owned preflight context and check coordination.
- `status/probes/`: bounded data/runtime probes for semantic completeness,
  sequence-view contracts, and host/runtime dependencies. Deeper Infer feature
  completion remains command-backed preflight behavior, not cheap status.
- `status/ops/`: OPS provider entrypoints and status registry metadata.
- `opal_batch0/`: OPAL candidate-table sampling for this study only.
