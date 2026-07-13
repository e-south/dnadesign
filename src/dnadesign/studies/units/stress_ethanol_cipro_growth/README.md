# Stress / Ethanol / Ciprofloxacin Growth Study

This package holds study-owned implementation, durable review prose, and static
curation records for the stress / ethanol / ciprofloxacin promoter study. Use
the docs-side study record for verified status, and use this source package
when the task needs executable study surfaces or checked-in review artifacts.

- Binding file: [contexts/latentdna/binding.yaml](../../../../../docs/studies/stress_ethanol_cipro_growth/contexts/latentdna/binding.yaml)
- Workspace snapshot consumer doc: [stress-ethanol-cipro-representation-comparison.md](../../../../latentdna/docs/workflows/stress-ethanol-cipro-representation-comparison.md)
- Active deliverables: `dataset_overview`, `representation_health_summary`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`, `candidate_decision_frontier`, `candidate_x_selection_scorecard`
- Companion visuals: `balanced_design_family_margin_gallery`, `sigma35_margin_ladder_gallery`, `sigma35_stress_margin_gallery`, `context_pair_summary`, `reference_to_plan_centroid_heatmap`, `reference_standard_strength_audit`
- Appendix support: `sigma35_centroid_distance_gallery`, `design_centroid_margin_gallery`, `reference_alignment_summary`, `representation_scree_diagnostic`
- Appendix deliverables: `appendix_geometry_review`, `appendix_umap_gallery`
- Deliverable docs: [workbench/deliverables/README.md](workbench/deliverables/README.md)
- Reference-view branch: `usr_promoter_references` -> `construct_prom_eth_cip_reference_core60` -> `construct_prom_eth_cip_reference_contexts` -> `infer_prom_eth_cip_reference_views_7b`
- Study notes: [workbench/notes/README.md](workbench/notes/README.md)
- Bidirectional-context audit: [2026-05-09 bidirectional context-anchor mean confidence](workbench/notes/audits/2026-05-09-bidirectional-context-anchor-mean-confidence.md)
- View-language prose audit: [2026-05-09 view-language prose](workbench/notes/audits/2026-05-09-view-language-prose.md)
- Candidate-X rationale and story surfaces: [2026-05-10 candidate-X story surfaces](workbench/notes/rationale/2026-05-10-candidate-x-story-surfaces.md)
- Native reference processing and ontology audit: [2026-05-10 native reference processing and ontology](workbench/notes/audits/2026-05-10-native-reference-processing-and-ontology.md)

## Source Orientation

```text
stress_ethanol_cipro_growth/
  promoter_candidate_bindings/ # exact alias -> candidate/sequence authority
  decision/
    opal/
      batch0/               # pre-assay OPAL candidate-table handoff
      densegen_axis_probe/  # DenseGen-label OPAL probe and TFBS learnability
      reader_promoter_evidence/ # OPAL discovery of Reader evidence bundles
      response_metastudy/   # SFXI/RMF evidence and promotion review
  operations/
    status/                 # OPS status/preflight provider implementation
  workbench/
    study.yaml              # LatentDNA deliverable-doc binding metadata
    deliverables/           # LatentDNA-facing checked-in review prose
    notes/                  # dated study interpretation and handoffs
    reference_sets/         # static study curation records
  tests/                    # mirrors decision/ and operations/
```

- `promoter_candidate_bindings/`: study-wide identity-routing authority for
  namespace-qualified promoter aliases, canonical candidate/sequence identity,
  and BaseRender adapter projections. Reader, synthesis, OPAL adapters, and
  other study narratives consume this artifact without redefining identity.
- `decision/opal/batch0/`: OPAL candidate-table sampling for this study only.
- `decision/opal/densegen_axis_probe/`: study-local OPAL probes that consume
  DenseGen construction metadata through study-owned contracts.
- `decision/opal/reader_promoter_evidence/`: verifies objective-neutral Reader
  evidence bundles for OPAL discovery without owning candidate identity or RMF
  mathematics.
- `decision/opal/response_metastudy/`: compares the declared SFXI source
  evidence with response-window and RMF requirements without merging their
  vector contracts.
- `operations/status/service.py`: study status service orchestration and OPS
  contract binding.
- `operations/status/snapshot.py`: record-backed snapshot assembly.
- `operations/status/preflight.py`: study-owned preflight context and check
  coordination.
- `operations/status/probes/`: bounded data/runtime probes for semantic
  completeness, sequence-view contracts, and host/runtime dependencies. Deeper
  Infer feature completion remains command-backed preflight behavior, not cheap
  status.
- `operations/status/ops/`: OPS provider entrypoints and status registry
  metadata.
- `workbench/`: durable prose and static curation records that are not Python
  execution surfaces.
