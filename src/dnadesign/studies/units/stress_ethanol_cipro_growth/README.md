---
id: stress-ethanol-cipro-growth-source
title: Stress promoter study source surfaces
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-08-01
first_hop: ../../../../../docs/studies/stress_ethanol_cipro_growth/routes/README.md
---

# Stress / Ethanol / Ciprofloxacin Growth Study

This package holds study-owned implementation, durable review prose, and static
curation records for the stress / ethanol / ciprofloxacin promoter study. Use
the docs-side study record for verified status, and use this source package
when the task needs executable study surfaces or checked-in review artifacts.

## Open the owning surface

| Need | Surface |
| --- | --- |
| Current study phase or blocker | [Checked-in status](../../../../../docs/studies/stress_ethanol_cipro_growth/record/status.md) |
| Reader reductions as candidate observations | [response_window_observations](response_window_observations/README.md) |
| Promoter alias, candidate, and sequence identity | [promoter_candidate_bindings](promoter_candidate_bindings/README.md) |
| Approved observations as OPAL labels | [response_window_label_promotion](decision/opal/response_window_label_promotion/README.md) |
| Objective interpretation, campaign inputs, or synthesis handoff | [OPAL decision surfaces](decision/opal/README.md) |
| Status and preflight implementation | [operations](operations/README.md) |
| Checked-in analysis prose and review artifacts | [workbench deliverables](workbench/deliverables/README.md) |

The path is Reader measurements → study identity and observation decisions →
study-approved labels and candidate features → OPAL objective and campaign.
None of these layers may silently supply a missing contract from another layer.

## Source orientation

- Binding file: [contexts/latentdna/binding.yaml](../../../../../docs/studies/stress_ethanol_cipro_growth/contexts/latentdna/binding.yaml)
- Workspace snapshot consumer doc: [stress-ethanol-cipro-representation-comparison.md](../../../latentdna/docs/workflows/stress-ethanol-cipro-representation-comparison.md)
- Deliverable docs: [workbench/deliverables/README.md](workbench/deliverables/README.md)
- Study notes: [workbench/notes/README.md](workbench/notes/README.md)

### LatentDNA review inventory

- Active deliverables: `dataset_overview`, `representation_health_summary`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`, `candidate_decision_frontier`, `candidate_x_selection_scorecard`
- Companion visuals: `balanced_design_family_margin_gallery`, `sigma35_margin_ladder_gallery`, `sigma35_stress_margin_gallery`, `context_pair_summary`, `reference_to_plan_centroid_heatmap`, `reference_standard_strength_audit`
- Appendix support: `sigma35_centroid_distance_gallery`, `design_centroid_margin_gallery`, `reference_alignment_summary`, `representation_scree_diagnostic`
- Appendix deliverables: `appendix_geometry_review`, `appendix_umap_gallery`
- Reference-view branch: `usr_promoter_references` -> `construct_prom_eth_cip_reference_core60` -> `construct_prom_eth_cip_reference_contexts` -> `infer_prom_eth_cip_reference_views_7b`
- Bidirectional-context audit: [2026-05-09 bidirectional context-anchor mean confidence](workbench/notes/audits/2026-05-09-bidirectional-context-anchor-mean-confidence.md)
- View-language prose audit: [2026-05-09 view-language prose](workbench/notes/audits/2026-05-09-view-language-prose.md)
- Candidate-X rationale and story surfaces: [2026-05-10 candidate-X story surfaces](workbench/notes/rationale/2026-05-10-candidate-x-story-surfaces.md)
- Native reference processing and ontology audit: [2026-05-10 native reference processing and ontology](workbench/notes/audits/2026-05-10-native-reference-processing-and-ontology.md)

```text
stress_ethanol_cipro_growth/
  promoter_candidate_bindings/ # exact alias -> candidate/sequence authority
  response_window_observations/ # Reader evidence -> candidate observations
  decision/
    opal/
      batch0/               # pre-assay OPAL candidate-table handoff
      densegen_axis_probe/  # DenseGen-label OPAL probe and TFBS learnability
      reader_promoter_evidence/ # OPAL discovery of Reader evidence bundles
      response_window_label_promotion/ # verified observations -> OPAL labels
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
- `response_window_observations/`: objective-neutral, study-owned repeat evidence,
  aggregation, uncertainty, and atomic candidate-observation publication.
- `decision/opal/batch0/`: OPAL candidate-table sampling for this study only.
- `decision/opal/densegen_axis_probe/`: study-local OPAL probes that consume
  DenseGen construction metadata through study-owned contracts.
- `decision/opal/reader_promoter_evidence/`: verifies objective-neutral Reader
  evidence bundles for OPAL discovery without owning candidate identity or RMF
  mathematics.
- `decision/opal/response_metastudy/`: compares the declared SFXI source
  evidence with response-window and RMF requirements without merging their
  vector contracts.
- `decision/opal/response_window_label_promotion/`: thin adapter from one
  verified observation bundle to OPAL's immutable label manifest.
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
