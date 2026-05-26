---
doc_id: study-rt-lnrna-sponging-construct-triage-routes
surface: study-route-map
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-26
entrypoint: self
status_surface: record-only
preflight_surface: planned-contract-checks
---

## rt_lnrna_sponging_construct_triage Routes

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-26

Use this page as the one-hop route record for the planned RT-lnRNA sponging
construct workbench. Ops batch orchestration for the six-view Infer lane is
registered through the study execution surfaces.

### Navigation Header

| Need | Surface |
| --- | --- |
| Study primer | `../contexts/construct-overview.md` |
| Current state | `../record/status.md` |
| Source inventories | `../record/datasets.yaml` |
| Reader SPOP label contract | `../contexts/reader-spop-label-contract.md` |
| GenBank source authority | `../workbench/provenance/genbank-source-authority.yaml` |
| Variant GenBank metadata | `../workbench/provenance/retron-variant-genbank-metadata.yaml` |
| Variant GenBank catalog | `../workbench/provenance/retron-variant-genbank-catalog.yaml` |
| Parsed feature-offset audit | `../workbench/provenance/genbank-feature-offset-audit.md` |
| Construct projection manifest | `../operations/contract/fixtures/construct/construct-projection-manifest.yaml` |
| Permuter RT-CDS DMS onboarding | `../contexts/permuter-onboarding.md` |
| Permuter RT-CDS DMS fixture | `../operations/contract/fixtures/permuter/rt-cds-dms-plan.yaml` |
| Permuter-to-Infer request fixture | `../operations/contract/fixtures/permuter/rt-cds-dms-infer-handoff.yaml` |
| Representation table contract | `../operations/contract/schemas/representation-table.schema.yaml` |
| Infer feature-bundle fixture | `../operations/contract/fixtures/infer/evo2-7b-six-view-feature-bundle.yaml` |
| Infer six-view batch runbook | `../operations/contract/surfaces/execution/runbooks/infer-7b-six-view.yaml` |
| LatentDNA workspace config | `../../../../src/dnadesign/latentdna/workspaces/rt_lnrna_sponging_construct_triage/config.yaml` |
| Machine-readable contract index | `../operations/ops.study.yaml` |
| Candidate/table schemas | `../operations/contract/schemas/` |
| Minimal construct-subject fixtures | `../operations/contract/fixtures/construct-subjects/` |

### Owner Routes

| Need | First owner surface | State |
| --- | --- | --- |
| Biological scope | this page and `../workbench/ontology/vocabulary.md` | Phase 0 planned |
| Construct subject universe | `../workbench/design_sets/v1-construct-subject-scope.md` | Phase 0 planned |
| Exact sequence authority | `../workbench/provenance/genbank-feature-offset-audit.md` | source-authority resolved |
| Additional variant GenBanks | `../workbench/provenance/retron-variant-genbank-catalog.yaml` | 36 cataloged sources: 35 retron whole-plasmid variants plus BL21 wild-type lnRNA; all Construct-representable under prefix/suffix flank adjustment |
| Construct projection | `../contexts/construct-contract.md` and `../contexts/representation-contract.md` | multi-slot strategy resolved; six source views declared |
| RT-CDS DMS variants | `../contexts/permuter-onboarding.md` | study-owned construct-subject-envelope promotion through public Permuter API |
| Source overlays | `../contexts/source-overlays.md` plus `../record/datasets.yaml` | source inventory pinned |
| Infer/LatentDNA handoff | `../contexts/representation-contract.md` plus `../operations/contract/schemas/representation-table.schema.yaml` | explicit view-name fixture, six-view batch runbook, and sidecar-backed LatentDNA review surfaces present |
| Reader SPOP labels | `../contexts/reader-spop-label-contract.md` plus `../operations/contract/readiness/checks/reader_spop_label_materialization.yaml` | planned materializer present; labels not materialized |
| OPAL readiness | `../contexts/opal-handoff.md` | `rt_lnrna_sponging_construct_triage_opal_training_examples_v1` absent; OPAL run blocked |

### Boundary Rules

- Construct subject rows are paired RT plus lnRNA constructs, not RT-only catalog rows.
- RT-CDS DMS generation uses public Permuter APIs; promotion into paired
  candidates stays study-owned.
- The working/failed `lab_anchor` names are source-history labels, not
  Construct placement roles; Construct sees `lnrna` and `rt_cds` slots.
- Literature abundance priors are not TF-sponging labels.
- Reader SPOP labels are assay evidence. They do not create Construct-backed
  construct subject ids unless explicit RT plus lnRNA sequence authority exists.
- Construct owns named-slot assembly and realized sequence projection.
- Infer owns feature aliases and vector sidecars.
- LatentDNA reviews geometry from current Infer sidecars.
- OPAL starts only after one construct subject table has one fixed-size vector `X` and
  real `SpongingAssayObservation` labels exist.
