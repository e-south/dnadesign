---
doc_id: study-rt-lnrna-sponging-construct-triage-routes
surface: study-route-map
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-08-01
entrypoint: self
status_surface: record-only
preflight_surface: planned-contract-checks
---

## rt_lnrna_sponging_construct_triage Routes

Use this page as the one-hop route for the RT-lnRNA study. Start with the
question, then leave this page for the named owner. The workspace bridge admits
cross-repository artifacts; it does not duplicate sequence, assay, structure,
or objective semantics.

### Start with one question

| Question | First surface |
| --- | --- |
| What is the exact RT-lnRNA subject? | [Subject bindings](../workbench/provenance/subject_bindings/README.md), then the exact query command documented there. |
| What did Reader measure, and how should variants be compared? | [Reporter-response route](reporter-response-evidence.md), then the [6-10 h descriptive reduction](../contexts/reporter-response-metastudy/README.md). |
| What sequence, part, or structure is authoritative? | [Construct overview](../contexts/construct-overview.md), [GenBank source authority](../workbench/provenance/genbank-source-authority.yaml), and the owning RT-provider or retron-hairpin route named there. |
| What representation or model input is available? | [Representation contract](../contexts/representation-contract.md), then [LatentDNA review surfaces](../contexts/latentdna/review-surfaces.md). |
| What is ready now, and where may it go next? | [Current status](../record/status.md), then [OPAL handoff](../contexts/opal-handoff.md). Manuscript composition remains in the private manuscript workspace. |

The machine-readable contract index is `../operations/ops.study.yaml`; source
inventories are in `../record/datasets.yaml`. Use those after the first route is
known, not as a competing navigation surface.

### Owner Routes

| Need | First owner surface | State |
| --- | --- | --- |
| Biological scope | this page and `../workbench/ontology/vocabulary.md` | Phase 0 planned |
| Cross-study manuscript framing | `../contexts/retron-tf-decoy-design-logic.md` | routed prose context present |
| Construct subject universe | `../workbench/design_sets/v1-construct-subject-scope.md` | Phase 0 planned |
| Exact sequence authority | `../workbench/provenance/genbank-feature-offset-audit.md` | source-authority resolved |
| Exact RT-lnRNA compositions and Reader aliases | `../workbench/provenance/subject_bindings/README.md` | 49 logical subjects; exact-only resolver; optional hairpin linkage |
| RT provider publication ingress | `../../../../src/dnadesign/contracts/sequence/rt_part_publication_v1.py` | provider-neutral shared contract; publication owner and exact part digests must close before composition |
| Reader evidence to compositional subjects | `../workbench/provenance/reader_evidence_bindings/README.md` | executable exact-alias materializer; no checked-in artifact |
| Additional variant GenBanks | `../workbench/provenance/retron-variant-genbank-catalog.yaml` | 46 cataloged sources: 35 retron whole-plasmid variants, ten retron-hairpin MSD-only handoffs paired with WT Eco1 RT, plus BL21 wild-type lnRNA; all Construct-representable under prefix/suffix flank adjustment |
| Construct projection | `../contexts/construct-contract.md` and `../contexts/representation-contract.md` | multi-slot strategy resolved; six source views declared |
| RT-CDS DMS variants | `../contexts/permuter-onboarding.md` | study-owned construct-subject-envelope promotion through public Permuter API |
| Source overlays | `../contexts/source-overlays.md` plus `../record/datasets.yaml` | source inventory pinned |
| Infer/LatentDNA handoff | `../contexts/representation-contract.md` plus `../operations/contract/schemas/representation-table.schema.yaml` | explicit view-name fixture, six-view batch runbook, and sidecar-backed LatentDNA review surfaces present |
| Reporter-response evidence | `reporter-response-evidence.md` | measurements and canonical visualization ready; 6-10 h reduction selected as `provisional_descriptive`; objective readiness remains blocked |
| OPAL readiness | `../contexts/opal-handoff.md` | blocked until a constrained objective, comparable profiles, and selected fixed-size `X` exist |

### Boundary Rules

- Construct subject rows are paired RT plus lnRNA constructs, not RT-only catalog rows.
- RT-CDS DMS generation uses public Permuter APIs; promotion into paired
  candidates stays study-owned.
- The working/failed `lab_anchor` names are source-history labels, not
  Construct placement roles; Construct sees `lnrna` and `rt_cds` slots.
- Literature abundance priors are not TF-sponging labels.
- Reporter-response profiles are assay evidence. They do not create
  Construct-backed subject ids unless explicit RT plus lnRNA sequence authority exists.
- Construct owns named-slot assembly and realized sequence projection.
- Infer owns feature aliases and vector sidecars.
- LatentDNA reviews geometry from current Infer sidecars.
- OPAL starts only after one construct-subject table has one fixed-size vector
  `X`, the reporter-response study has activated a constrained objective under
  [its evidence policy](../contexts/reporter-response-evidence.md), and
  comparable uncertainty-bearing profiles exist. The meta-study selects only
  the descriptive reduction.
