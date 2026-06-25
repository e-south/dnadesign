---
doc_id: study-eco1-rt-repack-routes
surface: study-route-map
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-25
entrypoint: self
status_surface: record-only
preflight_surface: planned-contract-checks
---

## Eco1 RT Repack Routes

Use this page as the one-hop route map for the Eco1 RT fixed-backbone redesign
study.

### Navigation Header

| Need | Surface |
| --- | --- |
| Current state | `../record/status.md` |
| Dataset and artifact posture | `../record/datasets.yaml` |
| Campaign/procedure set | `../record/campaign.yaml` |
| Fixed-backbone method | `../contexts/fixed-backbone-method.md` |
| MSA/conservation method | `../contexts/msa-method.md` |
| Implementation roadmap | `../contexts/implementation-roadmap.md` |
| Residue-mask policy | `../contexts/residue-mask-policy.md` |
| Fold validation policy | `../contexts/fold-validation-policy.md` |
| Synthesis feasibility policy | `../contexts/synthesis-feasibility-policy.md` |
| Profile fixture | `../operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml` |
| Conservative mask cases | `../operations/contract/fixtures/thread/conservative_mask_cases.yaml` |
| Eco1 profile schema | `../operations/contract/schemas/eco1-rt-profile.schema.yaml` |
| Artifact-chain schema | `../operations/contract/schemas/thread-artifact-chain.schema.yaml` |
| Candidate handoff schema | `../operations/contract/schemas/thread-candidate-handoff.schema.yaml` |
| RT-lnRNA acceptance schema | `../operations/contract/schemas/rt-lnrna-candidate-acceptance.schema.yaml` |
| Phase contract validator CLI | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contract_validation.py` |
| Contract validator package | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/` |
| Conservation contract validators | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/conservation/` |
| Mask contract validators | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/masks/` |
| Structure contract validators | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/structure/` |
| Fold-check request validator | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/foldcheck/` |
| Sequential materialization command group | `../operations/runtime/command-groups/pipeline.yaml` |
| Structure materializer | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure/` |
| Contact profile materializer | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact/` |
| Conservation provider-source materializer | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/` |
| Conservation roster-cache materializer | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/` |
| Conservation source-sequence materializer and sufficiency gate | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/` |
| Conservation profile materializer | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/` |
| Generic MSA visualization sidecars | `../../../src/dnadesign/aligner/msa/visualization/` |
| Vocabulary | `../workbench/ontology/vocabulary.md` |
| Eco1 RT annotation tracks | `../workbench/ontology/rt-annotation-tracks.yaml` |
| Eco1 MSA exemplar rows | `../workbench/ontology/msa-exemplar-rows.yaml` |
| Eco1 MSA panel spec | `../workbench/ontology/msa-panel-spec.yaml` |
| Design set | `../workbench/design_sets/eco1-rt-conservative-thread-v1.md` |
| Structure source posture | `../workbench/provenance/structure-sources.yaml` |
| Structure preprocessing policy | `../workbench/provenance/structure-preprocessing.yaml` |
| Residue-numbering policy | `../workbench/provenance/residue-numbering-policy.yaml` |
| Residue-numbering audit | `../workbench/provenance/residue-numbering-audit.md` |
| Conservation source discovery | `../workbench/provenance/conservation-source-discovery.md` |
| Conservation source contract | `../workbench/provenance/conservation-sources.yaml` |
| Machine-readable contract index | `../operations/ops.study.yaml` |

### Owner Routes

| Need | First owner surface | State |
| --- | --- | --- |
| Generic fixed-backbone IA | `../../../dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md` | proposed |
| Implementation sequence | `../contexts/implementation-roadmap.md` | proposed |
| Eco1 study profile | `../operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml` | scaffolded |
| Phase 0/1 contract validation | CLI: `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contract_validation.py`; shared validators: `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/`; domain validators: `operations/contracts/conservation/`, `operations/contracts/masks/`, and `operations/contracts/structure/` | implemented |
| Structure authority and numbering policy | `../workbench/provenance/structure-sources.yaml` and `../workbench/provenance/residue-numbering-policy.yaml` | selected |
| Structure materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure/` and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/` | materialized locally |
| Structure preprocessing provenance | `../workbench/provenance/structure-preprocessing.yaml`, `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure_preprocessing/`, and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/structure_preprocessing_manifest.yaml` | materialized locally for raw 7V9U to ec86kit protomer-1 context |
| Contact evidence materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact/` and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/contact_profile.parquet` | materialized locally |
| Contact geometry materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_geometry/` and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/contact_geometry_profile.parquet` | materialized from the selected ec86kit mmCIF model; the mask rule uses retained DNA/RNA contact within 5 A, not contact-density classes |
| MSA/conservation policy | `../contexts/msa-method.md`, `../contexts/residue-mask-policy.md`, and `../workbench/provenance/conservation-sources.yaml` | source authority selected |
| Conservation provider-source acquisition | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/` | materialized locally with explicit unresolved-provider ledger |
| Conservation roster-cache materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/` | materialized locally; full-Mestre rows are context only, and selected Ec86 clade 9 / II-A3 records carry declared QC metadata |
| Conservation source-sequence materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/` | source bundle materializer and sufficiency preflight implemented; selected source FASTA sufficiency passes locally |
| Conservation alignment materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation_alignments/` with `../../../src/dnadesign/aligner/msa/` | accepted local Clustal Omega aligned FASTA bundle exists for both selected profiles |
| Conservation MSA visualization | `../../../src/dnadesign/aligner/msa/visualization/` plus `../workbench/ontology/rt-annotation-tracks.yaml`, `../workbench/ontology/msa-exemplar-rows.yaml`, and `../workbench/ontology/msa-panel-spec.yaml` | generic diagnostic sidecar API implemented; current Eco1 local report covers both selected profiles with RT1-RT7 interval annotations, motif-anchor annotations, exemplar-window, selected-row overview, and plurality/gap histogram sidecars |
| Conservation evidence materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/` and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_profile.parquet` | materialized locally |
| Manual mask authority | `../workbench/ontology/manual-mask-authority.yaml`, `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/manual_mask_authority/`, and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/manual_mask_authority.yaml` | records NAxxH, YADD, VTG, and Wang/Ec86 direct-contact priors; RT1-RT7 spans are annotation/review labels, not blanket hard masks |
| Mask row algebra | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/masking/` | shared study-local row composition, source attribution, and summary logic used by both materialization and validation |
| Mask-set materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/mask_set/` and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/mask_set.yaml` | materialized under `eco1_rt_clade9_plurality25_direct_contact5a_v1`: protected = NAxxH/YADD/VTG, Wang/Ec86 direct contacts, Ec86 clade 9 >=25% WT plurality conservation, or mapped <=5 A retained DNA/RNA; terminal residues 1, 2, and 312-320 are `non_fixed_missing_backbone` |
| Thread-plan materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/thread_plan/`, `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/`, and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/thread_plan.yaml` | materialized locally as an explicit planned `proteinmpnn` request with seeds, temperatures, request hash, fixed/mutable positions from the simple mask, terminal missing-backbone exclusions, and no fallback |
| ProteinMPNN request adapter | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_request/`, `../../../src/dnadesign/thread/adapters/proteinmpnn/`, `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/`, and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/proteinmpnn_request/request_manifest.yaml` | materialized locally; Eco1 resolves study paths/provenance while `dnadesign.thread.adapters.proteinmpnn` owns generic ProteinMPNN helper sidecars, chain-local fixed positions, `--omit_AAs C`, request hashing, and no-fallback request validation |
| Contact-risk review | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_risk/`, `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/contact_risk/`, and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/contact_risk_profile.yaml` | retained as evidence review; contact-density and contact-class ideas do not decide protected residues |
| ProteinMPNN sample ingest | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_sample_ingest/`, `../../../src/dnadesign/thread/adapters/proteinmpnn/`, `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/proteinmpnn_outputs/backend_run_manifest.yaml`, and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/sample_table.parquet` | materialized locally from official ProteinMPNN commit `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`; active batch `eco1_rt_p25_5a_n96_20260624` has 96 accepted rows and Phase 2 validates |
| Candidate table | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/candidate_table/`, `../../../src/dnadesign/thread/candidates/`, and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/candidate_table.parquet` | materialized locally; 96 accepted candidate rows, zero protected-position edits, and canonical-position mutation summaries derived from the ProteinMPNN request manifest |
| Fold-check request | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_request/`, `../../../src/dnadesign/thread/foldcheck/`, and `../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/foldcheck_request/foldcheck_request_manifest.yaml` | materialized locally; one WT baseline plus 96 accepted candidates as full 320-aa canonical sequences, planned for ColabFold/AlphaFold-family CLI execution on BU SCC |
| Fold validation | `../contexts/fold-validation-policy.md`, `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_report/`, `../../../src/dnadesign/thread/adapters/colabfold/`, and `../../../src/dnadesign/thread/foldcheck/` | smoke report materialized from BU SCC ColabFold job `6224446`; full 96-candidate fold screen remains next |
| Atlas semantic context | `../contexts/fold-validation-policy.md` and planned `../../../src/dnadesign/thread/adapters/esm_atlas/` | planned optional ESM Atlas annotation for WT plus fold-accepted candidates; model-derived feature/similarity context only, not fold validation or candidate acceptance |
| Full-gene vs pooled-window economics | `../contexts/synthesis-feasibility-policy.md` | planned |
| Downstream RT-lnRNA collaboration | `../../rt_lnrna_sponging_construct_triage/routes/README.md` | explicit handoff needed |
| RT-only downstream acceptance | `../operations/contract/schemas/rt-lnrna-candidate-acceptance.schema.yaml` | scaffolded |

### Readiness Routes

| Gate | First surface | Blocks |
| --- | --- | --- |
| Structure authority | `../operations/contract/readiness/checks/structure_authority.yaml` | Contact and conservation evidence, masks, sampling. |
| Mask contract | `../operations/contract/readiness/checks/mask_contract.yaml` | MPNN/LigandMPNN request generation. |
| Sampling plan | `../operations/contract/readiness/checks/sampling_plan.yaml` | Backend sample ingest and candidate ids. |
| Fold-check runtime | `../operations/contract/readiness/checks/foldcheck_runtime.yaml` | Candidate acceptance. |
| Assembly feasibility | `../operations/contract/readiness/checks/assembly_feasibility.yaml` | Full-gene/window handoff decisions. |
| Candidate handoff | `../operations/contract/readiness/checks/candidate_handoff.yaml` | Downstream promotion. |

### Boundary Rules

- `thread` owns generic fixed-backbone design artifacts, request/result
  normalization, candidate ids, and handoff bundles after promotion.
- This study owns Eco1 profile policy, protected residues, and candidate-batch
  choices.
- `infer` owns model-process execution and sidecars only after explicit adapter
  contracts exist. It does not own mask algebra, candidate identity, or
  selection policy.
- `construct` owns any later named-slot realization or window feasibility.
- `permuter` remains the DMS surface; ProteinMPNN samples are not
  `permuter__var_id` records unless a later import contract says so.
