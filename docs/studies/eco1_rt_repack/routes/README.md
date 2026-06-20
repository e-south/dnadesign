---
doc_id: study-eco1-rt-repack-routes
surface: study-route-map
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-20
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
| Structure materializer | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure/` |
| Contact profile materializer | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact/` |
| Conservation source-sequence materializer and sufficiency gate | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/` |
| Conservation profile materializer | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/` |
| Vocabulary | `../workbench/ontology/vocabulary.md` |
| Design set | `../workbench/design_sets/eco1-rt-conservative-thread-v1.md` |
| Structure source posture | `../workbench/provenance/structure-sources.yaml` |
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
| Phase 0/1 contract validation | CLI: `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contract_validation.py`; validators: `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/` | implemented |
| Structure authority and numbering policy | `../workbench/provenance/structure-sources.yaml` and `../workbench/provenance/residue-numbering-policy.yaml` | selected |
| Structure materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure/` and `../../../outputs/thread/eco1_rt_conservative_v1/` | materialized locally |
| Contact evidence materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact/` and `../../../outputs/thread/eco1_rt_conservative_v1/contact_profile.parquet` | materialized locally |
| MSA/conservation policy | `../contexts/msa-method.md`, `../contexts/residue-mask-policy.md`, and `../workbench/provenance/conservation-sources.yaml` | source authority selected |
| Conservation source-sequence materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/` | materializer and sufficiency preflight implemented; real provider caches still missing |
| Generic MSA execution | `../../../src/dnadesign/aligner/msa/` | MAFFT bundle contract implemented; Eco1 source bundle must pass sufficiency first |
| Conservation evidence materialization | `../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/` and `../../../outputs/thread/eco1_rt_conservative_v1/conservation_profile.parquet` | materializer implemented; aligned FASTA bundle still missing |
| ProteinMPNN/LigandMPNN sampling | planned `thread` contract | not implemented |
| Fold validation | `../contexts/fold-validation-policy.md` | planned |
| Full-gene vs pooled-window economics | `../contexts/synthesis-feasibility-policy.md` | planned |
| Downstream RT-lnRNA collaboration | `../../rt_lnrna_sponging_construct_triage/routes/README.md` | explicit handoff needed |
| RT-only downstream acceptance | `../operations/contract/schemas/rt-lnrna-candidate-acceptance.schema.yaml` | scaffolded |

### Readiness Routes

| Gate | First surface | Blocks |
| --- | --- | --- |
| Structure authority | `../operations/contract/readiness/checks/structure_authority.yaml` | Contact and conservation evidence, masks, sampling. |
| Mask contract | `../operations/contract/readiness/checks/mask_contract.yaml` | MPNN/LigandMPNN request generation. |
| Sampling plan | `../operations/contract/readiness/checks/sampling_plan.yaml` | Raw sample ingest and candidate ids. |
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
