---
doc_id: study-eco1-rt-repack-routes
surface: study-route-map
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-11
entrypoint: self
status_surface: record-only
preflight_surface: runtime-and-contract-checks
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
| Runtime sequence | `../contexts/implementation-roadmap.md` |
| Selection hardening dev spec | `../contexts/selection-hardening-dev-spec.md` |
| Generation-policy contract | `../contexts/generation-policy-cleanup-dev-spec.md` |
| Candidate review / handoff dev spec | `../../../dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md` |
| Residue-mask policy | `../contexts/residue-mask-policy.md` |
| Fold validation policy | `../contexts/fold-validation-policy.md` |
| Profile fixture | `../operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml` |
| Conservative mask cases | `../operations/contract/fixtures/thread/conservative_mask_cases.yaml` |
| Eco1 profile schema | `../operations/contract/schemas/eco1-rt-profile.schema.yaml` |
| Artifact-chain schema | `../operations/contract/schemas/thread-artifact-chain.schema.yaml` |
| Candidate handoff schema | `../operations/contract/schemas/thread-candidate-handoff.schema.yaml` |
| RT-lnRNA acceptance schema | `../operations/contract/schemas/rt-lnrna-candidate-acceptance.schema.yaml` |
| Phase contract validator CLI | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contract_validation.py` |
| Contract validator package | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/` |
| Conservation contract validators | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/conservation/` |
| Mask contract validators | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/masks/` |
| Structure contract validators | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/structure/` |
| Fold-check request validator | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/foldcheck/` |
| Sequential materialization command group | `../operations/runtime/command-groups/pipeline.yaml` |
| Structure materializer | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure/` |
| Contact profile materializer | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact/` |
| Conservation provider-source materializer | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/` |
| Conservation roster-cache materializer | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/` |
| Conservation source-sequence materializer and sufficiency gate | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/` |
| Conservation profile materializer | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/` |
| Generic MSA visualization sidecars | `../../../../src/dnadesign/aligner/msa/visualization/` |
| Vocabulary | `../workbench/ontology/vocabulary.md` |
| Eco1 RT annotation tracks | `../workbench/ontology/rt-annotation-tracks.yaml` |
| Eco1 MSA exemplar rows | `../workbench/ontology/msa-exemplar-rows.yaml` |
| Eco1 MSA panel spec | `../workbench/ontology/msa-panel-spec.yaml` |
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
| Generic fixed-backbone IA and candidate-review dev spec | `../../../dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md` | v3 generation, fold review, candidate triage, within-group mutation-set selection, and RT-only handoff contract |
| Runtime sequence | `../contexts/implementation-roadmap.md` | active end-to-end execution order and fail-fast conditions |
| Eco1 study profile | `../operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml` | scaffolded |
| Phase 0/1 contract validation | CLI: `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contract_validation.py`; shared validators: `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/`; domain validators: `operations/contracts/conservation/`, `operations/contracts/masks/`, and `operations/contracts/structure/` | implemented |
| Structure authority and numbering policy | `../workbench/provenance/structure-sources.yaml` and `../workbench/provenance/residue-numbering-policy.yaml` | selected |
| Structure materialization | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/` | materialized locally |
| Structure preprocessing provenance | `../workbench/provenance/structure-preprocessing.yaml`, `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure_preprocessing/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/structure_preprocessing_manifest.yaml` | materialized locally for raw 7V9U to ec86kit protomer-1 context |
| Contact evidence materialization | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/contact_profile.parquet` | materialized locally |
| Contact geometry materialization | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_geometry/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/contact_geometry_profile.parquet` | materialized from the selected ec86kit mmCIF model; the mask rule uses retained DNA/RNA contact within 5 A, not contact-density classes |
| MSA/conservation policy | `../contexts/msa-method.md`, `../contexts/residue-mask-policy.md`, and `../workbench/provenance/conservation-sources.yaml` | source authority selected |
| Conservation provider-source acquisition | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/` | materialized locally with explicit unresolved-provider ledger |
| Conservation roster-cache materialization | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/` | materialized locally; full-Mestre rows are context only, and selected Ec86 clade 9 / II-A3 records carry declared QC metadata |
| Conservation source-sequence materialization | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/` | source bundle materializer and sufficiency preflight implemented; selected source FASTA sufficiency passes locally |
| Conservation alignment materialization | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation_alignments/` with `../../../../src/dnadesign/aligner/msa/` | accepted local Clustal Omega aligned FASTA bundle exists for both selected profiles |
| Conservation MSA visualization | `../../../../src/dnadesign/aligner/msa/visualization/` plus `../workbench/ontology/rt-annotation-tracks.yaml`, `../workbench/ontology/msa-exemplar-rows.yaml`, and `../workbench/ontology/msa-panel-spec.yaml` | generic diagnostic sidecar API implemented; current Eco1 local report covers both selected profiles with RT1-RT7 interval annotations, motif-anchor annotations, exemplar-window panels, all-record overview panels, and plurality/gap histogram sidecars |
| Conservation evidence materialization | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_profile.parquet` | materialized locally |
| Manual mask authority | `../workbench/ontology/manual-mask-authority.yaml`, `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/manual_mask_authority/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/manual_mask_authority.yaml` | records NAxxH, YADD, VTG, and Wang/Ec86 direct-contact priors; RT1-RT7 spans are annotation/review labels, not blanket hard masks |
| Mask row algebra | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/masking/` | shared study-local row composition, source attribution, and summary logic used by both materialization and validation |
| Mask-set materialization | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/mask_set/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/mask_set.yaml` | materialized under `eco1_rt_clade9_plurality25_direct_contact5a_v1`: protected = NAxxH/YADD/VTG, Wang/Ec86 direct contacts, Ec86 clade 9 >=25% WT plurality conservation, or mapped <=5 A retained DNA/RNA; terminal residues 1, 2, and 312-320 are `non_fixed_missing_backbone` |
| Generation-policy v3 requests | `../contexts/generation-policy-cleanup-dev-spec.md`, `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/generation_policies/`, `../../../../docs/bu-scc/jobs/eco1-proteinmpnn-generation-policy.qsub`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/` | materialized policy, position, alphabet, and ProteinMPNN request manifests for 1008 requested complete sequences |
| ProteinMPNN request adapter | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_request/`, `../../../../src/dnadesign/thread/adapters/proteinmpnn/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/` | Eco1 resolves study paths and policy provenance; `dnadesign.thread.adapters.proteinmpnn` owns helper sidecars, chain-local positions, declared global omissions, residue-specific `omit_AA_jsonl`, request hashing, helper parity checks, and public CLI execution |
| Contact-risk review | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_risk/`, `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/contact_risk/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/contact_risk_profile.yaml` | retained as evidence review; contact-density and contact-class ideas do not decide protected residues |
| ProteinMPNN sample ingest | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_sample_ingest/`, `../../../../src/dnadesign/thread/adapters/proteinmpnn/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/` | materialized for all three v3 policies with one policy id/version/hash per raw sample |
| Candidate table | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/candidate_table/`, `../../../../src/dnadesign/thread/candidates/`, and the v3 policy root | materialized as a 1007-row deduplicated complete-sequence pool |
| Fold-check request | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_request/` and `../../../../src/dnadesign/thread/foldcheck/` | materialized for WT plus the v3 candidate pool |
| Fold validation | `../contexts/fold-validation-policy.md`, `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_report/`, `../../../../src/dnadesign/thread/adapters/colabfold/`, and `../../../../src/dnadesign/thread/foldcheck/` | v3 ColabFold outputs and normalized report are materialized |
| Generation-policy fold-check review | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/` | v3 fold-review ranking and normalized local structures are materialized |
| Fold-check review | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/` | separates WT-runtime RMSD from direct cryoEM-reference RMSD and stages the materialized v3 structure set |
| Review deliverables | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/` | manifest-driven v3 selection plots, selected structures, and notebook; ESMC and SAE remain optional model checks |
| Communication visuals | `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/communication_visuals/` | source-derived residue map, py3Dmol structure story, 16:9 full-turn protected-evidence movie, 16:9 full-turn WT-plus-selected qualitative Coulombic movie with a fixed unit-bearing scale, structural screen, and selected-mutation map; all are exposed through the notebook communication evidence set |
| Interactive browser structure review | `../../../../src/dnadesign/thread/structure_views/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/structure_browser/` | materialized interactive inspection for the Ec86 reference and fitted fold models; uses py3Dmol through the generic `dnadesign.thread.structure_views` contract. The constraint-evidence section highlights one selected mask-evidence category at a time on the ec86kit/7V9U reference using a single high-contrast highlight color. The design/fold triage section reuses baseline foldcheck_review PDBs for audit. The panel-selection section uses the active selection root and shows WT plus all eight selected sequences, with pLDDT, RMSD, mutation count, MSA support, mutation count near retained DNA/RNA or thumb-track positions, charge change, and chemistry warnings beside the viewer. Structure group labels are section-specific, and molecule-visibility controls stay stable while moving between visuals. Raw local PDB files remain unchanged, and the browser view does not replace ColabFold fold validation |
| ChimeraX movie and pose review | `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/communication_visuals/` and `.agents/skills/chimerax-structure-review/` | reproducible scripts plus explicit opt-in movie rendering; ChimeraX provides publication output while py3Dmol provides interactive inspection |
| Panel-selection visuals | v3 `selection/plots/` under the generation-policy root | materialized; the notebook exposes only files declared present by the v3 selection manifest |
| Optional model checks | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/atlas_semantic_profile/`, `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sae_profile/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_wt_mutation_scoring/` | supplementary ESMC, SAE, and Atlas context only; none filters v3 candidates or validates function |
| Selection readiness | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/` and the v3 `selection/selection_readiness_manifest.yaml` | materialized flow: 1007 complete sequences, 738 local-geometry pass, design groups of 335 distal, 226 peripheral, and 177 combined rows, then eight selected sequences; R13 is reported but does not filter or rank rows |
| Candidate triage table | v3 `selection/candidate_triage_table.parquet` | materialized policy provenance, fold/local geometry, mutation geography, charge events, and MSA support without a composite activity score |
| Candidate selection panel | v3 `selection/candidate_selection_panel.parquet` | eight selected sequences: two distal, three peripheral, and three combined; within each group, mutated-position distance precedes exact-substitution distance |
| Selected protein sequence export | v3 `selection/candidate_handoff_sequences.csv` | materialized canonical 320-aa RT protein export with mapped-sequence provenance; it is not a DNA or construct design |
| Twist full-CDS handoff | v3 `twist_handoff/` | all eight selected 963-bp CDS designs with CSV, FASTA, annotated GenBank files, hashes, and sequence QC; assembly flanks remain pending |
| Downstream RT-lnRNA collaboration | `../../rt_lnrna_sponging_construct_triage/routes/README.md` | explicit handoff needed |
| RT-only downstream acceptance | `../operations/contract/schemas/rt-lnrna-candidate-acceptance.schema.yaml` | scaffolded |

### Readiness Routes

| Gate | First surface | Blocks |
| --- | --- | --- |
| Structure authority | `../operations/contract/readiness/checks/structure_authority.yaml` | Contact and conservation evidence, masks, sampling. |
| Mask contract | `../operations/contract/readiness/checks/mask_contract.yaml` | MPNN/LigandMPNN request generation. |
| Sampling plan | `../operations/contract/readiness/checks/sampling_plan.yaml` | Backend sample ingest and candidate ids. |
| Fold-check runtime | `../operations/contract/readiness/checks/foldcheck_runtime.yaml` | Candidate acceptance. |
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
