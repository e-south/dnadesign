---
doc_id: study-eco1-rt-repack-routes
surface: study-route-map
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-05
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
| Candidate review / handoff dev spec | `../../../dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md` |
| Residue-mask policy | `../contexts/residue-mask-policy.md` |
| Fold validation policy | `../contexts/fold-validation-policy.md` |
| Synthesis feasibility policy | `../contexts/synthesis-feasibility-policy.md` |
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
| Generic fixed-backbone IA and candidate-review dev spec | `../../../dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md` | materialized selection-readiness spec plus active RT-only candidate-handoff slice: feasibility, three-window SAE summary, candidate triage, six-slot selection, and downstream handoff |
| Implementation sequence | `../contexts/implementation-roadmap.md` | active study roadmap |
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
| Design-class expansion | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/design_class_manifest.yaml` | materialized request and generated-candidate surfaces for five added classes: clade 9 p25 contact 6/8/10 A, clade 9 p50 contact 5 A, and II-A3/`42_1` p50 contact 5 A. The current 5 A class remains the baseline |
| Thread-plan materialization | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/thread_plan/`, `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/thread_plan.yaml` | materialized locally as an explicit planned `proteinmpnn` request with seeds, temperatures, request hash, fixed/mutable positions from the simple mask, terminal missing-backbone exclusions, and no fallback |
| ProteinMPNN request adapter | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_request/`, `../../../../src/dnadesign/thread/adapters/proteinmpnn/`, `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/proteinmpnn_request/request_manifest.yaml` | materialized locally; Eco1 resolves study paths/provenance while `dnadesign.thread.adapters.proteinmpnn` owns generic ProteinMPNN helper sidecars, chain-local fixed positions, `--omit_AAs C`, request hashing, and no-fallback request validation |
| Contact-risk review | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_risk/`, `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/contact_risk/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/contact_risk_profile.yaml` | retained as evidence review; contact-density and contact-class ideas do not decide protected residues |
| ProteinMPNN sample ingest | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_sample_ingest/`, `../../../../src/dnadesign/thread/adapters/proteinmpnn/`, `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/proteinmpnn_outputs/backend_run_manifest.yaml`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/sample_table.parquet` | materialized locally from official ProteinMPNN commit `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57` using `protein_mpnn_run.py`; active batch `eco1_rt_p25_5a_n96_20260624` has 96 accepted rows and Phase 2 validates |
| Candidate table | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/candidate_table/`, `../../../../src/dnadesign/thread/candidates/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/candidate_table.parquet` | materialized locally; 96 candidate-table rows with `status=accepted`, zero protected-position edits, and canonical-position mutation summaries derived from the ProteinMPNN request manifest |
| Expanded candidate pool | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/candidate_pool_manifest.yaml` | materialized as a nonredundant sequence-hash pool with 576 synthetic candidates: 96 baseline rows plus 96 rows from each of the five generated design classes |
| Fold-check request | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_request/`, `../../../../src/dnadesign/thread/foldcheck/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/foldcheck_request/foldcheck_request_manifest.yaml` | materialized locally; one WT baseline plus 96 candidate-table rows with `status=accepted` as full 320-aa canonical sequences, planned for ColabFold `colabfold_batch` CLI execution on BU SCC |
| Fold validation | `../contexts/fold-validation-policy.md`, `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_report/`, `../../../../src/dnadesign/thread/adapters/colabfold/`, and `../../../../src/dnadesign/thread/foldcheck/` | baseline full WT plus 96-candidate report materialized from BU SCC ColabFold job `6228979`; expanded WT plus 576-candidate report is materialized under `design_classes/`; raw ColabFold outputs remain on SCC, compact reports are local |
| Expanded fold-check review | `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/foldcheck_review/foldcheck_candidate_ranking.parquet` | materialized expanded review ranking for 576 candidates; use this before feasibility and six-slot selection |
| Fold-check review | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/foldcheck_review/` | materialized locally; ranks 96 candidates, separates WT-runtime RMSD from direct cryoEM-reference RMSD, stages a selected review panel, stages a full local 97-PDB fold set for ChimeraX, writes ChimeraX scripts for selected-structure overlays, and emits an Atlas subset manifest plus alt-text-backed review visuals and a scoped marimo notebook without accepting candidates. PNG rendering is explicit opt-in |
| Review deliverables | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/` | materialized review bundle; writes mask/MSA evidence with 25%/50% threshold cues and subtype row marking, a design-class mask overview, baseline ProteinMPNN audit plots, an expanded design-class ProteinMPNN/ColabFold fold-validation plot, fold-review links, reference and fold structure browsers, selected-panel structure browsing, expanded-pool panel-selection visuals, and a marimo notebook. ESMC LLR, 300M-versus-6B score comparison, SAE feature review, and WT model-check plots remain supporting context rather than panel-selection evidence. |
| Interactive browser structure review | `../../../../src/dnadesign/thread/structure_views/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/structure_browser/` | materialized interactive inspection for the Ec86 reference and fitted fold models; uses py3Dmol through the generic `dnadesign.thread.structure_views` contract. The constraint-evidence section highlights one selected mask-evidence category at a time on the ec86kit/7V9U reference using a single high-contrast highlight color. The design/fold triage section reuses baseline foldcheck_review PDBs for audit. The panel-selection section uses the expanded fold-check structure set and shows WT plus the selected six with pLDDT, RMSD, mutation count, MSA support, mutation count near retained DNA/RNA or thumb-track positions, charge change, and chemistry warnings beside the viewer. Structure group labels are section-specific, and molecule-visibility controls stay stable while moving between visuals. Raw local PDB files remain unchanged, and the browser view does not replace ColabFold fold validation |
| ChimeraX still and pose review | `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/review_deliverables/mask_structure_context/` and `.agents/skills/chimerax-structure-review/` | materialized scripts plus explicit opt-in render paths; ChimeraX remains the still-render, manual pose-capture, and publication-image path. Browser-native py3Dmol views are for interactive review only |
| Panel-selection visuals | `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/selection/plots/` | materialized compact SVGs for `selection_design_class_gate_counts`, `selection_class_local_percentiles`, `selection_six_sequence_distance`, `selection_selected_substitutions_across_rt`, `selection_regional_mutation_burden`, and `selection_na_facing_chemistry_balance`; the review-deliverables notebook links these plus nonvisual evidence/export rows for the selected-panel table, selected protein-sequence CSV, and RT-only handoff readiness through the selection manifest |
| Atlas model-feature context | `../contexts/fold-validation-policy.md`, `../../../../src/dnadesign/thread/adapters/esm_atlas/`, `../../../../src/dnadesign/thread/structure_predictions/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/atlas_semantic_profile/` | all-97 hash-lookup/on-demand probe materialized locally; WT is accepted with compact sparse SAE activation artifacts and one Atlas/ESMFold structure row, the first synthetic ProteinMPNN candidates still return explicit Atlas 404 rows, and the remaining synthetic rows are capped as unattempted |
| Biohub ESMC SAE profile | `../contexts/fold-validation-policy.md`, `../../../../src/dnadesign/thread/adapters/biohub_esmc/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sae_profile/` | authenticated `/api/v1/encode` -> `/api/v1/logits` profile materialized locally for the baseline WT plus 96 candidates and for the expanded WT plus 576-candidate design-class pool; this is model annotation, not fold validation or candidate acceptance |
| WT ESMC mutation scoring | `../../../../src/dnadesign/permuter/src/scoring/esmc_masked_marginal/`, `../../../../src/dnadesign/thread/adapters/biohub_esmc/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_wt_mutation_scoring/` | full WT 320-position masked-marginal 300M and 6B runs materialized through authenticated Biohub `/api/v1/encode` -> `/api/v1/logits`; each writes position entropy, 6,080 non-WT single-substitution LLR rows, mask-context join, compact plots, and a redacted manifest. Non-default models write under model-specific subdirectories. This is a model check rather than a current acceptance gate |
| SAE window summary | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/sae_window_summary/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/biohub_esmc/sae_feature_window_summary.parquet` | materialized three-window WT-delta summary for 577 sequences across catalytic-palm control, thumb/palm nucleic-acid-binding surface, and mutable substrate-proximal annulus/basic-surface windows; this is review evidence, not an acceptance gate or selector |
| ESMC sequence pseudo-likelihood | `../../../../src/dnadesign/permuter/src/scoring/esmc_pseudolikelihood/`, `../../../../src/dnadesign/thread/adapters/biohub_esmc/`, and `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sequence_pseudolikelihood/` | deferred 6B leave-one-out sequence score. The method is documented and resumable, but whole-protein pseudo-likelihood is not required for the first protein review panel |
| Full-gene vs pooled-window economics | `../contexts/synthesis-feasibility-policy.md` | computational full-gene feasibility is materialized for the expanded pool; quote/build economics remain study policy |
| Selection readiness | `../../../../src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/` and `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/selection/selection_readiness_manifest.yaml` | materialized feasibility, triage, and six-row one-per-design-class panel for the expanded candidate pool; MSA support, mutation geography, local chemistry, and sequence nonredundancy explain panel choices |
| Candidate triage table | `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/selection/candidate_triage_table.parquet` | materialized table joining fold review, feasibility, MSA support, mutation geography, near-DNA/RNA or thumb-track chemistry, sequence diversity, SAE-window fields, and ESMC additive LLR fields without reducing them to one combined rank |
| Candidate selection panel | `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/selection/candidate_selection_panel.parquet` | materialized six-row panel with one feasible fold-preserved representative per design class; not a top-six score ranking and not selected by ESMC or SAE |
| Selected protein sequence export | `../../../../src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/design_classes/selection/candidate_handoff_sequences.csv` | flat RT-only protein sequence table for the selected six rows; it carries sequence hashes and explicit not-materialized DNA-design/restriction-screen status, and it is the sequence authority for later handoff planning |
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
