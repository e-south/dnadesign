---
name: eco1-rt-repack-status
description: Report record-backed status for eco1_rt_repack. Use for Eco1 RT phase, datasets, thread spec, mask/fold/synthesis policy, or RT-lnRNA handoff. Do not use for another study or for family-level routing.
metadata:
  version: 0.1.11
  category: workflow-automation
  tags: [studies, eco1-rt-repack, thread, status, routes]
---

# Eco1 RT Repack Status

## Purpose

Answer `where is eco1_rt_repack now?` from the checked-in study record and
route follow-up work to the current `thread` planning surfaces.

## Scope

In scope:
- `docs/studies/eco1_rt_repack/`
- `docs/dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md`
- Eco1 RT profile, residue-mask, fold-validation, synthesis-feasibility, and
  downstream RT-lnRNA handoff planning
- Eco1 RT MSA/conservation source authority and method routing
- implementation-roadmap and RT-only downstream acceptance planning
- record-only readiness from `operations/contract/readiness/`, including
  structure authority, mask contract, sampling plan, fold-check runtime,
  assembly feasibility, candidate handoff, and downstream handoff gates

Out of scope:
- generic `thread` package expansion beyond the implemented Eco1 tracer-bullet
  surfaces
- status for `rt_lnrna_sponging_construct_triage` or any other study
- ProteinMPNN, LigandMPNN, AlphaFold, or ColabFold execution
- wet-lab protocol advice or assay execution planning
- reconstructing current state from transient outputs when the checked-in
  record is missing

## Success Criteria

- Status answers come from `record/status.md`, `record/datasets.yaml`,
  `record/campaign.yaml`, and `operations/ops.study.yaml`.
- The answer distinguishes the implemented ProteinMPNN request, sample-ingest,
  candidate-table, fold-check request/report, ColabFold normalization, full
  fold-check coverage, study-owned fold-review bundle, local full-fold PDB set,
  alt-text-backed fold-review plots, scoped fold-review marimo notebook,
  study-owned review-deliverables bundle, selected-panel Atlas lookup surfaces,
  all-97 Biohub ESMC query-time SAE profile, and structure-prediction registry
  from feasibility and handoff tooling.
- Eco1-specific policy remains in the study; reusable ProteinMPNN request and
  sample-ingest mechanics route through `dnadesign.thread.adapters.proteinmpnn`,
  reusable candidate-table mechanics route through `dnadesign.thread.candidates`,
  and reusable fold-check request/report contracts route through
  `dnadesign.thread.foldcheck`. Reusable ColabFold output normalization routes
  through `dnadesign.thread.adapters.colabfold`. Reusable Atlas annotation
  mechanics route through `dnadesign.thread.adapters.esm_atlas`. Reusable
  authenticated Biohub ESMC query-time SAE mechanics route through
  `dnadesign.thread.adapters.biohub_esmc`. Reusable
  model-predicted-structure provenance routes through
  `dnadesign.thread.structure_predictions`. Fold-model execution, feasibility,
  and handoff tooling remain planned.
- RT-lnRNA collaboration is treated as a downstream handoff, not as ownership of
  this study's repacking policy.
- Missing or mismatched `study_id` fails visibly.

## Workflow

1. Read `docs/studies/eco1_rt_repack/operations/ops.study.yaml`,
   `record/status.md`, `record/datasets.yaml`, `record/campaign.yaml`, and
   `routes/README.md`.
2. For development-spec context, read
   `docs/dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md`.
3. For policy detail, open only the selected context page:
   `contexts/fixed-backbone-method.md`,
   `contexts/msa-method.md`,
   `contexts/residue-mask-policy.md`,
   `contexts/fold-validation-policy.md`, or
   `contexts/synthesis-feasibility-policy.md`.
4. For current contract scaffolding, inspect
   `operations/contract/readiness/`, `operations/contract/fixtures/thread/`,
   and `operations/contract/schemas/`.
5. For gate-specific blockers, select exactly one readiness group first:
   `structure_authority`, `mask_contract`, `sampling_plan`,
   `foldcheck_runtime`, `assembly_feasibility`, `candidate_handoff`, or
   `downstream_rt_lnrna_handoff`.
6. For downstream collaboration, route through
   `operations/contract/readiness/checks/downstream_rt_lnrna_handoff.yaml`
   and then the RT-lnRNA study route map only after that handoff is the actual
   question.

## Guardrails

- This skill is study-specific. Do not generalize it to another study.
- Report that `src/dnadesign/thread/` currently exposes generic ProteinMPNN
  request, sample-ingest, candidate-table, ColabFold output normalization,
  fold-check request/report contracts, and ESM Atlas sparse-activation
  normalization; do not imply that fold-model execution, feasibility, or
  handoffs are wired.
- Do not put Eco1 biology, catalytic masks, or downstream construct semantics
  into generic `thread` contracts.
- Route generic aligned FASTA generation and generic MSA QC visualization
  through public `dnadesign.aligner.msa` when needed; route Eco1
  provider-cache/source-sequence bundling, motif annotation data, exemplar row
  selections, and panel-display specs through the study surfaces, and do not
  route Eco1 source authority, conservation scoring, or mask policy into
  `aligner`.
- Report `ec86_clade9_conservation_v1` as the selected Mestre Ec86 RT clade 9
  conservation profile; Tao is the plurality/frequency masking method, not the
  source-set name. The full Mestre roster is candidate/context only and must not
  be described as accepted broad conservation evidence.
- Preserve the generic MSA visualization ontology: contracts belong under
  `aligner.msa.visualization.contracts`, orchestration and manifests under
  `aligner.msa.visualization.materialization`, and SVG drawing under
  `aligner.msa.visualization.renderers`.
- Route Eco1 motif anchors through
  `workbench/ontology/manual-mask-authority.yaml` and
  `operations/materialization/manual_mask_authority/`; route shared mask-row
  composition through `operations/masking/`; `rt-annotation-tracks.yaml` remains
  visualization/context unless a separate mask-authority record names the same
  positions. Use `eco1_rt_clade9_plurality25_direct_contact5a_v1` for the
  current mask rule: protect NAxxH, YADD, VTG, Wang/Ec86 direct
  substrate-contact priors, Ec86 clade 9 >=25% WT-plurality conservation calls,
  or mapped residues within 5 A of retained DNA/RNA. RT1-RT7 spans are
  annotation/review labels and do not blanket hard-fix residues. Terminal
  residues 1, 2, and 312-320 are `non_fixed_missing_backbone`: unprotected, but
  not directly fixed-backbone ProteinMPNN mutable until coordinates exist.
  `contact_risk_profile.yaml` is an evidence review; it does not decide which
  residues are protected unless a future task explicitly reopens the mask rule.
- Do not infer MSA source authority from review figures, prose, or public
  Eco1 accessions that disagree with the ec86kit target sequence hash.
- Do not route inverse-folding design into `permuter`; `permuter` may consume
  explicit candidate intent later through a public handoff contract.
- Do not promote candidates into RT-lnRNA construct subjects without the
  downstream study's explicit acceptance contract.

## Required Deliverables

- current phase and record posture
- canonical study record paths, the current `dnadesign.thread` adapter path, and
  the dev spec path for broader thread work
- declared dataset/artifact roots and whether they are planned or materialized
- next readiness blockers from the checked-in readiness checks
- selected route for residue masks, fold validation, synthesis feasibility, or
  RT-lnRNA handoff
- exact missing-record or mismatch errors when the scaffold is incomplete

## Trigger Tests

Should trigger:
- "Where is eco1_rt_repack now?"
- "What is the current Eco1 RT repack thread plan?"
- "Which files define the Eco1 residue-mask policy?"
- "What blocks candidate handoff from Eco1 RT repack to RT-lnRNA sponging?"

Should not trigger:
- "Where is rt_lnrna_sponging_construct_triage now?"
- "Run ProteinMPNN for Eco1 RT."
- "Implement the generic thread package."
- "Design a wet-lab Eco1 RT assay."

## References

- [study-surfaces.md](references/study-surfaces.md)
- [route-matrix.md](references/route-matrix.md)
- [refresh-loop.md](references/refresh-loop.md)
- [external-sources.md](references/external-sources.md)
- [test-matrix.md](references/test-matrix.md)
