---
name: eco1-rt-repack-status
description: Report record-backed status for eco1_rt_repack. Use for Eco1 RT phase, datasets, generation policy, fold/selection review, or RT-lnRNA handoff. Do not use for another study or for family-level routing.
metadata:
  version: 0.1.23
  category: workflow-automation
  tags: [studies, eco1-rt-repack, thread, status, routes]
---

# Eco1 RT Repack Status

## Purpose

Answer `where is eco1_rt_repack now?` from the checked-in study record and
route follow-up work to the current study and `thread` surfaces.

## Study Premise

This study asks whether complete ProteinMPNN-designed Eco1/Ec86 RT sequences
can keep declared catalytic, direct-contact, Wang thumb-track, and mapped
residues 255-311 fixed, preserve local C-alpha backbone geometry, and introduce
MSA-observed, non-acidifying substitutions in the declared peripheral
nucleic-acid-facing shell for a diversity-seeking experimental panel.

The study does not claim improved activity, affinity, processivity, strand
displacement, safety, or a monomeric RT-msDNA assembly state.

## Scope

In scope:
- `docs/studies/eco1_rt_repack/`
- `docs/dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md`
- Eco1 RT profile, residue-mask, fold-validation, panel-selection, and
  downstream RT-lnRNA handoff planning
- Eco1 RT MSA/conservation source authority and method routing
- implementation-roadmap and RT-only downstream acceptance planning
- record-only readiness from `operations/contract/readiness/`, including
  structure authority, mask contract, sampling plan, fold-check runtime,
  candidate handoff and downstream handoff gates

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
- Status answers describe the scientific flow directly: Eco1/Ec86 source
  authority, conservation/MSA evidence, protected-residue mask, ProteinMPNN
  proposals, ColabFold fold review, MSA/geography/charge review, an eight-row
  selected panel, optional ESMC/SAE annotation, and RT-only handoff.
- The answer reports the materialized v3 path: 1008 requested sequences across
  distal, peripheral, and combined peripheral-plus-distal policies; 1007 unique
  candidates; 738 local-geometry-pass rows; policy pools of 335 distal, 226
  peripheral, and 177 combined rows; and an eight-row selected panel containing
  two distal, three peripheral, and three combined sequences. All active rows
  carry the v3 policy hash.
- State that the fold workflow models one RT chain and does not establish
  RT-msDNA oligomeric state. Wang tested R13A as an interface-disrupting
  substitution; no sequence in the v3 pool contains R13A. Report exact F10/R13
  states without treating them as gates or monomer evidence.
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
  `dnadesign.thread.structure_predictions`. Reusable remote-execution
  orchestration and downstream acceptance tooling remain separate from the
  current protein-panel selection.
- RT-lnRNA collaboration is treated as a downstream handoff, not as ownership of
  this study's repacking policy.
- Report ESMC LLR and SAE windows as review evidence only. They do not select
  panel rows, do not define candidate acceptance, and do not show improved
  strand displacement. Current panel eligibility also requires local-structure
  metrics to stay at or below the declared 2.5 A review cutoff in every
  non-distal region after one global mapped C-alpha fit. Distal RMSD is review
  context. Within each policy, the first pair is chosen by exhaustive
  mutated-position Jaccard distance and then exact-substitution distance; each
  additional peripheral or combined row maximizes minimum distance from its
  policy pair. Charge counts, regional MSA support, local RMSD, fold metrics,
  and sequence hash are later tie-breaks. Policy counts define experimental
  contrasts rather than quality tiers. Whole-protein ESMC
  pseudo-likelihood and computational stability prediction stay deferred unless
  a later task explicitly reopens those paths.
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
   `contexts/selection-hardening-dev-spec.md` for panel-selection semantics,
   claim boundaries, and plot-role wording.
4. For current contract scaffolding, inspect
   `operations/contract/readiness/`, `operations/contract/fixtures/thread/`,
   and `operations/contract/schemas/`.
5. For gate-specific blockers, select exactly one readiness group first:
   `structure_authority`, `mask_contract`, `sampling_plan`,
   `foldcheck_runtime`, `candidate_handoff`, or `downstream_rt_lnrna_handoff`.
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
  positions. Active generation policies protect NAxxH, YADD, VTG, direct
  retained DNA/RNA contacts at or below 5 A, Wang thumb-track positions, mapped
  residues 255-311, and declared conserved/core positions. RT1-RT7 spans are
  annotation labels, not blanket protection rules. Terminal residues 1, 2, and
  312-320 lack mapped backbone coordinates and are not fixed-backbone
  ProteinMPNN design positions. Peripheral residues have an explicit
  `omit_AA_jsonl` alphabet, and v3 uses global `--omit_AAs C`. Peripheral
  alternatives must be MSA-observed and introduce no new D/E, P, or G. C233 is
  open and therefore forced to change under the no-cysteine rule; report that
  recurrence as generation bias, not a protected-position violation. Do not
  compose mutations across policies.
  `contact_risk_profile.yaml` is an evidence review; it does not decide which
  residues are protected unless a future task explicitly reopens the mask rule.
- Do not infer MSA source authority from review figures, prose, or public
  Eco1 accessions that disagree with the ec86kit target sequence hash.
- Do not route inverse-folding design into `permuter`; `permuter` may consume
  explicit candidate intent later through a public handoff contract.
- Treat whole-protein ESMC pseudo-likelihood as outside the current
  panel-selection path.
- Do not promote candidates into RT-lnRNA construct subjects without the
  downstream study's explicit acceptance contract.
- Route renderer-neutral molecule roles and py3Dmol behavior through
  `molecular-structure-visualization`; route ChimeraX GUI, REST, pose, and
  capture work through `chimerax-structure-review`. Eco1 complex views use gold
  DNA and salmon RNA for both backbone and nucleotide representations, and
  protein-only surfaces start off in the notebook and use `0.65` alpha (`35%` ChimeraX transparency) when shown.

## Required Deliverables

- current phase and record posture
- canonical study record paths, the current `dnadesign.thread` adapter path, and
  the dev spec path for broader thread work
- declared dataset/artifact roots and whether they are planned or materialized
- next readiness blockers from the checked-in readiness checks
- selected route for residue masks, fold validation, panel selection, or
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
