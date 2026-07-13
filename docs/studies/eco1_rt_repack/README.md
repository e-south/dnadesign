---
doc_id: study-eco1-rt-repack
surface: study-root
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-07-13
first_hop: routes/README.md
status_surface: record-only
preflight_surface: runtime-and-contract-checks
---

## Eco1 RT Repack Study

This directory owns the Eco1/Ec86 RT fixed-backbone sequence-design record.
Start with `routes/README.md`. Use `record/status.md` for current counts and
selected candidates, and `record/datasets.yaml` for artifact locations.

### Premise

This study asks whether complete ProteinMPNN-designed Eco1/Ec86 RT sequences
can keep declared catalytic, direct-contact, Wang thumb-track, and mapped
residues 255-311 fixed, preserve local C-alpha backbone geometry, and introduce
MSA-observed, non-acidifying substitutions in the declared peripheral
nucleic-acid-facing shell for a mutation-set-diverse experimental panel. Final
selection reports R13 and other alpha-1 substitutions without treating them as
functional gates.

The output is a set of protein hypotheses for testing. It is not evidence of
improved activity, affinity, processivity, strand displacement, or safety.

### Method

1. Use `7V9U` to define Eco1 residue numbering and retained DNA/RNA geometry.
2. Fix the `NAxxH`, `YADD`, and `VTG` contexts, direct retained DNA/RNA contacts,
   Wang thumb-track positions, mapped residues `255-311`, and declared
   conserved/core positions.
3. Generate complete sequences under one of three v3 policies: distal only,
   peripheral only, or peripheral plus distal. Never combine mutations from
   separate ProteinMPNN outputs.
4. Review each ColabFold model after one global mapped C-alpha fit. Apply one
   declared `2.5 A` local RMSD cutoff to every non-distal review region.
5. Keep passing rows in the distal, peripheral, or combined design group.
6. Select two distal, three peripheral, and three combined sequences. Within
   each group, minimize overlap in mutated positions first and exact
   substitutions second; use chemistry, MSA support, and structure metrics only
   for later ties.

Charge events, MSA support, fold metrics, local RMSD, and mutation count are
review fields and late tie-breakers. They are not activity scores. ESMC and SAE
are optional model checks and do not select rows.

The v3 global no-cysteine rule forced the open WT residue C233 to change in
proximal-policy sequences. C233 was not protected. Its recurrence is disclosed
as shared generation bias, not treated as functional evidence.

### Ownership

- `eco1_rt_repack` owns Eco1 structure, mask, chemistry, conservation, and
  panel-selection semantics.
- `dnadesign.thread.adapters.proteinmpnn` owns generic ProteinMPNN request and
  sample-ingest mechanics.
- `dnadesign.thread.candidates` owns generic candidate-table construction.
- `dnadesign.thread.foldcheck` and
  `dnadesign.thread.adapters.colabfold` own generic fold request and output
  normalization contracts.
- `dnadesign.thread.structure_views` owns the browser structure-view contract.
- Downstream RT-lnRNA studies own construct pairing and experimental acceptance.

Generated candidate, fold, selection, and notebook artifacts remain under the
workspace `outputs/` tree. Do not hand-edit them; change the materializer and
regenerate.

### Canonical Surfaces

```text
eco1_rt_repack/
  README.md
  record/
    campaign.yaml
    datasets.yaml
    status.md
  routes/
    README.md
  contexts/
    fixed-backbone-method.md
    msa-method.md
    residue-mask-policy.md
    fold-validation-policy.md
    selection-hardening-dev-spec.md
    generation-policy-cleanup-dev-spec.md
  operations/
    ops.study.yaml
    contract/
    runtime/
  workbench/
    ontology/
    provenance/
```
