---
doc_id: study-eco1-rt-repack
surface: study-root
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-20
first_hop: routes/README.md
status_surface: record-only
preflight_surface: planned-contract-checks
---

## Eco1 RT Repack Study

This study is the checked-in planning surface for Eco1 reverse-transcriptase
fixed-backbone redesign through the planned `thread` tool contract.

Eco1 is the first anchor/profile. The reusable ontology is fixed-backbone
protein sequence design, not Eco1 and not reverse transcriptases in general.

Use `routes/README.md` first. Current state is in `record/status.md`; source and
artifact posture are in `record/datasets.yaml`; machine-readable planning
surfaces are under `operations/`; durable rationale and policy live in
`contexts/`; design-set meaning and provenance live in `workbench/`.

This study intentionally separates three layers:

| Layer | Owner | Examples |
| --- | --- | --- |
| Study biology | `eco1_rt_repack` | Eco1 structure authority, catalytic masks, retron motif protection, first candidate-batch policy. |
| Reusable fixed-backbone mechanics | planned `thread` | Residue maps, mask algebra, candidate ids, fold-check normalization, handoff hash closure. |
| Downstream construct realization | `rt_lnrna_sponging_construct_triage` and `construct` | Pairing an accepted RT with lnRNA/TF-sponging construct subjects. |

The record must fail visibly when a layer is missing. It should not hide missing
structure authority, residue numbering, fold metrics, or downstream acceptance
behind fallback prose.

### Directory Ontology

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
    implementation-roadmap.md
    residue-mask-policy.md
    fold-validation-policy.md
    synthesis-feasibility-policy.md
  operations/
    ops.study.yaml
    contract/
      fixtures/
      lifecycle/
      readiness/
      schemas/
      status/
      surfaces/
    runtime/
      command-groups/
  workbench/
    ontology/
    design_sets/
    provenance/
      conservation-source-discovery.md
      conservation-sources.yaml
      residue-numbering-policy.yaml
      structure-sources.yaml
```

Generated MPNN samples, fold predictions, large candidate tables, and runtime
sidecars do not belong in this checked-in record root.

### Naming Rules

- Use `eco1_rt_repack` for the checked-in study id.
- Use `eco1-rt-repack` only for human-facing slugs, skills, or dated plan
  filenames where kebab case is already the local convention.
- Use `eco1_rt_v1` for the first Eco1 profile id.
- Use neutral artifact names such as `mask_set.yaml` and
  `candidate_handoff.yaml` for reusable `thread` contracts.
- Do not encode backend names into candidate ids; backend provenance belongs in
  candidate rows and upstream hashes.
- Do not create `permuter__var_id` or RT-lnRNA construct-subject ids until a
  later explicit handoff contract owns that promotion.

### Implementation Boundary

The next code slice should validate and materialize the artifact chain, not run
ProteinMPNN, LigandMPNN, AlphaFold, or ColabFold directly. `thread` owns
fixed-backbone design contracts and result normalization. `infer` may later own
backend process execution if an explicit adapter contract is added. Study code
under `src/dnadesign/studies/units/eco1_rt_repack/` may summarize status and
validate Eco1-specific policy, but generic residue maps, masks, candidate ids,
fold-check normalization, and handoff hash closure belong in `thread`.
