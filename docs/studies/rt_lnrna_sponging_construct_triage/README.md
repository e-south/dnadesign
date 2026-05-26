---
doc_id: study-rt-lnrna-sponging-construct-triage
surface: study-root
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-25
first_hop: routes/README.md
status_surface: record-only
preflight_surface: planned-contract-checks
---

## RT-lnRNA Sponging Construct Triage Study

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-25

This study is the checked-in Phase 0/1 record for synthetic RT-lnRNA expression
construct triage. It owns study framing, candidate-row semantics, source overlay
contracts, and OPAL training-dataset readiness rules. It does not own Construct assembly,
Infer feature extraction, LatentDNA materialization, or OPAL learning.

Use `routes/README.md` first for task routing. Use `record/status.md` for
current state and blockers, `record/datasets.yaml` for source inventories, and
`operations/ops.study.yaml` for the machine-readable contract index.
GenBank source authority lives in
`workbench/provenance/genbank-source-authority.yaml` with parsed offsets in
`workbench/provenance/genbank-feature-offset-audit.md`. The Phase 2a Construct
projection contract is the multi-slot projection manifest at
`operations/contract/fixtures/construct/construct-projection-manifest.yaml`.

### Directory Ontology

```text
rt_lnrna_sponging_construct_triage/
  README.md
  record/
    datasets.yaml
    status.md
  operations/
    ops.study.yaml
    contract/
      lifecycle/
      readiness/
        checks/
      schemas/
      fixtures/
      status/
      surfaces/
    runtime/
      command-groups/
  routes/
    README.md
  contexts/
    construct-contract.md
    permuter-onboarding.md
    representation-contract.md
    source-overlays.md
    opal-handoff.md
  workbench/
    ontology/
    design_sets/
    provenance/
```

Generated construct, Infer, LatentDNA, OPAL, and large construct-subject-table outputs
belong in explicit runtime workspaces, not in this record root.

### Candidate Expansion Contract

Study-owned candidate expansion emits Construct-compatible rows with
`construct_subject__lnrna_sequence` and `construct_subject__rt_cds_sequence`. In silico RT-CDS
DMS expansion must call the public `dnadesign.permuter` surface and keep
Permuter provenance in the study-owned `construct_subject__*` overlay fields. Construct
still consumes only the named `lnrna` and `rt_cds` slots. See
`contexts/permuter-onboarding.md` for the Permuter boundary and candidate
envelope rules.

Construct materialization is not considered Infer-ready until the executable
readiness gate in
`src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/infer_readiness.py`
passes: every construct subject must have one forward context, one
reverse-complement context, and the six explicit source sequence-view names
declared by `contexts/representation-contract.md`.
