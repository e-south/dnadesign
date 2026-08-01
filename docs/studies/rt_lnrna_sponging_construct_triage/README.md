---
doc_id: study-rt-lnrna-sponging-construct-triage
surface: study-root
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-08-01
first_hop: routes/README.md
status_surface: record-only
preflight_surface: planned-contract-checks
---

## RT-lnRNA Sponging Construct Triage Study

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-30

Start with the [route map](routes/README.md). For reporter-response work, the
plain path is bridge admission, study-owned subject binding, the
[reporter-response meta-study](contexts/reporter-response-metastudy/README.md),
then objective readiness. Each step has one owner and passes a typed artifact;
an acquisition never becomes a biological replicate unless Reader declares a
replicate identifier.

This study owns its framing, candidate-row semantics, source overlays, and
downstream readiness rules. It does not own Construct assembly, Infer feature
extraction, LatentDNA materialization, or OPAL learning. The route map also
sends sequence, structure, representation, and construct questions to their
owners in one jump. Use
`contexts/construct-overview.md` for the study primer and
`contexts/retron-tf-decoy-design-logic.md` for cross-study manuscript semantics.
Use `record/status.md` for current state and blockers,
`record/datasets.yaml` for source inventories, and
`operations/ops.study.yaml` for the machine-readable contract index.
GenBank source authority lives in
`workbench/provenance/genbank-source-authority.yaml` with parsed offsets in
`workbench/provenance/genbank-feature-offset-audit.md`. The Phase 2a Construct
projection contract is the multi-slot projection manifest at
`operations/contract/fixtures/construct/construct-projection-manifest.yaml`.
Exact RT-lnRNA component compositions and Reader aliases are resolved through
`workbench/provenance/subject_bindings/README.md`. Verified Reader records join
to those identities through the separate, measurement-free
`workbench/provenance/reader_evidence_bindings/README.md` contract.

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
    construct-overview.md
    construct-contract.md
    retron-tf-decoy-design-logic.md
    permuter-onboarding.md
    representation-contract.md
    source-overlays.md
    opal-handoff.md
  workbench/
    ontology/
    design_sets/
    figure_mocks/
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
