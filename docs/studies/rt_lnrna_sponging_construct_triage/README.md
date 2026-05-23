---
doc_id: study-rt-lnrna-sponging-construct-triage
surface: study-root
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-23
first_hop: routes/README.md
status_surface: record-only
preflight_surface: planned-contract-checks
---

## RT-lnRNA Sponging Construct Triage Study

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-23

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
    representation-contract.md
    source-overlays.md
    opal-handoff.md
  workbench/
    ontology/
    design_sets/
    provenance/
```

Generated construct, Infer, LatentDNA, OPAL, and large candidate-table outputs
belong in explicit runtime workspaces, not in this record root.
