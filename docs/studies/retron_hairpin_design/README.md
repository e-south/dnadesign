---
doc_id: study-retron-hairpin-design
surface: study-root
study_id: retron_hairpin_design
owner: dnadesign-maintainers
last_verified: 2026-05-18
first_hop: routes/README.md
status_surface: studies.retron-hairpin-design.status
preflight_surface: studies.retron-hairpin-design.preflight
---

## Retron Hairpin Design Study

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This study keeps only the ontology router at the root. Durable record,
operation, route, compiler, context, and workbench surfaces live in typed
subdirectories.

### Directory Ontology

```text
retron_hairpin_design/
  README.md                 # this directory ontology
  record/                   # factual study record
    campaign.yaml
    datasets.yaml
    status.md
  operations/               # machine-readable operating contracts
    ops.study.yaml
    catalog/                # OPS status/preflight catalog docs
    contract/
      lifecycle/
      surfaces/
        execution/
          commands/
      status/
      readiness/
        checks/
    runtime/
      command-groups/
        README.md
        pipeline.yaml
        lanes/
  routes/                   # one-hop router plus focused route details
    README.md
  compiler/                 # study compiler inputs and normalization metadata
  contexts/                 # long-form rationale and handoff notes
  workbench/                # durable experimental meaning
    ontology/               # directions and effect tags
    design_sets/            # authoritative cohorts
    provenance/             # compiler/materialization run records
```

Use `routes/README.md` first for task routing. Use `record/status.md` only for
current state, `operations/ops.study.yaml` for lifecycle/preflight declarations,
`operations/runtime/command-groups/README.md` for command-group lane routing,
`operations/runtime/command-groups/pipeline.yaml` for the machine-loaded command
payload, `workbench/` when the question is why a cohort exists or what
experimental direction it tests, and `compiler/` for Retron MSD label
normalization or catalog compilation.
