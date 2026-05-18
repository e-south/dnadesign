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
    contract/
    runtime/
      pipeline.yaml
  routes/                   # one-hop router plus focused route details
    README.md
  contracts/                # status/preflight contracts and registry sidecars
  compiler/                 # study compiler inputs and normalization metadata
  contexts/                 # long-form rationale and handoff notes
  workbench/                # durable experimental meaning
    ontology/               # directions and effect tags
    design_sets/            # authoritative cohorts
    provenance/             # compiler/materialization run records
```

Use `routes/README.md` first for task routing. Use `record/status.md` only for
current state, `operations/ops.study.yaml` for lifecycle/preflight declarations,
`operations/runtime/pipeline.yaml` for command groups, `workbench/` when the
question is why a cohort exists or what experimental direction it tests, and
`compiler/` for Retron MSD label normalization or catalog compilation.
