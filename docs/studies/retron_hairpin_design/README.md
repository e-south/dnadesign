## Retron Hairpin Design Study

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This study keeps entrypoints at the root and pushes durable detail into typed
subdirectories.

### Directory Ontology

```text
retron_hairpin_design/
  status.md                 # current state and concise next actions
  routes.md                 # one-hop router across owner surfaces
  ops.study.yaml            # Ops lifecycle and artifact contract
  pipeline.yaml             # machine-readable command groups
  compiler/                 # study compiler inputs and normalization metadata
  contexts/                 # long-form rationale and handoff notes
  routes/                   # focused route details for owner surfaces
  workbench/                # hypotheses, design sets, and run provenance
```

Use `routes.md` first for task routing. Use `workbench/` when the question is
why a cohort exists or what experimental direction it tests. Use `compiler/`
when the task is Retron MSD label normalization or catalog compilation.
