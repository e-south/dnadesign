## Stress Ethanol Cipro Growth Study

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This study keeps only the ontology router at the root. Durable factual state,
operating declarations, route maps, bindings, contracts, and audits live in
typed subdirectories.

### Directory Ontology

```text
stress_ethanol_cipro_growth/
  README.md        # this directory ontology
  record/          # factual study record
    campaign.yaml
    datasets.yaml
    status.md
  operations/      # machine-readable operating contracts
    ops.study.yaml
    pipeline.yaml
  routes/          # one-hop router plus focused route details
    README.md
  contracts/       # status/preflight contracts and registry sidecars
  bindings/        # cross-tool study bindings
  audits/          # typed sync/readiness evidence
    readiness/     # prose readiness and contract audits
    usr-sync/      # machine-readable USR sync evidence
```

Use `routes/README.md` first for owner-surface navigation. Use
`record/status.md` only for factual current state and `operations/` only for
Ops/pipeline declarations. Use `contracts/` when the task is status or
readiness, `bindings/` when a tool needs durable study context, and
`audits/readiness/` or `audits/usr-sync/` for evidence payloads.
