## RegulonDB Native Promoter Panel Study

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This study keeps only the ontology router at the root. Durable factual state,
operating declarations, route maps, bindings, and audits live in typed
subdirectories.

### Directory Ontology

```text
regulondb_native_promoter_panel/
  README.md        # this directory ontology
  record/          # factual study record
    campaign.yaml
    datasets.yaml
    status.md
  operations/      # machine-readable operating contracts
    ops.study.yaml
    pipeline.yaml
  routes/          # one-hop router
    README.md
  bindings/        # cross-tool study bindings
  audits/          # typed sync/readiness evidence
    usr-sync/      # machine-readable USR sync evidence
```

Use `routes/README.md` first for owner-surface navigation. Use
`record/status.md` only for factual current state and `operations/` only for
Ops/pipeline declarations. Use `bindings/` when a tool needs durable study
context and `audits/usr-sync/` for machine-readable sync payloads.
