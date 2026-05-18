## RegulonDB Native Promoter Panel Study

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This study keeps only the ontology router at the root. Durable factual state,
operating declarations, route maps, context bindings, and audits live in typed
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
    contract/
      lifecycle/
      surfaces/
      status/
      readiness/
    runtime/
      command-groups/
        pipeline.yaml
  routes/          # one-hop router
    README.md
  contexts/        # durable cross-tool context
    latentdna/
      binding.yaml
  audits/          # typed sync/readiness evidence
    usr-sync/usr/  # machine-readable USR sync evidence
```

Use `routes/README.md` first for owner-surface navigation. Use
`record/status.md` only for factual current state, `operations/ops.study.yaml`
for lifecycle/preflight declarations, and `operations/runtime/command-groups/pipeline.yaml`
for command groups. Use `contexts/latentdna/` when LatentDNA needs durable study
context and `audits/usr-sync/usr/` for machine-readable sync payloads.
