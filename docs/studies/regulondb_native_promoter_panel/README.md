---
doc_id: study-regulondb-native-promoter-panel
surface: study-root
study_id: regulondb_native_promoter_panel
owner: dnadesign-maintainers
last_verified: 2026-05-18
first_hop: routes/README.md
status_surface: record-only
preflight_surface: owner-tool-route-pages
---

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
    catalog/       # providerless OPS marker; no status/preflight provider
    contract/
      lifecycle/
      surfaces/
        execution/
          runbooks/
          commands/
      status/
      readiness/
        checks/
    runtime/
      command-groups/
        README.md
        pipeline.yaml
        lanes/
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
as the one-hop contract index, and `operations/runtime/command-groups/pipeline.yaml`
for machine-loaded command groups. Use
`operations/runtime/command-groups/README.md` for the owner-lane map before
opening the full pipeline. Split execution and readiness fragments under
`operations/contract/` by owner lane; do not flatten them back into one YAML
shelf. This study has no registered OPS status/preflight provider today, so
use checked-in record files and the route map for status. Use
`contexts/latentdna/` when LatentDNA needs durable study context and
`audits/usr-sync/usr/` for machine-readable sync payloads.
