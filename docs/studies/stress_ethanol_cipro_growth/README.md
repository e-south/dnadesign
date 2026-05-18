---
doc_id: study-stress-ethanol-cipro-growth
surface: study-root
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-05-18
first_hop: routes/README.md
status_surface: studies.stress-ethanol-cipro-growth.status
preflight_surface: studies.stress-ethanol-cipro-growth.preflight
---

## Stress Ethanol Cipro Growth Study

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This study keeps only the ontology router at the root. Durable factual state,
operating declarations, route maps, context bindings, and audits live in typed
subdirectories.

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
    catalog/       # OPS status/preflight catalog docs
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
  routes/          # one-hop router plus focused route details
    README.md
  contexts/        # durable cross-tool context
    latentdna/
      binding.yaml
  audits/          # typed sync/readiness evidence
    readiness/     # prose readiness and contract audits
    usr-sync/      # machine-readable USR sync evidence
```

Use `routes/README.md` first for owner-surface navigation. Use
`record/status.md` only for factual current state, `operations/ops.study.yaml`
as the one-hop contract index, and `operations/runtime/command-groups/pipeline.yaml`
for machine-loaded command groups. Use
`operations/runtime/command-groups/README.md` when a human or naive agent needs
the progressive-disclosure lane map before opening the full pipeline. Split
execution and readiness fragments under
`operations/contract/` by owner lane; do not flatten them back into one YAML
shelf. Use `operations/catalog/` when the task is status or readiness,
`contexts/latentdna/` when LatentDNA needs durable study context, and
`audits/readiness/` or `audits/usr-sync/` for evidence payloads.
