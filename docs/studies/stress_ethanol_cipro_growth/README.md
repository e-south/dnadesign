---
doc_id: study-stress-ethanol-cipro-growth
surface: study-root
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-08-01
first_hop: routes/README.md
status_surface: studies.stress-ethanol-cipro-growth.status
preflight_surface: studies.stress-ethanol-cipro-growth.preflight
---

## Stress Ethanol Cipro Growth Study

This directory is the checked-in entry point for the stress promoter study. It
routes work; it does not duplicate measured data, objective mathematics, or
campaign state.

### Start here

1. Read the [route map](routes/README.md) to choose the owner and next artifact.
2. Read the [checked-in status](record/status.md) before relying on campaign or
   dataset state.
3. Use the status and preflight commands when current runtime evidence matters:

```bash
uv run ops progress show studies.stress-ethanol-cipro-growth.status --json
uv run ops progress show studies.stress-ethanol-cipro-growth.preflight \
  --scope next --command-timeout-seconds 30 --json
```

### Ownership boundaries

| Concern | Owner | First local surface |
| --- | --- | --- |
| Assay measurements, event alignment, reductions, and experiment plots | Reader | [Study route map](routes/README.md#reader-to-opal-path) |
| Promoter aliases, candidate and sequence identity, repeat decisions, and observation publication | Stress study | [Study source map](../../../src/dnadesign/studies/units/stress_ethanol_cipro_growth/README.md) |
| Candidate features and study-approved labels | Stress study contracts consumed by OPAL | [OPAL context](contexts/opal/README.md) |
| Objective mathematics, model fitting, scoring, and selection | OPAL | [OPAL route](routes/decision/opal/README.md) |
| Factual phase, accepted handoffs, and blockers | Checked-in study record | [Status](record/status.md) |

SFXI, RMF, and MSRB are objective or comparison semantics downstream of Reader
measurements. Their source-of-truth documents remain in the study and OPAL
surfaces linked above.

### Directory map

```text
stress_ethanol_cipro_growth/
  README.md        # this entry point
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
    promoter-design-intent.md
    latentdna/
      binding.yaml
  audits/          # typed sync/readiness evidence
    readiness/     # prose readiness and contract audits
    usr-sync/      # machine-readable USR sync evidence
```

Keep factual state in `record/`, executable declarations in `operations/`,
owner handoffs in `routes/`, durable interpretation in `contexts/`, and review
evidence in `audits/`. The [route map](routes/README.md) provides the one-jump
human path; `operations/ops.study.yaml` is the machine-readable contract index.
