# BU SCC Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-18

This directory is the canonical source for BU SCC platform policy, submission vocabulary, and job templates for `dnadesign`.

## Choose by task

- First run on SCC (interactive to first batch submit): [BU SCC Quickstart](quickstart.md)
- One-time environment bootstrap and diagnostics: [BU SCC Install bootstrap](install.md)
- Batch patterns, arrays, Notify deployment, and transfer-node usage: [BU SCC Batch + Notify runbook](batch-notify.md)
- Submit-ready scripts and override examples: [BU SCC job templates](jobs/README.md)
- Web interactive sessions and portal workflows: [BU SCC OnDemand sessions](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/sessions/)
- Scheduler-generic execution contract: [SGE HPC Ops skill package](sge-hpc-ops/SKILL.md)

## Semantic boundaries

- BU-specific scheduler keys, examples, and constraints belong in this `bu-scc/` directory.
- Execution flow and probe-first operating rules belong in `sge-hpc-ops/SKILL.md`.
- Skill ownership model is BU-hosted generic core: scheduler-generic guidance is kept in the skill core, and BU policy remains in this `bu-scc/` directory.
- Notify watcher semantics and onboarding belong in `../notify/usr-events.md`.
- Skill package layout is intentionally minimal: `sge-hpc-ops/SKILL.md`, `sge-hpc-ops/agents/openai.yaml`, and `sge-hpc-ops/references/`.

## Fast links

- Notify USR events operator manual: [../notify/usr-events.md](../notify/usr-events.md)
- Large model and dataset transfer guidance: [batch-notify.md#7-large-downloads-and-datasetmodel-transfer](batch-notify.md#7-large-downloads-and-datasetmodel-transfer)
- BU SCC OnDemand overview: [https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/)
