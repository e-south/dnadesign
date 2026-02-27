## BU SCC Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-18

This directory is the canonical source for BU SCC platform policy, submission vocabulary, and job templates for `dnadesign`.

### Choose by task

- First run on SCC from interactive shell to first batch submit: [BU SCC Quickstart](quickstart.md).
- Bootstrap environment setup and run diagnostics once per host: [BU SCC Install bootstrap](install.md).
- Run batch patterns, arrays, Notify deployment, and transfer-node flows: [BU SCC Batch + Notify runbook](batch-notify.md).
- Start from submit-ready scripts and override patterns: [BU SCC job templates](jobs/README.md).
- Launch web interactive sessions through SCC OnDemand: [BU SCC OnDemand sessions](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/sessions/).
- Apply scheduler-generic execution contracts used by this repo: [SGE HPC Ops skill package](sge-hpc-ops/SKILL.md).

### Semantic boundaries

- BU-specific scheduler keys, examples, and constraints belong in this `bu-scc/` directory.
- Execution flow and probe-first operating rules belong in `sge-hpc-ops/SKILL.md`.
- Skill ownership model is BU-hosted generic core: scheduler-generic guidance is kept in the skill core, and BU policy remains in this `bu-scc/` directory.
- Notify watcher semantics and onboarding belong in `../notify/usr-events.md`.
- Skill package layout is intentionally minimal: `sge-hpc-ops/SKILL.md`, `sge-hpc-ops/agents/openai.yaml`, and `sge-hpc-ops/references/`.

### Fast links

- Notify watcher setup and event-stream operations: [../notify/usr-events.md](../notify/usr-events.md).
- Large model and dataset transfer patterns on SCC: [batch-notify.md#7-large-downloads-and-datasetmodel-transfer](batch-notify.md#7-large-downloads-and-datasetmodel-transfer).
- SCC OnDemand entrypoint and service overview: [https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/).
