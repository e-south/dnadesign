## BU SCC Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28

This directory is the canonical source for BU SCC platform policy, submission vocabulary, and job templates for `dnadesign`.

### Choose by task

- First run on SCC from interactive shell to first batch submit: [BU SCC Quickstart](quickstart.md).
- Bootstrap environment setup and run diagnostics once per host: [BU SCC Install bootstrap](install.md).
- Run batch patterns, arrays, Notify deployment, and transfer-node flows: [BU SCC Batch + Notify runbook](batch-notify.md).
- Start from submit-ready scripts and override patterns: [BU SCC job templates](jobs/README.md).
- Launch web interactive sessions through SCC OnDemand: [BU SCC OnDemand sessions](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/sessions/).
- Use status-first queue checks and operator defaults: [BU SCC Agent Cheat Sheet](agent-cheatsheet.md).

### Semantic boundaries

- BU-specific scheduler keys, examples, and constraints belong in this `bu-scc/` directory.
- Execution flow, queue fairness guidance, and status-first checks are documented in `quickstart.md`, `batch-notify.md`, and `agent-cheatsheet.md`.
- Notify watcher semantics and onboarding belong in `../notify/usr-events.md`.

### Fast links

- Notify watcher setup and event-stream operations: [../notify/usr-events.md](../notify/usr-events.md).
- Large model and dataset transfer patterns on SCC: [batch-notify.md#7-large-downloads-and-datasetmodel-transfer](batch-notify.md#7-large-downloads-and-datasetmodel-transfer).
- SCC OnDemand entrypoint and service overview: [https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/).
