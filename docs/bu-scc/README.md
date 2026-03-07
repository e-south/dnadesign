## BU SCC Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-07

This directory is the canonical source for BU SCC platform policy, submission vocabulary, and job templates for `dnadesign`.

### Read order

1. First run on SCC from interactive shell to first batch submit: [BU SCC Quickstart](quickstart.md).
2. Bootstrap environment setup and run diagnostics once per host: [BU SCC Install bootstrap](install.md).
3. Build Evo2 infer GPU environment: [BU SCC install GPU setup runbook](install.md#gpu-setup-and-verification-runbook).
4. Run batch patterns, arrays, Notify deployment, and transfer-node flows: [BU SCC Batch + Notify runbook](batch-notify.md).
5. Start from submit-ready scripts and override patterns: [BU SCC job templates](jobs/README.md).
6. Use submission defaults and queue checks: [BU SCC submission reference](submission-reference.md).
7. Launch web interactive sessions through SCC OnDemand: [BU SCC OnDemand sessions](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/sessions/).

### Semantic boundaries

- BU-specific scheduler keys, examples, and constraints belong in this `bu-scc/` directory.
- Execution flow, queue fairness guidance, and status-first checks are documented in `quickstart.md`, `batch-notify.md`, and `submission-reference.md`.
- Notify watcher semantics and onboarding belong in `../notify/usr-events.md`.

### Fast links

- Notify watcher setup and event-stream operations: [../notify/usr-events.md](../notify/usr-events.md).
- Large model and dataset transfer patterns on SCC: [batch-notify.md#7-large-downloads-and-datasetmodel-transfer](batch-notify.md#7-large-downloads-and-datasetmodel-transfer).
- SCC OnDemand entrypoint and service overview: [https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/).
