## BU SCC Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This directory is the canonical source for BU SCC platform policy, submission vocabulary, and job templates for `dnadesign`.

### Directory ontology

- `setup/`: first-run SCC setup, bootstrap, and host/toolchain diagnostics.
- `runbooks/`: scheduler-managed execution procedures.
- `reference/`: queue, submission, and scheduler vocabulary.
- `jobs/`: submit-ready `qsub` templates and small smoke helpers.
- `fixtures/`: checked-in scheduler fixture text for docs and skill audits.

### Read order

1. First run on SCC from interactive shell to first batch submit: [BU SCC Quickstart](setup/quickstart.md).
2. Bootstrap environment setup and run diagnostics once per host: [BU SCC Install bootstrap](setup/install.md).
3. Build Evo2 infer GPU environment: [BU SCC install GPU setup runbook](setup/install.md#gpu-setup-and-verification-runbook).
4. Run batch patterns, arrays, Notify deployment, and transfer-node flows: [BU SCC Batch + Notify runbook](runbooks/batch-notify.md).
5. Start from submit-ready scripts and override patterns: [BU SCC job templates](jobs/README.md).
6. Use submission defaults and queue checks: [BU SCC submission reference](reference/submission.md).
7. Launch web interactive sessions through SCC OnDemand: [BU SCC OnDemand sessions](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/sessions/).

### Semantic boundaries

- BU-specific scheduler keys, examples, and constraints belong in this `bu-scc/` directory.
- Execution flow, queue fairness guidance, and status-first checks are documented in `setup/quickstart.md`, `runbooks/batch-notify.md`, and `reference/submission.md`.
- Notify watcher semantics and onboarding belong in `../notify/usr-events.md`.
- Repo-local Codex automation lives in `../../.agents/skills/sge-hpc-ops/`; treat it as an optional overlay, not the source of truth.

### Fast links

- Notify watcher setup and event-stream operations: [../notify/usr-events.md](../notify/usr-events.md).
- Large model and dataset transfer patterns on SCC: [runbooks/batch-notify.md#7-large-downloads-and-datasetmodel-transfer](runbooks/batch-notify.md#7-large-downloads-and-datasetmodel-transfer).
- Optional Codex repo-local skill overlay: [../../.agents/skills/sge-hpc-ops/SKILL.md](../../.agents/skills/sge-hpc-ops/SKILL.md).
- SCC OnDemand entrypoint and service overview: [https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/).
