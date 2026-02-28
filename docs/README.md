## Documentation Index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28

This is the central documentation map for workflows, tool references, and repository policy.

### Use this index

1. If this is a new machine, start with [Installation](installation.md) first.
2. Continue to [Workflow lanes](#workflow-lanes) to choose a preflight -> run -> verify path by outcome.
3. Follow the lane's "Verify next" target before moving to downstream tools.
4. Use [Tool docs](#tool-docs) when you need package-level commands and data contracts.
5. Use [System records](#system-records), [Operations](#operations), and [Maintainer references](#maintainer-references) for policy, operations, and governance detail.
6. Return to this page as the single docs entrypoint.

### Workflow lanes

| Need | Primary workflow | Verify next |
| --- | --- | --- |
| Design a sequence library in a workspace | [DenseGen docs overview](../src/dnadesign/densegen/docs/README.md) | Verify generated artifacts and metadata with [DenseGen outputs reference](../src/dnadesign/densegen/docs/reference/outputs.md). |
| Run model inference and write outputs back to datasets | [Infer README](../src/dnadesign/infer/README.md) | Verify write-back columns and types with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md). |
| Operate Notify for local event watching and webhook setup | [Notify docs index](notify/README.md) | Verify event stream contract in [Notify USR events contract](notify/usr-events.md). |
| Sync iterative HPC outputs to local analysis safely | [USR workflow map](../src/dnadesign/usr/docs/operations/workflow-map.md) -> [USR HPC sync flow](../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md) | Verify transfer parity with [USR sync audit loop](../src/dnadesign/usr/docs/operations/sync-audit-loop.md). |
| Run cross-machine sync with stricter failure checks | [USR sync command contract](../src/dnadesign/usr/docs/operations/sync.md) | Verify sidecar and overlay fidelity with [USR sync fidelity drills](../src/dnadesign/usr/docs/operations/sync-fidelity-drills.md). |
| Chain DenseGen -> USR -> Infer -> USR updates | [Chained workflow demo](../src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md) | Verify downstream dataset state with [Infer docs](../src/dnadesign/infer/README.md). |
| Run BU SCC batch jobs with notifications | [BU SCC batch + notify runbook](bu-scc/batch-notify.md) | Verify event delivery contract in [Notify USR events contract](notify/usr-events.md). |

### Tool docs

| Tool | CLI | Docs |
| --- | --- | --- |
| `aligner` | n/a | [aligner README](../src/dnadesign/aligner/README.md) |
| `baserender` | `uv run baserender --help` | [baserender README](../src/dnadesign/baserender/README.md) |
| `billboard` | n/a | [billboard README](../src/dnadesign/billboard/README.md) |
| `cluster` | `uv run cluster --help` | [cluster README](../src/dnadesign/cluster/README.md) |
| `cruncher` | `uv run cruncher --help` | [cruncher README](../src/dnadesign/cruncher/README.md) |
| `densegen` | `uv run dense --help` | [densegen README](../src/dnadesign/densegen/README.md) |
| `infer` | `uv run infer --help` | [infer README](../src/dnadesign/infer/README.md) |
| `latdna` | n/a | [latdna README](../src/dnadesign/latdna/README.md) |
| `libshuffle` | n/a | [libshuffle README](../src/dnadesign/libshuffle/README.md) |
| `nmf` | n/a | [nmf README](../src/dnadesign/nmf/README.md) |
| `notify` | `uv run notify --help` | [notify README](../src/dnadesign/notify/README.md) |
| `opal` | `uv run opal --help` | [opal README](../src/dnadesign/opal/README.md) |
| `permuter` | `uv run permuter --help` | [permuter README](../src/dnadesign/permuter/README.md) |
| `tfkdanalysis` | n/a | [tfkdanalysis README](../src/dnadesign/tfkdanalysis/README.md) |
| `usr` | `uv run usr --help` | [usr README](../src/dnadesign/usr/README.md) |

### System records

- [Architecture](../ARCHITECTURE.md): component boundaries and cross-tool integration map.
- [Design](../DESIGN.md): engineering invariants and interface expectations.
- [Reliability](../RELIABILITY.md): operational model, failure posture, and recovery contracts.
- [Security](../SECURITY.md): secrets, dependency, and supply-chain handling policy.
- [Plans](../PLANS.md): proposal, execution-plan, and decision lifecycle.
- [Quality score](../QUALITY_SCORE.md): quality rubric and improvement framework.

### Operations

- [Installation](installation.md): environment setup and verification baseline.
- [BU SCC docs index](bu-scc/README.md): cluster setup, submission, and operations runbooks.
- [Notify docs index](notify/README.md): notifier setup, runtime behavior, and operator guidance.
- [Notify USR event contract](notify/usr-events.md): USR `.events.log` consumption contract for downstream notifications.

### Maintainer references

- [Developer docs](dev/README.md): CI/testing contracts and maintainer operations.
- [Execution plans index](exec-plans/README.md): active and completed execution-plan records.
- [Templates index](templates/README.md): reusable templates for runbooks, plans, and records.
- [Architecture decisions index](architecture/decisions/README.md): ADR catalog and decision history.
- [Quality docs index](quality/README.md): quality gates, audits, and measurement references.
