# Documentation Index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-19

This is the canonical docs entrypoint. Keep curated navigation here.

Quick entrypoint:
- [Start here](start-here.md): lightweight workflow-first navigation.

## Repository knowledge base

- [ARCHITECTURE.md](../ARCHITECTURE.md): high-level architecture map and boundary contracts.
- [DESIGN.md](../DESIGN.md): engineering principles and interface invariants.
- [SECURITY.md](../SECURITY.md): secrets and supply-chain handling policy.
- [RELIABILITY.md](../RELIABILITY.md): fail-fast model, observability, and operations map.
- [PLANS.md](../PLANS.md): proposal, execution-plan, and ADR lifecycle.
- [QUALITY_SCORE.md](../QUALITY_SCORE.md): quality rubric scaffold and improvement path.
- [AGENTS.md](../AGENTS.md): contributor task map and operational entrypoints.
- [docs/architecture/README.md](architecture/README.md): architecture-specific references and decisions.
- [docs/security/README.md](security/README.md): security references and runbooks.
- [docs/reliability/README.md](reliability/README.md): reliability references and runbooks.
- [docs/quality/README.md](quality/README.md): CI/coverage/docs quality references.
- [docs/exec-plans/README.md](exec-plans/README.md): execution plan index and conventions.
- [docs/templates/README.md](templates/README.md): reusable templates for docs, ADRs, and plans.

## Choose your path

### New contributor

- [Installation](installation.md)
- [Dependency maintenance](dependencies.md)
- [Notebooks quickstart](notebooks.md)

### SCC operator

- [BU SCC operations](bu-scc/README.md)
- [BU SCC quickstart](bu-scc/quickstart.md)
- [BU SCC batch + Notify runbook](bu-scc/batch-notify.md)

### Notify operator

- [Notify operations](notify/README.md)
- [Notify USR events operator manual](notify/usr-events.md)

### Maintainer

- [Developer docs index](dev/README.md)
- [Developer journal](dev/journal.md)
- [CI workflow](../.github/workflows/ci.yaml)

## Workflow index (task-first)

| Need | Primary workflow | Supporting references |
| --- | --- | --- |
| Need a task-first USR command chain before detailed runbooks | [USR workflow map](../src/dnadesign/usr/docs/operations/workflow-map.md) | [USR operations index](../src/dnadesign/usr/docs/operations/README.md), [USR sync contract](../src/dnadesign/usr/docs/operations/sync.md) |
| Need to sync iterative HPC batch outputs into local analysis | [USR HPC sync flow](../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md) | [USR sync contract](../src/dnadesign/usr/docs/operations/sync.md), [BU SCC quickstart](bu-scc/quickstart.md) |
| Need machine-readable sync decisions for chained runs | [USR sync audit loop](../src/dnadesign/usr/docs/operations/sync-audit-loop.md) | [USR sync contract](../src/dnadesign/usr/docs/operations/sync.md), [USR HPC sync flow](../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md) |
| Need a chained DenseGen -> USR -> Infer -> USR loop | [USR chained DenseGen+Infer demo](../src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md) | [DenseGen docs](../src/dnadesign/densegen/README.md), [Infer docs](../src/dnadesign/infer/README.md) |
| Need to pressure test sync fidelity and failure recovery | [USR sync fidelity drills](../src/dnadesign/usr/docs/operations/sync-fidelity-drills.md) | [USR sync contract](../src/dnadesign/usr/docs/operations/sync.md) |
| Need to audit USR event boundary integration | [Notify USR events operator manual](notify/usr-events.md) | [USR README event schema](../src/dnadesign/usr/README.md#event-log-schema) |

## Entrypoint ladders

- iterative HPC sync loop:
  1. [USR workflow map](../src/dnadesign/usr/docs/operations/workflow-map.md)
  2. [USR HPC sync flow](../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md)
  3. [USR sync audit loop](../src/dnadesign/usr/docs/operations/sync-audit-loop.md)
- DenseGen -> USR -> Infer -> USR chain:
  1. [USR workflow map](../src/dnadesign/usr/docs/operations/workflow-map.md)
  2. [USR chained DenseGen+Infer demo](../src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md)
  3. [Infer CLI and runtime contract](../src/dnadesign/infer/README.md)
- sync fidelity/adversarial diagnosis loop:
  1. [USR sync command contract](../src/dnadesign/usr/docs/operations/sync.md)
  2. [USR sync fidelity drills](../src/dnadesign/usr/docs/operations/sync-fidelity-drills.md)
  3. [USR sync audit loop](../src/dnadesign/usr/docs/operations/sync-audit-loop.md)

## Progressive workflows

- USR sync (quick -> advanced -> failure diagnosis): [src/dnadesign/usr/docs/operations/sync.md](../src/dnadesign/usr/docs/operations/sync.md)
- USR workflow map (task-first command chains): [src/dnadesign/usr/docs/operations/workflow-map.md](../src/dnadesign/usr/docs/operations/workflow-map.md)
- USR sync audit loop (machine-readable decisions + chained calls): [src/dnadesign/usr/docs/operations/sync-audit-loop.md](../src/dnadesign/usr/docs/operations/sync-audit-loop.md)
- USR HPC sync flow (preflight -> run -> verify): [src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md](../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md)
- USR chained DenseGen+Infer loop (batch -> pull -> infer -> push): [src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md](../src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md)
- BU SCC batch + Notify operations: [bu-scc/batch-notify.md](bu-scc/batch-notify.md)
- DenseGen SCC workflow reference: [src/dnadesign/densegen/docs/howto/bu-scc.md](../src/dnadesign/densegen/docs/howto/bu-scc.md)
- Infer CLI and runtime contract: [src/dnadesign/infer/README.md](../src/dnadesign/infer/README.md)

## Tool docs

| Tool | CLI | Docs |
| --- | --- | --- |
| `aligner` | n/a | [src/dnadesign/aligner/README.md](../src/dnadesign/aligner/README.md) |
| `baserender` | `uv run baserender --help` | [src/dnadesign/baserender/README.md](../src/dnadesign/baserender/README.md) |
| `billboard` | n/a | [src/dnadesign/billboard/README.md](../src/dnadesign/billboard/README.md) |
| `cluster` | `uv run cluster --help` | [src/dnadesign/cluster/README.md](../src/dnadesign/cluster/README.md) |
| `cruncher` | `uv run cruncher --help` | [src/dnadesign/cruncher/README.md](../src/dnadesign/cruncher/README.md) |
| `densegen` | `uv run dense --help` | [src/dnadesign/densegen/README.md](../src/dnadesign/densegen/README.md) |
| `infer` | `uv run infer --help` | [src/dnadesign/infer/README.md](../src/dnadesign/infer/README.md) |
| `latdna` | n/a | [src/dnadesign/latdna/README.md](../src/dnadesign/latdna/README.md) |
| `libshuffle` | n/a | [src/dnadesign/libshuffle/README.md](../src/dnadesign/libshuffle/README.md) |
| `nmf` | n/a | [src/dnadesign/nmf/README.md](../src/dnadesign/nmf/README.md) |
| `notify` | `uv run notify --help` | [src/dnadesign/notify/README.md](../src/dnadesign/notify/README.md) |
| `opal` | `uv run opal --help` | [src/dnadesign/opal/README.md](../src/dnadesign/opal/README.md) |
| `permuter` | `uv run permuter --help` | [src/dnadesign/permuter/README.md](../src/dnadesign/permuter/README.md) |
| `tfkdanalysis` | n/a | [src/dnadesign/tfkdanalysis/README.md](../src/dnadesign/tfkdanalysis/README.md) |
| `usr` | `uv run usr --help` | [src/dnadesign/usr/README.md](../src/dnadesign/usr/README.md) |
