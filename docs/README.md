# Documentation Index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-19

This is the canonical docs entrypoint. Keep curated navigation here.

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

## Progressive workflows

- USR sync (quick -> advanced -> failure diagnosis): [src/dnadesign/usr/docs/operations/sync.md](../src/dnadesign/usr/docs/operations/sync.md)
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
