## Documentation Index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

This is the central documentation map for workflows, tool references, and repository policy.

### Use this index

1. If this is a new machine, start with [Installation](installation.md) first.
2. Continue to [Workflow routes](#workflow-routes) and follow the preflight -> run -> verify sequence for the relevant outcome.
3. Follow the route's "Verify next" target before moving to downstream tools.
4. Use [Tool docs](#tool-docs) when you need package-level commands and data contracts.
5. Use [System records](#system-records), [Operations](#operations), and [Maintainer references](#maintainer-references) for policy, operations, and governance detail.
6. Return to this page as the central docs map.

### How docs are organized

- `route`: tells you where to go next.
- `runbook`: authoritative operator procedure with ordered commands and verification.
- `workflow`: downstream tool-owned branch after a handoff into that tool.
- `tutorial` or `demo`: sample or pedagogical path; use the linked runbook/workflow when you need the authoritative contract.
- The workflow routes below are grouped by the owner of the next deep procedure so you do not have to infer control-plane versus data-plane ownership from path names alone.

### Workflow routes

#### Single-tool starts

Choose this section when the next authoritative document is still package-local and you do not yet need a shared durable USR handoff.

| Need | Primary workflow | Verify next |
| --- | --- | --- |
| Design a sequence library in a workspace | [DenseGen docs overview](../src/dnadesign/densegen/docs/README.md) | Verify generated artifacts and metadata with [DenseGen outputs reference](../src/dnadesign/densegen/docs/reference/outputs.md). |
| Realize contextualized or multi-part DNA constructs into derived datasets | [construct docs overview](../src/dnadesign/construct/docs/README.md) | Verify resulting lineage and sequence identity in [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md). |
| Run model inference and write outputs back to datasets | [Infer docs index](../src/dnadesign/infer/docs/README.md) | Verify write-back columns and types with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md). |
| Build SCC Evo2 infer GPU environment deterministically | [BU SCC install bootstrap](bu-scc/install.md#gpu-setup-and-verification-runbook) | Verify infer model capabilities with [infer SCC Evo2 GPU runbook](../src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md). |
| Operate Notify for local event watching and webhook setup | [Notify docs index](notify/README.md) | Verify mode and delivery contracts in [Notify command contracts](../src/dnadesign/notify/docs/reference/command-contracts.md). |
| Inspect Notify package internals or extension seams | [Notify package docs index](../src/dnadesign/notify/docs/README.md) | Verify module boundaries in [Notify maintainer architecture map](../src/dnadesign/notify/docs/dev/architecture.md). |

#### Shared USR-backed data-plane flows

These routes hand off into authoritative USR-backed runbooks or downstream tool workflows after the USR handoff is already explicit.
Choose this section when the next durable artifact is a shared USR dataset, overlay namespace, or infer-annotated feature matrix.

| Need | Primary workflow | Verify next |
| --- | --- | --- |
| Assemble multiple USR-backed inputs into one downstream source-of-truth flow, then hand off through construct and infer | [Multi-source source-of-truth assembly](../src/dnadesign/usr/docs/operations/multi-source-source-of-truth-assembly.md) | Verify carried overlays, construct lineage, and infer write-back contracts with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md) and [Infer docs](../src/dnadesign/infer/docs/README.md). |
| Consolidate construct realizations into one USR-backed source-of-truth dataset, then hand off to Infer | [Construct -> USR -> Infer source-of-truth runbook](../src/dnadesign/usr/docs/operations/construct-infer-source-of-truth-runbook.md) | Verify lineage plus downstream write-back contracts with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md) and [Infer docs](../src/dnadesign/infer/docs/README.md). |
| Build one promoter feature matrix from DenseGen anchors, wildtype/manual promoters, optional construct-expanded contexts, and infer-derived representations | [Promoter characterization feature matrix](../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md) | Verify one explicit `infer__...` column is chosen as `X`, then branch into the exploratory [cluster workflow](../src/dnadesign/cluster/docs/workflows/exploratory-clustering.md) or the downstream [USR dataset with infer-derived X -> OPAL active learning](../src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md) workflow. |
| Sync iterative HPC outputs to local analysis safely | [USR workflow map](../src/dnadesign/usr/docs/operations/workflow-map.md) -> [USR HPC sync flow](../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md) | Verify transfer parity with [USR sync audit loop](../src/dnadesign/usr/docs/operations/sync-audit-loop.md). |
| Run cross-machine sync with stricter failure checks | [USR sync command contract](../src/dnadesign/usr/docs/operations/sync.md) | Verify sidecar and overlay fidelity with [USR sync fidelity drills](../src/dnadesign/usr/docs/operations/sync-fidelity-drills.md). |
| Chain DenseGen -> USR -> Infer -> USR updates | [Chained DenseGen and Infer sync runbook](../src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-runbook.md) | Verify downstream dataset state with [Infer docs](../src/dnadesign/infer/docs/README.md). |

#### Operations and infrastructure

These routes hand off into control-plane or environment runbooks rather than durable USR-backed data-plane procedures.
Choose this section when the next artifact is orchestration state, environment setup, or audit output rather than a durable USR dataset mutation.

| Need | Primary workflow | Verify next |
| --- | --- | --- |
| Run BU SCC batch jobs with notifications | [BU SCC batch + notify runbook](bu-scc/batch-notify.md) | Verify event delivery contract in [Notify USR events contract](notify/usr-events.md). |
| Plan and execute deterministic DenseGen/Infer HPC orchestration runbooks | [Ops orchestration index](operations/README.md) | Verify command ordering and outcomes in [orchestration audit contract](operations/orchestration-runbooks.md#contract-rules). |

### Tool docs

| Tool | CLI | Docs |
| --- | --- | --- |
| `aligner` | n/a | [aligner README](../src/dnadesign/aligner/README.md) |
| `baserender` | `uv run baserender --help` | [baserender README](../src/dnadesign/baserender/README.md) |
| `billboard` | n/a | [billboard README](../src/dnadesign/billboard/README.md) |
| `cluster` | `uv run cluster --help` | [cluster README](../src/dnadesign/cluster/README.md) |
| `cruncher` | `uv run cruncher --help` | [cruncher README](../src/dnadesign/cruncher/README.md) |
| `densegen` | `uv run dense --help` | [densegen README](../src/dnadesign/densegen/README.md) |
| `construct` | `uv run construct --help` | [construct README](../src/dnadesign/construct/README.md) |
| `infer` | `uv run infer --help` | [infer README](../src/dnadesign/infer/README.md) |
| `latdna` | n/a | [latdna README](../src/dnadesign/latdna/README.md) |
| `libshuffle` | n/a | [libshuffle README](../src/dnadesign/libshuffle/README.md) |
| `nmf` | n/a | [nmf README](../src/dnadesign/nmf/README.md) |
| `ops` | `uv run ops --help` | [ops README](../src/dnadesign/ops/README.md) |
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
- Cross-tool data-plane source-of-truth contract lives in [Architecture](../ARCHITECTURE.md#cross-tool-information-architecture), [Design](../DESIGN.md#information-architecture-invariants), [Multi-source source-of-truth assembly](../src/dnadesign/usr/docs/operations/multi-source-source-of-truth-assembly.md), [Construct -> USR -> Infer source-of-truth runbook](../src/dnadesign/usr/docs/operations/construct-infer-source-of-truth-runbook.md), and [Promoter characterization feature matrix](../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md).
- Adjacent control-plane accumulation policy for scheduler loops lives in [Ops orchestration runbook contracts](operations/orchestration-runbooks.md#single-study-accumulation-contract); Ops does not own the construct-led source-of-truth data-plane procedure.

### Operations

- [Installation](installation.md): environment setup and verification baseline.
- [Ops orchestration index](operations/README.md): task-first control-plane runbook planning and execution routes.
- [BU SCC docs index](bu-scc/README.md): cluster setup, submission, and operator runbooks.
- [Notify docs index](notify/README.md): notifier setup, runtime behavior, and recovery routes.
- [USR operations index](../src/dnadesign/usr/docs/operations/README.md): sync, transfer, and cross-tool source-of-truth runbooks for iterative workspace updates.

### Maintainer references

- [Developer docs](dev/README.md): CI/testing contracts and maintainer operations.
- [Execution plans index](exec-plans/README.md): active and completed execution-plan records.
- [Templates index](templates/README.md): reusable templates for runbooks, plans, and records.
- [Architecture decisions index](architecture/decisions/README.md): ADR catalog and decision history.
- [Quality docs index](quality/README.md): quality gates, audits, and measurement references.
