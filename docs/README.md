## Documentation Index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-20

Start here when you need the next workflow, tool doc, or repository record.

### Start here

1. If this is a new machine, start with [Installation](installation.md).
2. Use [Workspace and dataset lookup](#workspace-and-dataset-lookup) when you need to see what already exists.
3. Use [Workflow routes](#workflow-routes) to choose the next task.
4. Use the [Runbook catalog](runbooks/README.md) when you need commands first.
5. Jump to [Tool docs](#tool-docs) when you already know which package owns the next step.
6. Use [System records](#system-records), [Operations](#operations), and [Maintainer references](#maintainer-references) for repo contracts, operational docs, and maintainer references.

DenseGen docs live under `densegen`, while the CLI entrypoint is `uv run dense --help`.

### Workspace and dataset lookup

Use this table when the first question is "what is available right now?" rather than "which workflow should I run?"

| Tool | What to inspect | First command | Next doc |
| --- | --- | --- | --- |
| `densegen` | packaged workspaces plus current output state | `uv run dense workspace list` | [DenseGen workspaces](../src/dnadesign/densegen/workspaces/README.md) |
| `construct` | packaged workspaces plus current output state | `uv run construct workspace list` | [Construct workspaces](../src/dnadesign/construct/workspaces/README.md) |
| `infer` | packaged workspaces plus current output state | `uv run infer workspace list` | [infer workspaces](../src/dnadesign/infer/workspaces/README.md) |
| `cluster` | packaged workspaces plus current output state | `uv run cluster workspace list` | [Cluster workspaces](../src/dnadesign/cluster/workspaces/README.md) |
| `usr` | datasets under one root, not workspaces | `uv run usr --help` to see the default root, then `uv run usr ls --root <usr-root>` | [USR CLI quickstart](../src/dnadesign/usr/docs/getting-started/cli-quickstart.md) |

The singular `cluster workspace ...` form matches the other workspace-owning tools. `cluster workspaces ...` still works.

### Quick notes

- workflows and runbooks are maintained task paths; tutorials and demos are examples first
- workspace discovery is tool-local; `usr` uses dataset roots rather than a workspace tree.
- reference pages are for flags, schemas, and artifact contracts once you know the next tool

### Workflow routes

#### Single-tool starts

Use these when one tool still owns the work and no shared USR handoff is involved.

| Need | Primary workflow | Verify next |
| --- | --- | --- |
| Design a sequence library in a workspace | [DenseGen docs overview](../src/dnadesign/densegen/docs/README.md) | Verify generated artifacts and metadata with [DenseGen outputs reference](../src/dnadesign/densegen/docs/reference/outputs.md). |
| Realize contextualized or multi-part DNA constructs into derived datasets | [Construct docs overview](../src/dnadesign/construct/docs/README.md) | Verify resulting lineage and sequence identity in [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md). |
| Run model inference and write outputs back to datasets | [Infer docs index](../src/dnadesign/infer/docs/README.md) | Verify write-back columns and types with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md). |
| Build SCC Evo2 infer GPU environment deterministically | [BU SCC install bootstrap](bu-scc/install.md#gpu-setup-and-verification-runbook) | Verify infer model capabilities with [infer SCC Evo2 GPU runbook](../src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md). |
| Operate Notify for local event watching and webhook setup | [Notify docs index](notify/README.md) | Verify mode and delivery contracts in [Notify command contracts](../src/dnadesign/notify/docs/reference/command-contracts.md). |
| Inspect Notify package internals or extension seams | [Notify package docs index](../src/dnadesign/notify/docs/README.md) | Verify module boundaries in [Notify maintainer architecture map](../src/dnadesign/notify/docs/dev/architecture.md). |

#### Cross-tool USR dataset flows

Use these when work already moves through a shared USR dataset, carried overlays, or infer-derived feature columns.

| Need | Primary workflow | Verify next |
| --- | --- | --- |
| Assemble multiple USR-backed inputs into one downstream source-of-truth flow, then hand off through construct and infer | [Multi-source source-of-truth assembly](../src/dnadesign/usr/docs/operations/multi-source-source-of-truth-assembly.md) | Verify carried overlays, construct lineage, and infer write-back contracts with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md) and [Infer docs](../src/dnadesign/infer/docs/README.md). |
| Consolidate construct realizations into one USR-backed source-of-truth dataset, then hand off to Infer | [Construct -> USR -> Infer source-of-truth runbook](../src/dnadesign/usr/docs/operations/construct-infer-source-of-truth-runbook.md) | Verify lineage plus downstream write-back contracts with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md) and [Infer docs](../src/dnadesign/infer/docs/README.md). |
| Understand the full DenseGen/manual/wildtype -> optional Construct -> Infer Evo2 -> Notify/Cluster/OPAL promoter route before choosing a branch | [Promoter Evo2 workflow journey](../src/dnadesign/usr/docs/operations/promoter-evo2-journey.md) | Continue to [Promoter characterization feature matrix](../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md) for the shared handoff. |
| Build one promoter feature dataset from DenseGen anchors, wildtype/manual promoters, optional construct-expanded contexts, and infer-derived representations | [Promoter characterization feature matrix](../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md) | Verify one explicit `infer__...` column is chosen as `X`, then continue to the exploratory [cluster workflow](../src/dnadesign/cluster/docs/workflows/exploratory-clustering.md) or the downstream [USR dataset with infer-derived X -> OPAL active learning](../src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md) workflow once OPAL is pointed at that dataset and `X`. |
| Sync iterative HPC outputs to local analysis safely | [USR workflow map](../src/dnadesign/usr/docs/operations/workflow-map.md) -> [USR HPC sync flow](../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md) | Verify transfer parity with [USR sync audit loop](../src/dnadesign/usr/docs/operations/sync-audit-loop.md). |
| Run cross-machine sync with stricter failure checks | [USR sync command contract](../src/dnadesign/usr/docs/operations/sync.md) | Verify sidecar and overlay fidelity with [USR sync fidelity drills](../src/dnadesign/usr/docs/operations/sync-fidelity-drills.md). |
| Chain DenseGen -> USR -> Infer -> USR updates | [Chained DenseGen and Infer sync runbook](../src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-runbook.md) | Verify downstream dataset state with [Infer docs](../src/dnadesign/infer/docs/README.md). |

#### Operations and infrastructure

Use these when the next step is orchestration, environment setup, or audit output rather than dataset mutation.

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
- Cross-tool source-of-truth and handoff rules live in [Architecture](../ARCHITECTURE.md#cross-tool-information-architecture), [Design](../DESIGN.md#information-architecture-invariants), and the shared USR runbooks linked above.
- Scheduler accumulation rules live in [Ops orchestration runbook contracts](operations/orchestration-runbooks.md#single-study-accumulation-contract).

### Operations

- [Installation](installation.md): environment setup and verification baseline.
- [Runbook catalog](runbooks/README.md): shell command index for cross-tool procedures and tool entrypoints.
- [Ops orchestration index](operations/README.md): orchestration docs for init, plan, execute, and status checks.
- [BU SCC docs index](bu-scc/README.md): cluster setup, submission, and operator runbooks.
- [Notify docs index](notify/README.md): notifier setup, runtime behavior, and recovery routes.
- [USR operations index](../src/dnadesign/usr/docs/operations/README.md): sync, transfer, and cross-tool source-of-truth runbooks for iterative workspace updates.

### Maintainer references

- [Developer docs](dev/README.md): CI/testing contracts and maintainer operations.
- [Execution plans index](exec-plans/README.md): active and completed execution-plan records.
- [Templates index](templates/README.md): reusable templates for runbooks, plans, and records.
- [Architecture decisions index](architecture/decisions/README.md): ADR catalog and decision history.
- [Quality docs index](quality/README.md): quality gates, audits, and measurement references.
