## Documentation Index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-21

Use this page to find the next task, command, or reference doc.

Start with:

- [Installation](installation.md) to set up the repo on a new machine.
- [Inspect available work](#inspect-available-work) to list packaged workspaces or dataset roots before choosing a workflow.
- [Choose a workflow](#choose-a-workflow) to follow a single-tool path, a cross-tool dataset path, or an operations path.
- [Runbook catalog](runbooks/README.md) when you want commands before longer docs.
- [Tool docs](#tool-docs) when you already know the package.
- [System records](#system-records), [Operations](#operations), and [Maintainer references](#maintainer-references) for repo contracts and operator docs.

### Inspect available work

Use this table when the first question is "what is available right now?" rather than "which workflow should I run?"

| Tool | What it owns | First command | Next doc |
| --- | --- | --- | --- |
| `densegen` | packaged workspaces and their current output state | `uv run dense workspace list` | [DenseGen workspaces](../src/dnadesign/densegen/workspaces/README.md) |
| `construct` | packaged workspaces and their current output state | `uv run construct workspace list` | [Construct workspaces](../src/dnadesign/construct/workspaces/README.md) |
| `infer` | packaged workspaces and their current output state | `uv run infer workspace list` | [infer workspaces](../src/dnadesign/infer/workspaces/README.md) |
| `cluster` | packaged workspaces and their current output state | `uv run cluster workspace list` | [Cluster workspaces](../src/dnadesign/cluster/workspaces/README.md) |
| `usr` | dataset roots and datasets, not workspaces | `uv run usr --help` to see the default root, then `uv run usr ls --root <usr-root>` | [USR CLI quickstart](../src/dnadesign/usr/docs/getting-started/cli-quickstart.md) |

### Choose a workflow

#### Single-tool workflows

Use these when one tool owns the next step.

| Need | Primary workflow | Verify next |
| --- | --- | --- |
| Design a sequence library in a workspace | [DenseGen docs overview](../src/dnadesign/densegen/docs/README.md) | Verify generated artifacts and metadata with [DenseGen outputs reference](../src/dnadesign/densegen/docs/reference/outputs.md). |
| Build DNA constructs from templates or multiple parts into derived datasets | [Construct docs overview](../src/dnadesign/construct/docs/README.md) | Verify resulting lineage and sequence identity in [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md). |
| Run model inference and write outputs back to datasets | [Infer docs index](../src/dnadesign/infer/docs/README.md) | Verify write-back columns and types with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md). |
| Set up the SCC Evo2 infer GPU environment | [BU SCC install bootstrap](bu-scc/install.md#gpu-setup-and-verification-runbook) | Verify infer model capabilities with [infer SCC Evo2 GPU runbook](../src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md). |
| Run Notify for local event watching and webhook setup | [Notify docs index](notify/README.md) | Verify mode and delivery contracts in [Notify command contracts](../src/dnadesign/notify/docs/reference/command-contracts.md). |
| Inspect or extend Notify package internals | [Notify package docs index](../src/dnadesign/notify/docs/README.md) | Verify module boundaries in [Notify maintainer architecture map](../src/dnadesign/notify/docs/dev/architecture.md). |

#### Cross-tool dataset workflows

Use these when data moves through more than one tool and the shared record lives in USR.

| Need | Primary workflow | Verify next |
| --- | --- | --- |
| Assemble multiple producer datasets before construct and infer share one downstream dataset | [Multi-source shared dataset assembly](../src/dnadesign/usr/docs/operations/multi-source-shared-dataset-assembly.md) | Verify carried overlays, construct lineage, and infer write-back contracts with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md) and [Infer docs](../src/dnadesign/infer/docs/README.md). |
| Hand one construct-backed dataset to infer and downstream watchers | [Construct -> USR -> Infer shared dataset runbook](../src/dnadesign/usr/docs/operations/construct-infer-shared-dataset-runbook.md) | Verify lineage plus downstream write-back contracts with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md) and [Infer docs](../src/dnadesign/infer/docs/README.md). |
| Review the promoter-study Evo2 route before choosing a branch | [Promoter study Evo2 workflow journey](../src/dnadesign/usr/docs/operations/promoter-evo2-journey.md) | Continue to [Promoter characterization feature matrix](../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md) for the shared handoff. |
| Keep one real promoter-study status surface that naive agents can refresh | [Promoter study status contract](../src/dnadesign/usr/docs/operations/promoter-study-status-contract.md) | Verify the live study with `ops progress campaign`, `ops progress show usr.data-plane.promoter-feature-matrix`, and the sync-aware study record under [Study records](studies/README.md). |
| Build a promoter-study feature dataset from anchors, wildtype/manual promoters, optional construct contexts, and infer outputs | [Promoter characterization feature matrix](../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md) | Verify one explicit `infer__...` column is chosen as `X` or export a flattened matrix before continuing to the exploratory [cluster workflow](../src/dnadesign/cluster/docs/workflows/exploratory-clustering.md) or the downstream [USR dataset with infer-derived X -> OPAL active learning](../src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md) workflow. |
| Sync iterative HPC outputs to local analysis safely | [USR workflow map](../src/dnadesign/usr/docs/operations/workflow-map.md) -> [USR HPC sync flow](../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md) | Verify transfer parity with [USR sync audit loop](../src/dnadesign/usr/docs/operations/sync-audit-loop.md). |
| Run cross-machine sync with stricter failure checks | [USR sync command contract](../src/dnadesign/usr/docs/operations/sync.md) | Verify sidecar and overlay fidelity with [USR sync fidelity drills](../src/dnadesign/usr/docs/operations/sync-fidelity-drills.md). |
| Chain DenseGen -> USR -> Infer -> USR updates | [Chained DenseGen and Infer sync runbook](../src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-runbook.md) | Verify downstream dataset state with [Infer docs](../src/dnadesign/infer/docs/README.md). |

#### Scheduler and environment workflows

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
- Cross-tool handoff rules live in [Architecture](../ARCHITECTURE.md#cross-tool-information-architecture), [Design](../DESIGN.md#information-architecture-invariants), and the shared dataset workflows linked above.
- Scheduler accumulation rules live in [Ops orchestration runbook contracts](operations/orchestration-runbooks.md#single-study-accumulation-contract).

### Operations

- [Installation](installation.md): environment setup and verification baseline.
- [Runbook catalog](runbooks/README.md): shell command index for cross-tool procedures and tool entrypoints.
- [Ops orchestration index](operations/README.md): orchestration docs for init, plan, execute, and status checks.
- [BU SCC docs index](bu-scc/README.md): cluster setup, submission, and operator runbooks.
- [Notify docs index](notify/README.md): notifier setup, runtime behavior, and recovery routes.
- [USR operations index](../src/dnadesign/usr/docs/operations/README.md): sync, transfer, dataset assembly, and downstream handoff runbooks.

### Maintainer references

- [Developer docs](dev/README.md): CI/testing contracts and maintainer operations.
- [Execution plans index](exec-plans/README.md): active and completed execution-plan records.
- [Study records index](studies/README.md): checked-in live study manifests, affiliated-dataset registries, and status notes for real cross-tool efforts.
- [Templates index](templates/README.md): reusable templates for runbooks, plans, and records.
- [Architecture decisions index](architecture/decisions/README.md): ADR catalog and decision history.
- [Quality docs index](quality/README.md): quality gates, audits, and measurement references.
