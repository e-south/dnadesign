## Documentation Index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Find the next task, command, or reference doc from the routes below.

Start with:

- [Installation](setup/installation.md) to set up the repo on a new machine.
- [Inspect available work](#inspect-available-work) to list packaged workspaces or dataset roots before choosing a workflow.
- [Choose a workflow](#choose-a-workflow) to follow a single-tool path, a cross-tool dataset path, or an operations path.
- [Runbook catalog](runbooks/README.md) when you want commands before longer docs.
- [Tool docs](#tool-docs) when you already know the package.
- [Checked-in study routes](#checked-in-study-routes) when a task names a live
  study and you need its route/status surface.
- [System records](#system-records), [Operations](#operations), and [Maintainer references](#maintainer-references) for repo contracts and operator docs.

### Inspect available work

Use this table when the first question is "what is available right now?"
rather than "which workflow should I run?" It lists durable workspace and
dataset roots only. Folding is workspace-less and consumes producer-owned
bundles. BaseRender can run optional demo/ad hoc workspaces, but most
cross-tool use should target producer-emitted render job files or visual
contracts.

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
| Compose declared linear ssDNA parts into a local artifact bundle | [Construct linear ssDNA composition](../src/dnadesign/construct/docs/reference/linear-ssdna-composition.md) | Verify the bundle manifest, optional Folding plot artifacts, and optional BaseRender review handoff. |
| Run model inference and write outputs back to datasets | [Infer docs index](../src/dnadesign/infer/docs/README.md) | Verify write-back columns and types with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md). |
| Set up the SCC Evo2 infer GPU environment | [BU SCC install bootstrap](bu-scc/setup/install.md#gpu-setup-and-verification-runbook) | Verify infer model capabilities with [infer SCC Evo2 GPU runbook](../src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md). |
| Run Notify for local event watching and webhook setup | [Notify docs index](notify/README.md) | Verify mode and delivery contracts in [Notify command contracts](../src/dnadesign/notify/docs/reference/command-contracts.md). |
| Inspect or extend Notify package internals | [Notify package docs index](../src/dnadesign/notify/docs/README.md) | Verify module boundaries in [Notify maintainer architecture map](../src/dnadesign/notify/docs/dev/architecture.md). |

#### Cross-tool dataset workflows

Use these when data moves through more than one tool and the shared record lives in USR.

| Need | Primary workflow | Verify next |
| --- | --- | --- |
| Assemble multiple producer datasets before construct and infer share one downstream dataset | [Multi-source shared dataset assembly](../src/dnadesign/usr/docs/operations/assembly/multi-source-shared-dataset.md) | Verify carried overlays, construct lineage, and infer write-back contracts with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md) and [Infer docs](../src/dnadesign/infer/docs/README.md). |
| Hand one construct-backed dataset to infer and downstream watchers | [Construct -> USR -> Infer shared dataset runbook](../src/dnadesign/usr/docs/operations/assembly/construct-infer-shared-dataset-runbook.md) | Verify lineage plus downstream write-back contracts with [USR schema contract](../src/dnadesign/usr/docs/reference/schema-contract.md) and [Infer docs](../src/dnadesign/infer/docs/README.md). |
| Review the stress/ethanol/cipro Evo2 route before choosing the next step | [Stress/ethanol/cipro Evo2 workflow journey](../src/dnadesign/usr/docs/operations/promoter/evo2-journey.md) | Continue to [promoter characterization feature matrix](../src/dnadesign/usr/docs/operations/promoter/characterization-feature-matrix.md) only when the task is the generic USR handoff. |
| Check the current stress/ethanol/cipro study record | [Stress ethanol/cipro status contract](studies/stress_ethanol_cipro_growth/operations/catalog/contracts/status.md) | Read the concrete study record in `docs/studies/stress_ethanol_cipro_growth/`, then open its route map when the next need is owner-surface navigation. |
| Run the deeper stress/ethanol/cipro command preflight | [Stress ethanol/cipro preflight contract](studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md) | Use the study-owned preflight only for blocker or next-run readiness for `stress_ethanol_cipro_growth`. |
| Navigate a checked-in study without exposing study-specific routes here | [Study records index](studies/README.md) | Use the active selector or named study directory, then open `docs/studies/<study-id>/routes/README.md`, `operations/ops.study.yaml`, and `operations/runtime/command-groups/pipeline.yaml` for declared status, preflight, or compiler routes. |
| Check study dataset-root semantics and affiliated-dataset registry terms | [Study records index](studies/README.md) | Verify the active study selector and `record_root` in [Study index](studies/index.yaml), then read `operations/ops.study.yaml` for explicit Ops surfaces. |
| Build a promoter feature dataset from anchors, wildtype/manual promoters, optional construct contexts, and infer outputs | [Promoter characterization feature matrix](../src/dnadesign/usr/docs/operations/promoter/characterization-feature-matrix.md) | Verify one explicit `infer__...` column is chosen as `X` or export a flattened matrix before continuing to the exploratory [cluster workflow](../src/dnadesign/cluster/docs/workflows/exploratory-clustering.md) or the downstream [USR dataset with infer-derived X -> OPAL active learning](../src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md) workflow. |
| Sync iterative HPC outputs to local analysis safely | [USR workflow map](../src/dnadesign/usr/docs/operations/routes/workflow-map.md) -> [USR HPC sync flow](../src/dnadesign/usr/docs/operations/sync/hpc-agent-flow.md) | Verify transfer parity with [USR sync audit loop](../src/dnadesign/usr/docs/operations/sync/audit-loop.md). |
| Run cross-machine sync with stricter failure checks | [USR sync command contract](../src/dnadesign/usr/docs/operations/sync/README.md) | Verify sidecar and overlay fidelity with [USR sync fidelity drills](../src/dnadesign/usr/docs/operations/sync/fidelity-drills.md). |
| Chain DenseGen -> USR -> Infer -> USR updates | [Chained DenseGen and Infer sync runbook](../src/dnadesign/usr/docs/operations/sync/chained-densegen-infer-runbook.md) | Verify downstream dataset state with [Infer docs](../src/dnadesign/infer/docs/README.md). |

### Checked-In Study Routes

Use this table when the task names one concrete study. Do not borrow a status
provider across studies.

| Study | First route | Status surface | When to use Ops |
| --- | --- | --- | --- |
| `stress_ethanol_cipro_growth` | [routes](studies/stress_ethanol_cipro_growth/routes/README.md) | [status](studies/stress_ethanol_cipro_growth/operations/catalog/contracts/status.md), [preflight](studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md) | Use `ops progress show` for status, blocker, or readiness questions. |
| `regulondb_native_promoter_panel` | [routes](studies/regulondb_native_promoter_panel/routes/README.md) | record-only: `record/status.md`, `record/datasets.yaml`, `operations/ops.study.yaml` | Do not use Ops status/preflight; no provider is registered. |
| `retron_hairpin_design` | [routes](studies/retron_hairpin_design/routes/README.md) | [status](studies/retron_hairpin_design/operations/catalog/contracts/status.md), [preflight](studies/retron_hairpin_design/operations/catalog/contracts/preflight.md) | Use Ops only for explicit progress/readiness questions; deliverable requests route to materialize. |

#### Scheduler and environment workflows

Use these when the next step is orchestration, environment setup, or audit output rather than dataset mutation.

| Need | Primary workflow | Verify next |
| --- | --- | --- |
| Run BU SCC batch jobs with notifications | [BU SCC batch + notify runbook](bu-scc/runbooks/batch-notify.md) | Verify event delivery contract in [Notify USR events contract](notify/usr-events.md). |
| Plan and execute deterministic DenseGen/Infer HPC orchestration runbooks | [Ops orchestration index](operations/README.md) | Verify command ordering and outcomes in [orchestration audit contract](operations/orchestration/runbooks.md#contract-rules). |

### Tool docs

These sections keep unlike package surfaces separate without adding more
columns. Prefer workspace/dataset routers for user work; call artifact services
such as `folding` only when a producer has already emitted the required
contract artifacts or bundle manifest.

#### Workspace and analysis tools

| Tool | CLI | Docs |
| --- | --- | --- |
| `aligner` | n/a | [aligner docs](../src/dnadesign/aligner/docs/README.md) |
| `billboard` | n/a | [billboard docs](../src/dnadesign/billboard/docs/README.md) |
| `cluster` | `uv run cluster --help` | [cluster README](../src/dnadesign/cluster/README.md) |
| `cruncher` | `uv run cruncher --help` | [cruncher README](../src/dnadesign/cruncher/README.md) |
| `densegen` | `uv run dense --help` | [densegen README](../src/dnadesign/densegen/README.md) |
| `construct` | `uv run construct --help` | [construct README](../src/dnadesign/construct/README.md) |
| `infer` | `uv run infer --help` | [infer README](../src/dnadesign/infer/README.md) |
| `latentdna` | `uv run latentdna --help` | [latentdna README](../src/dnadesign/latentdna/README.md) |
| `libshuffle` | n/a | [libshuffle docs](../src/dnadesign/libshuffle/docs/README.md) |
| `nmf` | n/a | [nmf docs](../src/dnadesign/nmf/docs/README.md) |
| `opal` | `uv run opal --help` | [opal README](../src/dnadesign/opal/README.md) |
| `permuter` | `uv run permuter --help` | [permuter docs](../src/dnadesign/permuter/docs/README.md) |
| `tfkdanalysis` | n/a | [tfkdanalysis docs](../src/dnadesign/tfkdanalysis/docs/README.md) |
| `usr` | `uv run usr --help` | [usr README](../src/dnadesign/usr/README.md) |

#### Artifact services

| Tool | CLI | Docs |
| --- | --- | --- |
| `baserender` | `uv run baserender --help` | [baserender README](../src/dnadesign/baserender/README.md); contract renderer, job files, optional demo/ad hoc workspaces |
| `folding` | `uv run folding --help` | [folding README](../src/dnadesign/folding/README.md); workspace-less service |

#### Operator surfaces

| Tool | CLI | Docs |
| --- | --- | --- |
| `notify` | `uv run notify --help` | [notify README](../src/dnadesign/notify/README.md) |
| `ops` | `uv run ops --help` | [ops README](../src/dnadesign/ops/README.md) |

#### Shared contracts and routing

| Surface | CLI | Docs |
| --- | --- | --- |
| `contracts` | n/a | [contracts docs](../src/dnadesign/contracts/docs/README.md) |
| `studies` | n/a | [studies README](../src/dnadesign/studies/README.md) |

### System records

- [Architecture](../ARCHITECTURE.md): component boundaries and cross-tool integration map.
- [Design](../DESIGN.md): engineering invariants and interface expectations.
- [Reliability](../RELIABILITY.md): operational model, failure posture, and recovery contracts.
- [Security](../SECURITY.md): secrets, dependency, and supply-chain handling policy.
- [Plans](../PLANS.md): proposal, execution-plan, and decision lifecycle.
- [Quality score](../QUALITY_SCORE.md): quality rubric and improvement framework.
- Cross-tool handoff rules live in [Architecture](../ARCHITECTURE.md#cross-tool-information-architecture), [Design](../DESIGN.md#information-architecture-invariants), and the shared dataset workflows linked above.
- Scheduler accumulation rules live in [Ops orchestration runbook contracts](operations/orchestration/runbooks.md#single-study-accumulation-contract).

### Operations

- [Installation](setup/installation.md): environment setup and verification baseline.
- [Runbook catalog](runbooks/README.md): shell command index for cross-tool procedures and tool entrypoints.
- [Ops orchestration index](operations/README.md): orchestration docs for init, plan, execute, and status checks.
- [OPS mental model](operations/model/mental-model.md): plane model, shared state semantics, and snapshot versus preflight guidance.
- [OPS failure contract](operations/contracts/failure.md): CLI exit-code and stderr contract for automation and maintainers.
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
