## USR docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

### Choose a task

- [Create or inspect one dataset locally](getting-started/cli-quickstart.md): shortest path to one validated USR lifecycle.
- [Sync an existing dataset between local and HPC](operations/workflow-map.md): choose the sync doc for clone, pull, push, and audit work.
- [Assemble multiple producer datasets into one shared dataset](operations/multi-source-shared-dataset-assembly.md): shared USR-backed merge/carry path before construct and infer.
- [Hand a construct-backed shared dataset to infer and downstream watchers](operations/construct-infer-shared-dataset-runbook.md): one construct -> USR -> infer handoff.
- [Understand the promoter-study Evo2 path before choosing a concrete branch](operations/promoter-evo2-journey.md): linked route from DenseGen/manual/wildtype inputs through optional construct contexts, infer Evo2 bundles, Notify watcher validation, and Cluster/OPAL branches.
- [Keep one promoter-study status record for current-status checks](operations/promoter-study-status-contract.md): required manifest, affiliated-dataset registry, checked-in `docs/studies/` location, status note template, and refresh commands for one real study.
- [Build an infer-annotated promoter-study feature dataset, then choose cluster or prepare OPAL](operations/promoter-characterization-feature-matrix.md): shared downstream branch once DenseGen/manual/construct inputs should all become one feature dataset and one explicit `X` column can be selected.

### Read order

1. [Top README](../README.md): package intent, hard boundaries, and where to branch next.
2. [Getting started index](getting-started/README.md): first local run paths for CLI and notebook setup.
3. [Operations index](operations/README.md): sync and cross-tool shared-dataset runbooks for iterative remote and HPC updates.
4. [Reference index](reference/README.md): schema, overlay, event, and API contracts.
5. [Architecture introspection](architecture-introspection.md): deep lifecycle and module interaction map for integration work.

### Getting started

- [CLI quickstart](getting-started/cli-quickstart.md): create a dataset and execute the full local lifecycle once end-to-end.
- [Interactive notebook](getting-started/notebook.md): inspect and iterate on datasets in marimo with path-first helpers.

### Operations

- [Operations index](operations/README.md): full runbook map with task shortcuts and an execution order.
- [Workflow map](operations/workflow-map.md): choose a command chain by intent before diving into details.
- [Sync over SSH](operations/sync.md): choose quickstart, setup, modes, and troubleshooting for SSH-based dataset sync.
- [Sync quickstart](operations/sync-quickstart.md): minimal daily loop for iterative HPC pull and push updates.
- [Sync setup](operations/sync-setup.md): one-time SSH keys, remote profile wiring, and key rotation.
- [Sync target modes](operations/sync-modes.md): path mapping for dataset-directory sync versus single-file sync.
- [Sync troubleshooting](operations/sync-troubleshooting.md): failure signatures with deterministic diagnosis order.
- [Sync audit loop](operations/sync-audit-loop.md): machine-readable transfer decisions for chained command execution.
- [HPC sync flow](operations/hpc-agent-sync-flow.md): preflight/run/verify loop for batch-driven workspace updates.
- [Chained DenseGen and Infer sync runbook](operations/chained-densegen-infer-sync-runbook.md): end-to-end cross-tool update loop with bidirectional sync.
- [Multi-source shared dataset assembly](operations/multi-source-shared-dataset-assembly.md): merge multiple USR-backed producer datasets before construct and infer share one downstream dataset.
- [Construct -> USR -> Infer shared dataset runbook](operations/construct-infer-shared-dataset-runbook.md): construct-led consolidation path for one USR-backed dataset plus infer handoff.
- [Promoter study status contract](operations/promoter-study-status-contract.md): maintain a study-specific manifest, affiliated-dataset registry, and status note under `docs/studies/` for current-status checks.
- [Promoter characterization feature matrix](operations/promoter-characterization-feature-matrix.md): combine DenseGen/manual sources, optional construct expansion, and infer feature write-back before downstream cluster use or OPAL setup.
- [Sync fidelity drills](operations/sync-fidelity-drills.md): adversarial checks for sidecar, overlay, and hash parity.

### Reference

- [Reference index](reference/README.md): entrypoint for stable USR contracts.
- [Dataset layout and code map](reference/dataset-layout-and-code-map.md): on-disk structure and source module map.
- [Schema contract](reference/schema-contract.md): required columns, types, and metadata keys.
- [Overlay and registry contract](reference/overlay-and-registry.md): merge semantics and namespace governance.
- [Event log contract](reference/event-log.md): `.events.log` payload fields and downstream integration boundary.
- [Python API quickstart](reference/python-api.md): minimal `Dataset` usage flow for scripts and notebooks.
- [Maintenance patterns](reference/maintenance.md): dedupe, merge, compaction, snapshot, and export routines.

### Integration boundaries

- DenseGen outputs and overlays: [../../densegen/docs/reference/outputs.md](../../densegen/docs/reference/outputs.md) (write contracts for what DenseGen persists into USR).
- Notify event consumer contract: [../../../../docs/notify/usr-events.md](../../../../docs/notify/usr-events.md) (read contract for how `.events.log` is consumed downstream).

Boundary reminder:
- DenseGen telemetry `outputs/meta/events.jsonl` is not Notify input.
- Notify consumes USR dataset `.events.log`.
