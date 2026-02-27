## USR docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27

### Read order

1. [Top README](../README.md): package intent, hard boundaries, and where to branch next.
2. [Getting started index](getting-started/README.md): first local run paths for CLI and notebook setup.
3. [Operations index](operations/README.md): task-first sync runbooks for iterative remote and HPC updates.
4. [Reference index](reference/README.md): authoritative contracts for schema, overlays, events, and API usage.
5. [Architecture introspection](architecture-introspection.md): deep lifecycle and module interaction map for integration work.

### Getting started

- [CLI quickstart](getting-started/cli-quickstart.md): create a dataset and execute the full local lifecycle once end-to-end.
- [Interactive notebook](getting-started/notebook.md): inspect and iterate on datasets in marimo with path-first helpers.

### Operations

- [Operations index](operations/README.md): full runbook map with task shortcuts and an execution order.
- [Workflow map](operations/workflow-map.md): choose a command chain by intent before diving into details.
- [Sync over SSH](operations/sync.md): sync router page linking quickstart, setup, modes, and troubleshooting.
- [Sync quickstart](operations/sync-quickstart.md): minimal daily loop for iterative HPC pull and push updates.
- [Sync setup](operations/sync-setup.md): one-time SSH keys, remote profile wiring, and key rotation.
- [Sync target modes](operations/sync-modes.md): path mapping for dataset-directory sync versus single-file sync.
- [Sync troubleshooting](operations/sync-troubleshooting.md): failure signatures with deterministic diagnosis order.
- [Sync audit loop](operations/sync-audit-loop.md): machine-readable transfer decisions for chained command execution.
- [HPC sync flow](operations/hpc-agent-sync-flow.md): preflight/run/verify loop for batch-driven workspace updates.
- [Chained DenseGen and Infer sync demo](operations/chained-densegen-infer-sync-demo.md): end-to-end cross-tool update loop with bidirectional sync.
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
