# USR docs

## Contents
- [At a glance](#at-a-glance)
- [Read order](#read-order)
- [Architecture introspection](#architecture-introspection)
- [Integration](#integration)
- [Operations](#operations)
- [Progressive workflows](#progressive-workflows)

## At a glance

**Intent:** USR is the canonical, auditable sequence store and mutation/event boundary for `dnadesign`.

If you are new here:
- read `../README.md` for the mental model and CLI quickstart
- use `operations/sync.md` when you need SSH/HPC transfer workflows
- use `dev/journal.md` for maintainers-only history and decisions

**When to use:**
- Store canonical sequence datasets (generated and curated).
- Attach derived metrics or annotations without rewriting base records.
- Emit mutation events for operators and automation (Notify).
- Govern derived columns with explicit namespace registry contracts.

**When not to use:**
- Not a sequence generator (use DenseGen and related tools).
- Not an alerting transport (use Notify).

**Boundary / contracts:**
- `.events.log` is the integration boundary; Notify consumes this stream.
- Derived columns must be namespaced as `<namespace>__<field>`.
- Namespaces must be registered before attach/materialize operations.

**Start here:**
- [../README.md](../README.md) (concepts + CLI quickstart)
- [../README.md#namespace-registry-required](../README.md#namespace-registry-required)
- [../README.md#event-log-schema](../README.md#event-log-schema)
- [operations/sync.md](operations/sync.md) (remote sync)

## Read order

- USR concepts plus CLI quickstart: [../README.md](../README.md)
- USR architecture/lifecycle/config introspection: [architecture-introspection.md](architecture-introspection.md)
- Overlay plus registry contract: [../README.md#namespace-registry-required](../README.md#namespace-registry-required)
- Overlay merge semantics: [../README.md#how-overlays-merge-conflict-resolution](../README.md#how-overlays-merge-conflict-resolution)
- Event log schema (Notify input): [../README.md#event-log-schema](../README.md#event-log-schema)

## Architecture introspection

- Package intent, lifecycle, and interaction map: [architecture-introspection.md](architecture-introspection.md)
- Use this when you need source-backed boundaries before changing sync/materialize/registry behavior.

## Integration

- DenseGen writes into USR via the `densegen` namespace:
  - DenseGen outputs reference: [../../densegen/docs/reference/outputs.md](../../densegen/docs/reference/outputs.md)
- Notify consumes USR `.events.log`:
  - Notify operators doc: [../../../../docs/notify/usr-events.md](../../../../docs/notify/usr-events.md)

Boundary reminder:
- DenseGen runtime telemetry (`outputs/meta/events.jsonl`) is not Notify input.
- Notify reads USR dataset `.events.log` only.

## Operations

- Operations index: [operations/README.md](operations/README.md)
- Workflow map (task-first command chains): [operations/workflow-map.md](operations/workflow-map.md)
- Remote sync: [operations/sync.md](operations/sync.md)
  - Progressive disclosure path: Quick path -> Advanced path -> Failure diagnosis
- Sync audit loop: [operations/sync-audit-loop.md](operations/sync-audit-loop.md)
- HPC sync runbook: [operations/hpc-agent-sync-flow.md](operations/hpc-agent-sync-flow.md)
- Chained DenseGen+Infer sync demo: [operations/chained-densegen-infer-sync-demo.md](operations/chained-densegen-infer-sync-demo.md)
- Dev notes: [dev/journal.md](dev/journal.md)

## Progressive workflows

- Start with USR concepts + CLI basics: [../README.md](../README.md)
- Pick a command chain first: [operations/workflow-map.md](operations/workflow-map.md)
- Move to SSH sync mechanics: [operations/sync.md](operations/sync.md)
- Add machine-readable audit artifacts: [operations/sync-audit-loop.md](operations/sync-audit-loop.md)
- Use the batch-safe operator loop: [operations/hpc-agent-sync-flow.md](operations/hpc-agent-sync-flow.md)
- Use the chained DenseGen/Infer loop: [operations/chained-densegen-infer-sync-demo.md](operations/chained-densegen-infer-sync-demo.md)
- Then apply sibling-tool docs:
  - DenseGen: [../../densegen/README.md](../../densegen/README.md)
  - Infer: [../../infer/README.md](../../infer/README.md)
