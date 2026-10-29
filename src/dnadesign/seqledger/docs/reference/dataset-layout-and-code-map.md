<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../README.md) / [Reference](README.md)

# Dataset layout and code map

## Dataset layout

A root directory contains the namespace registry and one directory per dataset. Keep this data outside the installed package. The default root is `./seqledger_datasets`; select another with `--root` or `DNADESIGN_SEQLEDGER_ROOT`.

```text
<data-root>/
├── registry.yaml                 # Annotation namespaces and column types
└── <dataset-id>/
    ├── records.parquet           # Base sequence records
    ├── _derived/                 # Annotation overlays and incremental parts
    ├── _views/                   # Sequence-view identity and semantics
    ├── meta.md                   # Dataset notes
    ├── .events.log               # Mutation history
    ├── _registry/                # Dataset registry snapshots
    └── _snapshots/               # Base-table snapshots
```

Some directories appear only after the corresponding operation. Reads through SeqLedger combine base records and overlays; a direct read of `records.parquet` sees only the base file. See the [schema](schema-contract.md), [overlay](overlay-and-registry.md) and [event](event-log.md) references for their contracts.

## Maintainer code map

Consumers import `dnadesign.seqledger`. The public facade is [the package initializer](../../__init__.py); implementation packages live under [`src/`](../../src/).

| Package | Responsibility |
| --- | --- |
| `api/` | Public-library facade implementation |
| `cli/` | Command registration, handlers, target resolution and presentation |
| `contracts/` | Errors, schema constants, return types, roots and sequence identity |
| `dataset/` | Dataset coordinator |
| `datasets/` | Ingest, lifecycle, overlay, query, merge, state, validation and read/export operations |
| `events/` | Mutation events, redaction, fingerprints and append operations |
| `genbank/` | GenBank import and feature extraction |
| `maintenance/` | Explicit context for operations that rewrite dataset state |
| `overlays/` | Overlay discovery, metadata, validation and support helpers |
| `regulondb/` | RegulonDB annotation enrichment |
| `registry/` | Namespace models, persistence and type validation |
| `runtime/` | Shared runtime setup, including DuckDB UTC sessions |
| `sequence_views/` | View identity, sidecar storage and mutable semantics |
| `storage/` | Parquet I/O, snapshots and write locks |
| `sync/` | Remote configuration, comparison, locking, transfer and verification |
| `version/` | Package version |

CLI helpers are grouped by purpose:

- [`cli/support/resolution/dataset_targets.py`](../../src/cli/support/resolution/dataset_targets.py) resolves dataset targets.
- [`cli/support/wiring/dependencies.py`](../../src/cli/support/wiring/dependencies.py) and [`registration.py`](../../src/cli/support/wiring/registration.py) supply dependencies and register commands.
- [`cli/support/presentation/runtime.py`](../../src/cli/support/presentation/runtime.py) formats runtime output.

Dataset operations belong under their corresponding `datasets/` family rather than in CLI handlers.

The [architecture guide](../architecture-introspection.md) explains how these components interact. Contributor instructions and exact source ownership are in [AGENTS.md](../../AGENTS.md).

## Validation tools

The local sync drill lives at `src/dnadesign/seqledger/ops/sync_audit_drill.py`, exposed as `seqledger-sync-audit-drill`. Shared test fixtures live at `src/dnadesign/devtools/tests/support/usr.py`; cross-tool tests should use those fixtures instead of importing another tool's tests.

See [sync verification drills](../operations/sync/fidelity-drills.md) for runnable checks.
