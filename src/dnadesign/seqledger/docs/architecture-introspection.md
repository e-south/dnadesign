<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](README.md)

# SeqLedger architecture

SeqLedger owns sequence identity, annotation storage and dataset transfer. Other tools produce sequences or annotations through its public API; analysis consumers read the combined records and overlays. [Notify](../../notify/) can consume mutation events independently of those producers.

## From input to annotated records

1. The registry declares annotation namespaces and types.
2. Import normalizes sequence identity and adds base records to `records.parquet`.
3. Annotation writers join values by a declared key and write overlays under `_derived/`.
4. Dataset reads apply overlays in deterministic order. Export produces this combined view without rewriting the base table.
5. Maintenance operations can compact parts, materialize annotations or rewrite records when explicitly requested.

Sequence views are separate sidecars for lineage and semantic identity. Their storage does not change a base record's sequence ID. The [schema reference](reference/schema-contract.md) defines both identities.

## Ownership and entry points

Cross-tool code imports `dnadesign.seqledger`; `dnadesign.seqledger.src` is internal. The CLI parses arguments and delegates to dataset, maintenance or sync operations. The [code map](reference/dataset-layout-and-code-map.md#maintainer-code-map) identifies the owning packages.

- `dataset/` coordinates helpers under `datasets/` for reads and mutations.
- `registry/`, `contracts/` and `overlays/` enforce column and namespace rules.
- `storage/` provides Parquet operations and dataset locks.
- `events/` records mutations and redacts sensitive arguments.
- `sync/` manages remote comparison, transfer and verification.

Keep producer-specific choices in the producing tool. A storage call should receive explicit sequences, annotation values or view metadata rather than infer scientific interpretation from a dataset name.

## Mutation and read contracts

Base records have content-derived IDs. Annotations are namespaced and typed by the registry. Overlay resolution is last-writer-wins in part order, with creation time and filename determining that order. `write_overlay_part(overwrite=False)` checks occupied cells while holding the dataset write lock.

Normal reads include overlays and exclude deleted records by default. Direct Parquet reads bypass those rules. Compaction consolidates overlay parts; materialization writes the effective annotations into the base table. Neither is required to inspect or export annotations.

Use [maintenance commands](reference/maintenance.md) for deliberate rewrites. Event-log gardening also requires coordination with active consumers because their saved byte offsets become invalid after the log is rewritten.

## Transfer contracts

Dataset sync compares the primary table, events and sidecars. It defaults to hash verification and content-hash checks for `_derived` and `_auxiliary` files. Hashing cost grows with the amount of data, but disabling checks changes the fidelity guarantee.

Pull stages data and verifies it before promotion. Push checks compatible event history while holding transfer locks. Transfers do not merge independently diverged histories. The [sync reference](operations/sync/modes.md#transfer-consistency) describes locking, history and verification rules; the [audit guide](operations/sync/audit-loop.md) explains machine-readable results.

## Configuration boundaries

| Setting | Effect |
| --- | --- |
| `--root` / `DNADESIGN_SEQLEDGER_ROOT` | Select the local dataset root; otherwise use `./seqledger_datasets` |
| `--remotes-config` / `SEQLEDGER_REMOTES_PATH` | Select the private SSH remote configuration |
| `--verify` | Choose primary-file verification for sync |
| `--verify-sidecars` | Check complete dataset sidecar parity |
| `--verify-derived-hashes` | Check overlay and auxiliary content hashes |
| `--strict-bootstrap-id` | Require an explicit canonical ID for a bootstrap pull |

Explicit CLI choices take precedence over environment defaults. Sidecar verification options apply to dataset transfers, not single-file transfers. For commands and setup, use the [sync guide](operations/sync/README.md).

## Validate a change

Start with the tests for the affected owner. Schema and overlay changes also need read/export coverage; sync changes need interruption, divergence and fidelity checks. [Sync verification drills](operations/sync/fidelity-drills.md) provide local test entry points. [Repository change gates](../../../../docs/repo-change-gate.md) define the shared documentation, architecture and lint checks.
