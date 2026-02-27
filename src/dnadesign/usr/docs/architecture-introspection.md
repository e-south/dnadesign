# USR Architecture Introspection

## Decision summary

- Scope: `src/dnadesign/usr/src/` package behavior, with emphasis on sync reliability, overlay materialization, and operator-facing contracts.
- Assumptions: repository-local package layout is authoritative; CLI path is `uv run usr`.
- Depth: deep package-level introspection with lifecycle, config-to-behavior mapping, and operational interaction boundaries.

## Intent and use-case map

- Problem intent:
  - Maintain a canonical sequence store that supports iterative updates without losing reproducibility.
  - Preserve strict mutation/event boundaries for sibling tools and operators.
- Primary use cases:
  - Initialize/import canonical records for namespace-qualified datasets.
  - Attach tool-derived overlays incrementally and materialize deterministically.
  - Run iterative pull/push cycles between local and HPC roots with strict verification.
  - Emit `.events.log` as operator integration boundary.
- Secondary use cases:
  - Snapshot, restore, and state transitions (`delete`, `restore`, `state set/clear/get`).
  - Namespace registry governance and schema freeze checks.
- Non-goals:
  - Not a sequence generation runtime.
  - Not a webhook transport runtime.

## Core functionality and behavior contract

- Canonical primary:
  - Exactly one `records.parquet` per dataset root.
- Overlay contract:
  - Derived columns must be namespaced (`<namespace>__<field>`), with registry-governed schema.
  - Overlay parts are append-only; read/materialize semantics are deterministic last-writer-wins.
- Sync contract:
  - Dataset sync defaults to primary hash verification plus strict sidecar parity.
  - High-assurance mode (`--verify-derived-hashes`) verifies `_derived` and `_auxiliary` file-content hashes.
- Failure behavior:
  - Hard errors on invalid schema, missing required files, unsupported paths, and verification mismatches.
  - Pull promotes staged payload only after verification; partial transfer does not mutate canonical local primary.

## Lifecycle model

1. Bootstrap:
  - Register namespace contract, init dataset, import canonical records.
2. Enrichment:
  - Attach overlays by key (`id`, `sequence`, `sequence_norm`, `sequence_ci`) under namespace constraints.
3. Consolidation:
  - Materialize overlays into base records with maintenance-gated mutation.
4. Distribution:
  - Diff/pull/push between roots with verification and sync audit summaries.
5. Governance:
  - Event emission, snapshots, tombstones/state overlays, and registry freeze checkpoints.

Edge cases handled:
- Interrupted pull/push retries.
- Sidecar inventory mismatch (`_derived`, `_snapshots`, `_auxiliary`).
- Hash mismatch in high-assurance sync.
- Legacy dataset path rejection and strict bootstrap dataset-id enforcement.

## Architecture view stack

- Context view:
  - DenseGen/Infer write annotations into USR datasets.
  - Notify consumes USR `.events.log`.
- Container/module view:
  - CLI/wiring: `cli.py`, `cli_commands/*`
  - Dataset core: `dataset.py`, `dataset_overlay_ops.py`, `dataset_overlay_maintenance.py`, `dataset_materialize.py`, `dataset_state.py`
  - Sync core: `sync.py`, `sync_sidecars.py`, `remote.py`, `diff.py`
  - Contracts: `schema.py`, `registry.py`, `event_schema.py`
- Component/function view:
  - Sync: `execute_pull`, `execute_push`, `verify_sidecar_state_match`, `stat_dataset`
  - Overlay materialize: `materialize_dataset`, `validate_overlay_schema`
  - Registry enforcement: `register_namespace`, `parse_columns_spec`

Runtime interaction scenario:
- HPC batch appends overlays -> local `usr pull` stages and verifies -> local analysis/infer writes overlays -> `usr push` verifies remote parity -> operators inspect sync audit and `.events.log`.

## Config-schema to behavior mapping

| Key | Source | Behavior effect |
| --- | --- | --- |
| `--root` | CLI root callback | Selects dataset root boundary and path resolution context. |
| `--verify` (`hash|auto|size|parquet`) | Sync CLI + diff resolver | Chooses primary verification method for diff/pull/push. |
| `--verify-sidecars` / `--no-verify-sidecars` | Sync policy | Enables/disables strict sidecar parity checks for dataset mode. |
| `--verify-derived-hashes` | Sync policy + sidecar verifier | Enables high-assurance content-hash verification for `_derived` and `_auxiliary`. |
| `USR_SYNC_STRICT_BOOTSTRAP_ID=1` | Sync CLI policy | Requires namespace-qualified dataset id on bootstrap pulls. |
| `USR_SHOW_DEV_COMMANDS=1` | CLI app registration | Enables hidden dev subcommands in CLI surface. |
| `USR_REMOTES_PATH` | Remote config loading | Selects remotes registry path for SSH sync profiles. |

Precedence notes:
- Explicit CLI flags override defaults.
- Sidecar/derived-hash flags are dataset-only and fail fast in file mode.

## Interaction map

- Upstream inputs:
  - Tool outputs (DenseGen, Infer) as overlay attachments.
  - Remote SSH roots for cross-location sync.
- Downstream consumers:
  - Notebook and analysis consumers reading canonical records + overlays.
  - Notify operator workflows via `.events.log`.
- Control/data flow:
  - CLI -> command handlers -> dataset/sync/runtime modules -> filesystem/remote subprocesses.
  - Registry and schema checks gate writes/materialization.

## Math and operations notes

- Primary identity:
  - `id = sha1("bio_type|sequence_norm")` (stable identity over normalized sequence and bio_type).
- Diff/verification complexity:
  - Primary hash verification is `O(file_size)`.
  - High-assurance sidecar hashes scale with number and size of `_derived` and auxiliary files.
- Overlay resolution:
  - Last-writer-wins by overlay part ordering (`created_at`, filename tiebreak).

## Evidence ledger

- CLI wiring and sync surface:
  - `src/dnadesign/usr/src/cli.py`
  - `src/dnadesign/usr/src/cli_commands/sync.py`
  - `src/dnadesign/usr/src/cli_commands/sync_cli.py`
- Sync behavior and verification:
  - `src/dnadesign/usr/src/sync.py`
  - `src/dnadesign/usr/src/sync_sidecars.py`
  - `src/dnadesign/usr/src/remote.py`
  - `src/dnadesign/usr/src/diff.py`
- Overlay/materialize and schema contracts:
  - `src/dnadesign/usr/src/dataset_overlay_ops.py`
  - `src/dnadesign/usr/src/dataset_overlay_maintenance.py`
  - `src/dnadesign/usr/src/dataset_materialize.py`
  - `src/dnadesign/usr/src/registry.py`
  - `src/dnadesign/usr/src/schema.py`
- Behavior and reliability tests:
  - `src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`
  - `src/dnadesign/usr/tests/test_sync_schema_adversarial.py`
  - `src/dnadesign/usr/tests/test_usr_docs_contract.py`

## Open questions and risk notes

- `dataset.py` remains a large orchestration surface; additional extraction slices should continue around query and reserved-overlay paths.
- High-assurance hash mode can add runtime cost on very large overlay trees; operators should choose cadence based on transfer window constraints.
- Sync audit output is strong for decision support; adding machine-readable audit snapshots may further improve automated orchestration loops.
