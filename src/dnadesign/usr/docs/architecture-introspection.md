# USR Architecture Introspection

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-23


## Decision summary

- Scope: `src/dnadesign/usr/src/` package behavior, with emphasis on sync reliability, overlay materialization, and operator-facing contracts.
- Assumptions: repository-local package layout is authoritative; CLI path is `uv run usr`.
- Depth: deep package-level introspection with lifecycle, config-to-behavior mapping, and operational interaction boundaries.

## Intent and use-case map

- Problem intent:
  - Maintain a canonical sequence store that supports iterative updates without losing reproducibility.
  - Preserve strict mutation/event boundaries for sibling tools and operators.
- Primary use cases:
  - Initialize/import canonical records for flat or namespace-qualified datasets.
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
- Archived storage is kept out of default live dataset-id discovery while explicit archived paths remain readable.
- Legacy promoter-focused PyTorch archives live under `datasets/archived/promoter_misc_pytorch/` and stay outside the `records.parquet` dataset API.
- Strict bootstrap dataset-id enforcement.

## Architecture view stack

Unless noted otherwise, the modules in this section live under `src/dnadesign/usr/src/`.
Package-root Python should stay limited to `__init__.py` and `__main__.py`; cross-tool consumers import from `dnadesign.usr`, not sibling root modules.

- Context view:
  - DenseGen/Infer write annotations into USR datasets.
  - Notify consumes USR `.events.log`.
- Container/module view:
  - CLI/wiring: `cli.py`, `cli_commands/*`, `cli_commands/datasets/*`, `cli_commands/lifecycle/*`, `cli_commands/maintenance/*`, `cli_commands/namespace/*`, `cli_commands/query/*`, `cli_commands/read_views/*`, `cli_commands/remotes/*`, `cli_commands/sync/*`, `cli_commands/tooling/*`, `cli_support/*`
  - Dataset core: `dataset.py` coordinating helpers under `datasets/*`, with lifecycle/registry coordination under `datasets/lifecycle/*`, maintenance-gated merge execution under `datasets/merge/*`, overlay mutation logic under `datasets/overlay/*`, overlay query/catalog logic under `datasets/query/*`, reserved state lifecycle under `datasets/state/*`, integrity checks under `datasets/validate/*`, and read/export flows under `datasets/views/*`
  - Sync core: `sync.py` coordinating helpers under `remote_sync/*`, including execution orchestration under `remote_sync/execution.py`
  - Legacy conversion/repair: `convert_legacy.py` coordinating helpers under `legacy/*`
  - Overlay support: `overlays.py` plus `overlay_support/*`
  - Contracts: `schema.py`, `registry.py`, with USR event-version ownership co-located in `events.py`
- Component/function view:
  - Sync: `execute_pull`, `execute_push`, `verify_sidecar_state_match`, `stat_dataset`
  - Overlay materialize: `materialize_dataset`, `validate_overlay_schema`
  - Registry enforcement: `register_namespace`, `parse_columns_spec`
  - Repair pre-clean dedupe: `apply_casefold_sequence_dedupe`

Runtime interaction scenario:
- HPC batch appends overlays -> local `usr pull` stages and verifies -> local analysis/infer writes overlays -> `usr push` verifies remote parity -> operators inspect sync audit and `.events.log`.

## Config-schema to behavior mapping

| Key | Source | Behavior effect |
| --- | --- | --- |
| `--root` | CLI root callback | Selects dataset root boundary and path resolution context. |
| `--verify` (`hash|auto|size|parquet`) | Sync CLI + diff resolver | Chooses primary verification method for diff/pull/push. |
| `--verify-sidecars` / `--no-verify-sidecars` | Sync policy | Enables/disables strict sidecar parity checks for dataset mode. |
| `--verify-derived-hashes` | Sync policy + sidecar verifier | Enables high-assurance content-hash verification for `_derived` and `_auxiliary`. |
| `USR_SYNC_STRICT_BOOTSTRAP_ID=1` | Sync CLI policy | Requires an explicit canonical dataset id on bootstrap pulls and disables local name guessing. |
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
  - `src/dnadesign/usr/src/cli_support/pretty.py`
  - `src/dnadesign/usr/src/cli_support/rendering.py`
  - `src/dnadesign/usr/src/cli_support/roots.py`
  - `src/dnadesign/usr/src/cli_commands/datasets/__init__.py`
  - `src/dnadesign/usr/src/cli_commands/datasets/catalog.py`
  - `src/dnadesign/usr/src/cli_commands/datasets/resolution.py`
  - `src/dnadesign/usr/src/cli_commands/lifecycle/__init__.py`
  - `src/dnadesign/usr/src/cli_commands/lifecycle/cli.py`
  - `src/dnadesign/usr/src/cli_commands/lifecycle/materialize.py`
  - `src/dnadesign/usr/src/cli_commands/lifecycle/snapshot.py`
  - `src/dnadesign/usr/src/cli_commands/lifecycle/state.py`
  - `src/dnadesign/usr/src/cli_commands/lifecycle/write.py`
  - `src/dnadesign/usr/src/cli_commands/maintenance/__init__.py`
  - `src/dnadesign/usr/src/cli_commands/maintenance/cli.py`
  - `src/dnadesign/usr/src/cli_commands/maintenance/dedupe.py`
  - `src/dnadesign/usr/src/cli_commands/maintenance/merge.py`
  - `src/dnadesign/usr/src/cli_commands/maintenance/overlay.py`
  - `src/dnadesign/usr/src/cli_commands/maintenance/registry.py`
  - `src/dnadesign/usr/src/cli_commands/namespace/__init__.py`
  - `src/dnadesign/usr/src/cli_commands/namespace/cli.py`
  - `src/dnadesign/usr/src/cli_commands/query/__init__.py`
  - `src/dnadesign/usr/src/cli_commands/query/cli.py`
  - `src/dnadesign/usr/src/cli_commands/query/read.py`
  - `src/dnadesign/usr/src/cli_commands/query/runtime.py`
  - `src/dnadesign/usr/src/cli_commands/read_views/__init__.py`
  - `src/dnadesign/usr/src/cli_commands/read_views/parquet_targets.py`
  - `src/dnadesign/usr/src/cli_commands/remotes/__init__.py`
  - `src/dnadesign/usr/src/cli_commands/remotes/cli.py`
  - `src/dnadesign/usr/src/cli_commands/sync/__init__.py`
  - `src/dnadesign/usr/src/cli_commands/sync/cli.py`
  - `src/dnadesign/usr/src/cli_commands/tooling/__init__.py`
  - `src/dnadesign/usr/src/cli_commands/tooling/cli.py`
  - `src/dnadesign/usr/src/cli_commands/tooling/densegen.py`
  - `src/dnadesign/usr/src/cli_commands/tooling/dev.py`
  - `src/dnadesign/usr/src/cli_commands/tooling/legacy.py`
  - `src/dnadesign/usr/src/cli_commands/tooling/shared.py`
- Sync behavior and verification:
  - `src/dnadesign/usr/src/sync.py`
  - `src/dnadesign/usr/src/remote_sync/execution.py`
  - `src/dnadesign/usr/src/remote_sync/sidecars.py`
  - `src/dnadesign/usr/src/remote_sync/remote.py`
  - `src/dnadesign/usr/src/remote_sync/diff.py`
- Overlay/materialize and schema contracts:
  - `src/dnadesign/usr/src/datasets/lifecycle/__init__.py`
  - `src/dnadesign/usr/src/datasets/lifecycle/registry.py`
  - `src/dnadesign/usr/src/datasets/lifecycle/write_session.py`
  - `src/dnadesign/usr/src/datasets/merge/__init__.py`
  - `src/dnadesign/usr/src/datasets/merge/execution.py`
  - `src/dnadesign/usr/src/datasets/merge/overlay_carry.py`
  - `src/dnadesign/usr/src/datasets/overlay/__init__.py`
  - `src/dnadesign/usr/src/datasets/overlay/attach.py`
  - `src/dnadesign/usr/src/datasets/overlay/maintenance.py`
  - `src/dnadesign/usr/src/datasets/overlay/policy.py`
  - `src/dnadesign/usr/src/datasets/overlay/write.py`
  - `src/dnadesign/usr/src/datasets/materialize.py`
  - `src/dnadesign/usr/src/datasets/query/__init__.py`
  - `src/dnadesign/usr/src/datasets/query/catalog.py`
  - `src/dnadesign/usr/src/datasets/query/planner.py`
  - `src/dnadesign/usr/src/datasets/state/__init__.py`
  - `src/dnadesign/usr/src/datasets/state/facade.py`
  - `src/dnadesign/usr/src/datasets/validate/__init__.py`
  - `src/dnadesign/usr/src/datasets/validate/registry_modes.py`
  - `src/dnadesign/usr/src/datasets/views/__init__.py`
  - `src/dnadesign/usr/src/datasets/views/reporting.py`
  - `src/dnadesign/usr/src/datasets/views/read_keys.py`
  - `src/dnadesign/usr/src/overlay_support/digest_ledger.py`
  - `src/dnadesign/usr/src/registry.py`
  - `src/dnadesign/usr/src/schema.py`
- Legacy repair decomposition:
  - `src/dnadesign/usr/src/convert_legacy.py`
  - `src/dnadesign/usr/src/legacy/inputs.py`
  - `src/dnadesign/usr/src/legacy/tfbs.py`
  - `src/dnadesign/usr/src/legacy/dedupe.py`
- Behavior and reliability tests:
  - `src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`
  - `src/dnadesign/usr/tests/test_sync_schema_adversarial.py`
  - `src/dnadesign/usr/tests/test_convert_legacy_dedupe_module.py`
  - `src/dnadesign/usr/tests/test_usr_docs_contract.py`

## Open questions and risk notes

- `dataset.py` remains a large orchestration surface; the registry freeze/write-session cluster now lives under `datasets/lifecycle/*`, the merge engine now lives under `datasets/merge/*`, and additional extraction slices should continue around reserved-overlay paths.
- The layout contract now depends on helper families remaining under sanctioned subpackages; adding new top-level `usr/src/*.py` helpers should be treated as an architecture regression unless they are true coordinators.
- Closed helper clusters should keep nesting under their owning family instead of adding new sibling flat modules; `cli_commands/datasets/*`, `cli_commands/lifecycle/*`, `cli_commands/maintenance/*`, `cli_commands/namespace/*`, `cli_commands/query/*`, `cli_commands/read_views/*`, `cli_commands/remotes/*`, `cli_commands/sync/*`, `cli_commands/tooling/*`, `datasets/lifecycle/*`, `datasets/merge/*`, `datasets/overlay/*`, `datasets/query/*`, `datasets/state/*`, `datasets/validate/*`, and `datasets/views/*` are the current precedent.
- `repair_densegen_used_tfbs()` still combines multiple optional mutation/drop paths; next extraction slice should isolate single-TF and id/sequence-only drop policy handling.
- High-assurance hash mode can add runtime cost on very large overlay trees; operators should choose cadence based on transfer window constraints.
- Sync audit output is strong for decision support; adding machine-readable audit snapshots may further improve automated orchestration loops.
