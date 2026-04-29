# USR dataset layout and code map

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-24


## Dataset layout

```text
src/dnadesign/usr/
├─ __init__.py, __main__.py  # package root only; public API is exported from __init__.py
├─ assets/
│  ├─ usr-banner.svg
│  └─ demo_material/
├─ src/
├─ datasets/
│  ├─ <dataset_id>/
│  │  ├─ records.parquet
│  │  ├─ _derived/
│  │  ├─ meta.md
│  │  ├─ .events.log
│  │  ├─ _registry/
│  │  └─ _snapshots/
│  ├─ <namespace>/<dataset_name>/...
│  └─ archived/
│     ├─ <dataset_id-or-qualified-path>/...
│     └─ promoter_misc_pytorch/
```

Dataset ids may be flat (`dataset`) or namespace-qualified (`namespace/dataset`).
Active shared study dataset ids must be flat owner-first ids such as
`densegen_prom_eth_cip_source`, `usr_prom_eth_cip_anchor`,
`construct_prom_eth_cip_context`, and `usr_prom_eth_cip_matrix`.

Choose the least-coupled semantic id that still keeps the dataset understandable.

- Prefer a flat dataset id when the biological collection is already specific, such as `usr_promoter_references`, `usr_pdual10_plasmid_template`, or `anchor_template_slot_a_window_1kb_demo`.
- Use namespace-qualified ids only when they genuinely improve disambiguation instead of encoding tool routing.
- Do not encode active handoffs as nested owner/study folders. Use `root_kind`, `owner_tool`, overlays, and study metadata for provenance.
- Keep tool provenance in namespaced overlay columns such as `construct__*`, `densegen__*`, or `infer__*`, not in the dataset id itself, unless the dataset is truly tool-private scratch state.
- Carry human-readable record names in record columns such as `usr_label__primary` / `usr_label__aliases`, not only in local notes or workspace conventions.

`archived/` is the canonical dataset archive root. Archived datasets are intentionally excluded from the default live dataset-id namespace under `datasets/`, but explicit paths under `datasets/archived/**` remain supported. `datasets/archived/promoter_misc_pytorch/` is a legacy promoter-focused `.pt` archive bucket and is not a `records.parquet` dataset root.

## Maintainer code map

Package-root public facades:

- `src/dnadesign/usr/__init__.py`
- `src/dnadesign/usr/__main__.py`

Public package exports are re-exported from `src/dnadesign/usr/__init__.py`; avoid adding sibling
root implementation files such as `dataset.py` or `roots.py`.

High-level coordinator packages at `src/dnadesign/usr/src/`:

- `src/dnadesign/usr/src/api/__init__.py`
- `src/dnadesign/usr/src/cli/__init__.py`
- `src/dnadesign/usr/src/legacy/convert.py`
- `src/dnadesign/usr/src/dataset/__init__.py`
- `src/dnadesign/usr/src/maintenance/__init__.py`
- `src/dnadesign/usr/src/sync/__init__.py`
- `src/dnadesign/usr/src/version/__init__.py`

Helper packages under `src/dnadesign/usr/src/`:

- `api/`
  - internal public-library facade that re-exports the sanctioned USR library surface
- `cli/`
  - Typer CLI coordinator and registration surface over `cli/commands/*` and `cli/support/*`
- `cli/commands/`
  - command registration and user-facing command handlers
  - `cli/commands/datasets/` is the closed CLI dataset-target cluster for listing live datasets, canonical-id resolution, and interactive dataset selection
  - `cli/commands/lifecycle/` is the closed lifecycle command cluster for dataset init/import/attach, state mutation, materialize flow, snapshot, and Typer registration
  - `cli/commands/maintenance/` is the closed maintenance command cluster for registry/overlay maintenance, dedupe, merge, and Typer registration
  - `cli/commands/namespace/` is the closed namespace registry cluster for handler logic and Typer registration
  - `cli/commands/query/` is the closed query command cluster for Typer registration plus `ls`/`info`/`schema` and runtime query handlers
  - `cli/commands/read_views/` is the closed read-view cluster for `head`, `cols`, `describe`, `cell`, and parquet-target selection helpers
  - `cli/commands/remotes/` is the closed remotes cluster for SSH remote management handlers and Typer registration
  - `cli/commands/sync/` is the closed sync command cluster for diff/pull/push registration, policy, target resolution, execution, and audit output
  - `cli/commands/tooling/` is the closed tooling command cluster for DenseGen repair, dev-only mock/demo helpers, legacy import handlers, and Typer registration
- `cli/support/`
  - CLI helper families grouped as `cli/support/resolution/`, `cli/support/wiring/`, and `cli/support/presentation/`
- `src/dnadesign/usr/src/cli/support/resolution/dataset_targets.py`
- `src/dnadesign/usr/src/cli/support/resolution/roots.py`
- `src/dnadesign/usr/src/cli/support/wiring/dependencies.py`
- `src/dnadesign/usr/src/cli/support/wiring/registration.py`
- `src/dnadesign/usr/src/cli/support/presentation/pretty.py`
- `src/dnadesign/usr/src/cli/support/presentation/rendering.py`
- `src/dnadesign/usr/src/cli/support/presentation/runtime.py`
- `contracts/`
  - shared contract surfaces for errors, schema constants, API response dataclasses, and sequence normalization/id rules
- `src/dnadesign/usr/src/contracts/__init__.py`
- `src/dnadesign/usr/src/contracts/errors.py`
- `src/dnadesign/usr/src/contracts/normalize.py`
- `src/dnadesign/usr/src/contracts/schema.py`
- `src/dnadesign/usr/src/contracts/types.py`
- `legacy/convert.py`
  - DenseGen legacy conversion and repair facade over legacy parsing helpers
- `dataset/`
  - dataset coordinator surface over lifecycle, overlay, query, validate, and view helpers
- `events/`
  - event actor normalization, argument redaction, parquet fingerprinting, payload defaults, and append-only logging
- `src/dnadesign/usr/src/events/__init__.py`
- `src/dnadesign/usr/src/events/actor.py`
- `src/dnadesign/usr/src/events/defaults.py`
- `src/dnadesign/usr/src/events/fingerprint.py`
- `src/dnadesign/usr/src/events/recording.py`
- `src/dnadesign/usr/src/events/redaction.py`
- `genbank/`
  - GenBank parsing, source-hash fidelity, annotation import manifests, and optional feature extraction into USR overlays
- `src/dnadesign/usr/src/genbank/__init__.py`
- `src/dnadesign/usr/src/genbank/importer.py`
- `src/dnadesign/usr/src/genbank/models.py`
- `src/dnadesign/usr/src/genbank/parser.py`
- `datasets/`
  - dataset helper packages for core ingest/activity primitives, materialization, maintenance-gated mutation operations, overlay operations, validation, and read/query flows
  - `datasets/core/` is the closed dataset core cluster for identity, ingest, activity notes, and dataset-scoped events
  - `datasets/demo/` is the closed demo/mock dataset generation cluster
  - `datasets/lifecycle/` is the closed dataset lifecycle cluster for registry freeze/state helpers, materialization, snapshot coordination, and write-session orchestration
  - `datasets/maintenance/` is the closed maintenance-gated mutation cluster for operations such as dedupe
  - `datasets/merge/` is the closed dataset merge cluster for maintenance-gated merge execution and overlay-carry planning
  - `datasets/overlay/` is the closed overlay mutation cluster for policy, attach/write flows, and maintenance
  - `datasets/query/` is the closed overlay-query cluster for SQL helpers, overlay-catalog loading, and DuckDB overlay planning
  - `datasets/state/` is the closed dataset state/tombstone cluster for state lifecycle core and facade helpers
  - `datasets/validate/` is the closed validation cluster for dataset integrity checks and registry-mode policy
  - `datasets/views/` is the closed dataset read/export cluster for scan/head/get/grep/export entrypoints plus reporting and read-key helpers
- `src/dnadesign/usr/src/datasets/core/__init__.py`
- `src/dnadesign/usr/src/datasets/core/activity.py`
- `src/dnadesign/usr/src/datasets/core/events.py`
- `src/dnadesign/usr/src/datasets/core/identity.py`
- `src/dnadesign/usr/src/datasets/core/ingest.py`
- `src/dnadesign/usr/src/datasets/demo/__init__.py`
- `src/dnadesign/usr/src/datasets/demo/mock.py`
- `src/dnadesign/usr/src/datasets/lifecycle/__init__.py`
- `src/dnadesign/usr/src/datasets/lifecycle/materialize.py`
- `src/dnadesign/usr/src/datasets/lifecycle/registry.py`
- `src/dnadesign/usr/src/datasets/lifecycle/snapshot.py`
- `src/dnadesign/usr/src/datasets/lifecycle/write_session.py`
- `src/dnadesign/usr/src/datasets/maintenance/__init__.py`
- `src/dnadesign/usr/src/datasets/maintenance/dedupe.py`
- `src/dnadesign/usr/src/datasets/merge/__init__.py`
- `src/dnadesign/usr/src/datasets/merge/execution.py`
- `src/dnadesign/usr/src/datasets/merge/overlay_carry.py`
- `src/dnadesign/usr/src/datasets/overlay/__init__.py`
- `src/dnadesign/usr/src/datasets/overlay/attach.py`
- `src/dnadesign/usr/src/datasets/overlay/maintenance.py`
- `src/dnadesign/usr/src/datasets/overlay/policy.py`
- `src/dnadesign/usr/src/datasets/overlay/write.py`
- `runtime/`
  - reusable runtime helpers such as DuckDB UTC session initialization
- `src/dnadesign/usr/src/runtime/__init__.py`
- `src/dnadesign/usr/src/runtime/duckdb.py`
- `src/dnadesign/usr/src/cli/commands/datasets/__init__.py`
- `src/dnadesign/usr/src/cli/commands/datasets/catalog.py`
- `src/dnadesign/usr/src/cli/commands/datasets/resolution.py`
- `src/dnadesign/usr/src/cli/commands/lifecycle/__init__.py`
- `src/dnadesign/usr/src/cli/commands/lifecycle/cli.py`
- `src/dnadesign/usr/src/cli/commands/lifecycle/materialize.py`
- `src/dnadesign/usr/src/cli/commands/lifecycle/snapshot.py`
- `src/dnadesign/usr/src/cli/commands/lifecycle/state.py`
- `src/dnadesign/usr/src/cli/commands/lifecycle/write.py`
- `src/dnadesign/usr/src/cli/commands/maintenance/__init__.py`
- `src/dnadesign/usr/src/cli/commands/maintenance/cli.py`
- `src/dnadesign/usr/src/cli/commands/maintenance/dedupe.py`
- `src/dnadesign/usr/src/cli/commands/maintenance/merge.py`
- `src/dnadesign/usr/src/cli/commands/maintenance/overlay.py`
- `src/dnadesign/usr/src/cli/commands/maintenance/registry.py`
- `src/dnadesign/usr/src/cli/commands/namespace/__init__.py`
- `src/dnadesign/usr/src/cli/commands/namespace/cli.py`
- `src/dnadesign/usr/src/cli/commands/query/__init__.py`
- `src/dnadesign/usr/src/cli/commands/query/cli.py`
- `src/dnadesign/usr/src/cli/commands/query/read.py`
- `src/dnadesign/usr/src/cli/commands/query/runtime.py`
- `src/dnadesign/usr/src/cli/commands/read_views/__init__.py`
- `src/dnadesign/usr/src/cli/commands/read_views/parquet_targets.py`
- `src/dnadesign/usr/src/cli/commands/remotes/__init__.py`
- `src/dnadesign/usr/src/cli/commands/remotes/cli.py`
- `src/dnadesign/usr/src/cli/commands/sync/__init__.py`
- `src/dnadesign/usr/src/cli/commands/sync/cli.py`
- `src/dnadesign/usr/src/cli/commands/tooling/__init__.py`
- `src/dnadesign/usr/src/cli/commands/tooling/cli.py`
- `src/dnadesign/usr/src/cli/commands/tooling/densegen.py`
- `src/dnadesign/usr/src/cli/commands/tooling/dev.py`
- `src/dnadesign/usr/src/cli/commands/tooling/legacy.py`
- `src/dnadesign/usr/src/cli/commands/tooling/shared.py`
- `src/dnadesign/usr/src/datasets/query/__init__.py`
- `src/dnadesign/usr/src/datasets/query/catalog.py`
- `src/dnadesign/usr/src/datasets/query/planner.py`
- `src/dnadesign/usr/src/datasets/state/__init__.py`
- `src/dnadesign/usr/src/datasets/state/facade.py`
- `src/dnadesign/usr/src/datasets/state/reserved_overlay.py`
- `src/dnadesign/usr/src/datasets/validate/__init__.py`
- `src/dnadesign/usr/src/datasets/validate/registry_modes.py`
- `src/dnadesign/usr/src/datasets/views/__init__.py`
- `src/dnadesign/usr/src/datasets/views/reporting.py`
- `src/dnadesign/usr/src/datasets/views/read_keys.py`
- `legacy/`
  - DenseGen `.pt` import helpers and repair/decomposition utilities
- `maintenance/`
  - maintenance-only mutation gate/context surface
- `overlays/support/`
  - overlay-digest and projection helpers that support the dataset coordinator
- `overlays/`
  - overlay path discovery, part caching, and overlay metadata/schema helpers
- `src/dnadesign/usr/src/overlays/__init__.py`
- `src/dnadesign/usr/src/overlays/constants.py`
- `src/dnadesign/usr/src/overlays/metadata.py`
- `src/dnadesign/usr/src/overlays/paths.py`
- `sync/remote/`
  - remote config loading plus remote-stat, diff, execution orchestration, sidecar, and staged-transfer helpers used by `sync/`
- `registry/`
  - namespace registry models, YAML/cache/hash helpers, and registry validation/type parsing
- `src/dnadesign/usr/src/registry/__init__.py`
- `src/dnadesign/usr/src/registry/models.py`
- `src/dnadesign/usr/src/registry/storage.py`
- `src/dnadesign/usr/src/registry/typespec.py`
- `src/dnadesign/usr/src/registry/validation.py`
- `sequence_views/`
  - semantic sequence-view identity, view-id derivation, parquet sidecar IO, and selector/store helpers
- `src/dnadesign/usr/src/sequence_views/__init__.py`
- `src/dnadesign/usr/src/sequence_views/models.py`
- `src/dnadesign/usr/src/sequence_views/store.py`
- `storage/`
  - low-level parquet IO, snapshotting, and dataset write-lock primitives
- `sync/`
  - sync facade over `sync/remote/*` execution/runtime helpers
- `version/`
  - version singleton package

Coordinator packages should stay thin and compose the helper packages instead of accumulating new
sibling flat files. When a helper cluster becomes internally cohesive, prefer a nested helper
package over another root-level or first-level flat file.

Internal archive tooling:

- `src/dnadesign/usr/scripts/archived_pytorch_manager.py`
  - USR-owned maintenance helper for the legacy `datasets/archived/promoter_misc_pytorch/` bucket; not a public cross-tool API surface.
- `src/dnadesign/usr/ops/sync_audit_drill.py`
  - stable ops-facing deterministic sync drill entrypoint behind `uv run usr-sync-audit-drill`.

Shared repo-level test support:

- `src/dnadesign/devtools/tests/support/usr.py`
  - shared USR registry fixture helpers for sibling package tests; cross-tool tests should not import `dnadesign.usr.tests.*`.

## Next steps

- Schema details: [schema-contract.md](schema-contract.md)
- Overlay and registry semantics: [overlay-and-registry.md](overlay-and-registry.md)
