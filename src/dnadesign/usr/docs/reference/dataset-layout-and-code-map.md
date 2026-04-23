# USR dataset layout and code map

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-23


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

Choose the least-coupled semantic id that still keeps the dataset understandable.

- Prefer a flat dataset id when the biological collection is already specific, such as `mg1655_promoters`, `plasmids`, or `anchor_template_slot_a_window_1kb_demo`.
- Use namespace-qualified ids only when they genuinely improve disambiguation instead of encoding tool routing.
- Keep tool provenance in namespaced overlay columns such as `construct__*`, `densegen__*`, or `infer__*`, not in the dataset id itself, unless the dataset is truly tool-private scratch state.
- Carry human-readable record names in record columns such as `usr_label__primary` / `usr_label__aliases`, not only in local notes or workspace conventions.

`archived/` is the canonical dataset archive root. Archived datasets are intentionally excluded from the default live dataset-id namespace under `datasets/`, but explicit paths under `datasets/archived/**` remain supported. `datasets/archived/promoter_misc_pytorch/` is a legacy promoter-focused `.pt` archive bucket and is not a `records.parquet` dataset root.

## Maintainer code map

Package-root public facades:

- `src/dnadesign/usr/__init__.py`
- `src/dnadesign/usr/__main__.py`

Public package exports are re-exported from `src/dnadesign/usr/__init__.py`; avoid adding sibling root modules such as `dataset.py` or `roots.py`.

High-level coordinators at `src/dnadesign/usr/src/`:

- `api.py`
- `cli.py`
- `src/dnadesign/usr/src/dataset.py`
- `src/dnadesign/usr/src/sync.py`
- `src/dnadesign/usr/src/convert_legacy.py`

Helper packages under `src/dnadesign/usr/src/`:

- `cli_commands/`
  - command registration and user-facing command handlers
  - `cli_commands/datasets/` is the closed CLI dataset-target cluster for listing live datasets, canonical-id resolution, and interactive dataset selection
  - `cli_commands/lifecycle/` is the closed lifecycle command cluster for dataset init/import/attach, state mutation, materialize flow, snapshot, and Typer registration
  - `cli_commands/maintenance/` is the closed maintenance command cluster for registry/overlay maintenance, dedupe, merge, and Typer registration
  - `cli_commands/namespace/` is the closed namespace registry cluster for handler logic and Typer registration
  - `cli_commands/query/` is the closed query command cluster for Typer registration plus `ls`/`info`/`schema` and runtime query handlers
  - `cli_commands/read_views/` is the closed read-view cluster for `head`, `cols`, `describe`, `cell`, and parquet-target selection helpers
  - `cli_commands/remotes/` is the closed remotes cluster for SSH remote management handlers and Typer registration
  - `cli_commands/sync/` is the closed sync command cluster for diff/pull/push registration, policy, target resolution, execution, and audit output
  - `cli_commands/tooling/` is the closed tooling command cluster for DenseGen repair, dev-only mock/demo helpers, legacy import handlers, and Typer registration
- `cli_support/`
  - CLI wiring helpers such as root/path resolution, bindings, merge policy, output shaping, pretty formatting, rich/plain rendering, and stderr-noise suppression
- `src/dnadesign/usr/src/cli_support/pretty.py`
- `src/dnadesign/usr/src/cli_support/rendering.py`
- `src/dnadesign/usr/src/cli_support/roots.py`
- `datasets/`
  - dataset helper modules for ingest, materialization, maintenance-gated merge operations, overlay operations, validation, and read/query flows
  - `datasets/lifecycle/` is the closed dataset lifecycle cluster for registry freeze/state helpers and write-session orchestration
  - `datasets/merge/` is the closed dataset merge cluster for maintenance-gated merge execution and overlay-carry planning
  - `datasets/overlay/` is the closed overlay mutation cluster for policy, attach/write flows, and maintenance
  - `datasets/query/` is the closed overlay-query cluster for SQL helpers, overlay-catalog loading, and DuckDB overlay planning
  - `datasets/state/` is the closed dataset state/tombstone cluster for state lifecycle core and facade helpers
  - `datasets/validate/` is the closed validation cluster for dataset integrity checks and registry-mode policy
  - `datasets/views/` is the closed dataset read/export cluster for scan/head/get/grep/export entrypoints plus reporting and read-key helpers
- `src/dnadesign/usr/src/datasets/activity.py`
- `src/dnadesign/usr/src/datasets/lifecycle/__init__.py`
- `src/dnadesign/usr/src/datasets/lifecycle/registry.py`
- `src/dnadesign/usr/src/datasets/lifecycle/write_session.py`
- `src/dnadesign/usr/src/datasets/merge/__init__.py`
- `src/dnadesign/usr/src/datasets/merge/execution.py`
- `src/dnadesign/usr/src/datasets/merge/overlay_carry.py`
- `src/dnadesign/usr/src/datasets/materialize.py`
- `src/dnadesign/usr/src/datasets/overlay/__init__.py`
- `src/dnadesign/usr/src/datasets/overlay/attach.py`
- `src/dnadesign/usr/src/datasets/overlay/maintenance.py`
- `src/dnadesign/usr/src/datasets/overlay/policy.py`
- `src/dnadesign/usr/src/datasets/overlay/write.py`
- `src/dnadesign/usr/src/datasets/dedupe.py`
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
- `legacy/`
  - DenseGen `.pt` import helpers and repair/decomposition utilities
- `overlay_support/`
  - overlay-digest and projection helpers that support the dataset coordinator
- `remote_sync/`
  - remote config loading plus remote-stat, diff, execution orchestration, sidecar, and staged-transfer helpers used by `sync.py`
- `storage/`
  - low-level parquet IO, snapshotting, and dataset write-lock primitives

Root coordinators should stay thin and compose the helper packages instead of accumulating new sibling flat files.
When a helper cluster becomes internally cohesive, prefer a nested helper package over another root-level or first-level flat file.

Internal archive tooling:

- `src/dnadesign/usr/scripts/archived_pytorch_manager.py`
  - USR-owned maintenance helper for the legacy `datasets/archived/promoter_misc_pytorch/` bucket; not a public cross-tool API surface.

## Next steps

- Schema details: [schema-contract.md](schema-contract.md)
- Overlay and registry semantics: [overlay-and-registry.md](overlay-and-registry.md)
