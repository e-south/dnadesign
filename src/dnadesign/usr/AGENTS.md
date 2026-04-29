## `usr` for agents

Supplement to repo-root `AGENTS.md` with `usr`-specific contracts + navigation.

### Key paths
- Code (CLI + library): `src/dnadesign/usr/src/`
  - Package-root Python should stay limited to `__init__.py` and `__main__.py`; public API exports belong on the package root, but implementation modules belong under `src/`.
  - `usr/src` root should contain package directories only plus `__init__.py`; do not reintroduce flat implementation files at that level.
  - Coordinator packages are reserved for high-level roots such as `api/`, `cli/`, `dataset/`, `maintenance/`, `sync/`, and `version/`.
  - Helper families should live under explicit subpackages: `cli/commands/`, `cli/support/`, `contracts/`, `datasets/`, `events/`, `genbank/`, `legacy/`, `overlays/support/`, `overlays/`, `registry/`, `runtime/`, `sequence_views/`, `storage/`, and `sync/remote/`.
  - `contracts/` owns shared error classes, schema constants, response dataclasses, and sequence normalization/id rules; do not reintroduce sibling root files for those concerns.
  - `api/` owns the internal public-library export facade that feeds `dnadesign.usr`.
  - `events/` owns event actor normalization, redaction, fingerprinting, payload defaults, and append-only logging; keep event helper concerns out of coordinator packages.
  - `genbank/` owns GenBank parsing, import manifests, source-hash fidelity, and optional feature extraction into USR overlays; keep source-annotation import concerns out of `dataset/` and `cli/`.
  - `dataset/` owns the main dataset coordinator surface and should stay as a package root over the `datasets/*` implementation families.
  - `cli/support/` owns CLI-only helper families and should stay split into `resolution/`, `wiring/`, and `presentation/` rather than re-accumulating a flat helper bucket.
  - `cli/support/resolution/` owns root/path resolution, dataset-target heuristics, and merge-policy lookup.
  - `cli/support/wiring/` owns dependency assembly, command bindings, Typer surface construction, and registration glue.
  - `cli/support/presentation/` owns output shaping, pretty formatting, rich/plain rendering, and stderr-noise suppression.
  - `cli/` owns the Typer coordinator surface and registration glue over `cli/commands/*` and `cli/support/*`.
  - `legacy/convert.py` owns DenseGen legacy conversion and repair entrypoints; keep legacy tool orchestration out of sibling flat roots.
  - `maintenance/` owns the maintenance-only mutation gate/context surface, even though the implementation is small.
  - `overlays/` owns overlay path discovery, part caching, and overlay schema/metadata helpers; `overlays/support/` remains the home for higher-level overlay digest/projection helpers that compose those primitives.
  - `sync/remote/` owns remote endpoint config/loading plus SSH stat, diff, execution orchestration, sidecar, and transfer helpers used by `sync/`.
  - `registry/` owns namespace registry models, YAML/cache/hash IO, and registry type/schema validation; keep registry mechanics out of `dataset/` and `cli/`.
  - `runtime/` owns reusable runtime helpers such as DuckDB session initialization and timezone contracts; keep runtime stateful helpers out of the root coordinator layer.
  - `sequence_views/` owns semantic sequence-view identity, stable `view_id` derivation, sidecar IO, and selector/store helpers; keep view-ontology mechanics out of `datasets/views/`.
  - `storage/` owns low-level parquet IO, snapshotting, and dataset write-lock primitives; do not reintroduce sibling root files for those concerns.
  - `sync/` owns the sync facade and runtime wiring over `sync/remote/*`.
  - `version/` owns the version singleton so even tiny constants do not leak back into the root.
  - When a helper family grows into a closed internal cluster, nest it again instead of re-flattening `usr/src/`; current sanctioned second-level packages are `cli/commands/datasets/`, `cli/commands/genbank/`, `cli/commands/lifecycle/`, `cli/commands/maintenance/`, `cli/commands/namespace/`, `cli/commands/query/`, `cli/commands/read_views/`, `cli/commands/remotes/`, `cli/commands/sync/`, `cli/commands/tooling/`, `cli/support/presentation/`, `cli/support/resolution/`, `cli/support/wiring/`, `datasets/core/`, `datasets/demo/`, `datasets/lifecycle/`, `datasets/maintenance/`, `datasets/merge/`, `datasets/overlay/`, `datasets/query/`, `datasets/state/`, `datasets/validate/`, `datasets/views/`, `overlays/support/`, and `sync/remote/`.
  - Tests should stay organized by family under `src/dnadesign/usr/tests/cli/`, `datasets/`, `legacy/`, `overlays/`, and `sync/`; keep root `tests/` for cross-cutting contracts only.
  - `tests/cli/` is allowed one more owned layer when a command family becomes a closed internal cluster; the current sanctioned CLI test buckets are `tests/cli/commands/`, `tests/cli/support/`, and `tests/cli/sync/`.
  - `tests/cli/commands/` must mirror command namespaces instead of becoming another flat bucket; current sanctioned command test buckets are `datasets/`, `lifecycle/`, `maintenance/`, `namespace/`, `query/`, `read_views/`, `remotes/`, and `tooling/`.
  - `tests/datasets/` is allowed one more owned layer when mirroring stable source families; the current sanctioned dataset test buckets are `tests/datasets/core/`, `tests/datasets/lifecycle/`, `tests/datasets/merge/`, `tests/datasets/overlay/`, `tests/datasets/query/`, `tests/datasets/state/`, `tests/datasets/validate/`, and `tests/datasets/views/`.
  - `tests/overlays/support/` mirrors higher-level overlay support helpers; low-level overlay primitive tests stay in `tests/overlays/`.
  - `tests/sync/remote/` mirrors remote endpoint, diff, transfer, and sidecar helpers; `tests/sync/` stays focused on the sync facade.
  - Shared test fixtures consumed outside USR belong under `src/dnadesign/devtools/tests/support/`, not under `dnadesign.usr.tests`.
- Ops integration: `src/dnadesign/usr/ops/`
  - Keep only Ops-facing provider glue, stable ops entrypoints, and status registries here; dataset/sync implementation stays under `src/`.
  - `src/dnadesign/usr/ops/sync_audit_drill.py` owns the stable deterministic sync drill entrypoint exposed as `uv run usr-sync-audit-drill`.
- Internal drills and harness helpers: `src/dnadesign/usr/scripts/`
  - These scripts may import `dnadesign.usr.src.*` because they are USR-owned internal tooling, not a public cross-tool API surface.
- Datasets root: `src/dnadesign/usr/datasets/`
  - Dataset layout (recommended):
    - `datasets/<dataset_id>/records.parquet` (canonical base table)
    - `datasets/<dataset_id>/_derived/` (derived overlays)
    - `datasets/<dataset_id>/meta.md` (hand-edited notes/snippets)
    - `datasets/<dataset_id>/.events.log` (append-only; generated)
    - `datasets/<dataset_id>/_snapshots/` (generated)
    - `datasets/archived/<dataset_id-or-qualified-path>/...` (canonical dataset archive location)
  - Archive handling:
    - `datasets/archived/**` is the sanctioned dataset archive root.
    - Archived datasets are not part of the default live dataset-id namespace under `datasets/`; target them by explicit path or by setting `--root` to `datasets/archived/` when operating inside the archive root.
    - `datasets/archived/promoter_misc_pytorch/` is the legacy promoter-focused PyTorch archive bucket (`.pt`, summary, and progress YAML artifacts), not a `records.parquet` dataset root.
- Demo fixtures: `src/dnadesign/usr/assets/demo_material/`
  - Stable example inputs for notebooks, mock generation, and CLI docs.
- Docs and media assets: `src/dnadesign/usr/assets/`
  - Static documentation/supporting assets only.
- Notebooks: `src/dnadesign/usr/notebooks/`
- Archived PyTorch manager: `src/dnadesign/usr/scripts/archived_pytorch_manager.py`
  - Inspects and updates the legacy PyTorch archive bucket under `datasets/archived/promoter_misc_pytorch/`.
- Remote sync config: prefer `uv run usr --remotes-config <remotes.yaml> ...`; `USR_REMOTES_PATH` is the fallback for shell-scoped sessions
- Namespace registry: `registry.yaml` under the datasets root
- Sync details: `src/dnadesign/usr/docs/operations/sync.md`
- Repo-local BU SCC sync skill: `.agents/skills/bu-scc-usr-sync/SKILL.md`
- Repo-local promoter-study status skill: `.agents/skills/promoter-study-status/SKILL.md`
  - Do not add tool-local skills under `src/dnadesign/usr/skills/`; the canonical skill root for this repo is `.agents/skills/`.
- Checked-in study records: `docs/studies/README.md`
- Active study registry: `docs/studies/index.yaml`
- Dataset naming ontology: active shared dataset ids must be flat owner-first ids such as `usr_mg1655_promoter_controls`, `usr_pdual10_plasmid_template`, or `densegen_prom_eth_cip_source`; use `root_kind`, `owner_tool`, overlays, and study metadata for provenance instead of nested tool-routing paths.
- Human-readable record names belong in `usr_label__primary` / `usr_label__aliases`.

### Generated vs hand-edited
- Hand-edited: `datasets/**/meta.md`, `remotes.yaml`, `docs/operations/sync.md`
- Generated / run artifacts: `datasets/**/.events.log`, `datasets/**/_snapshots/**`, `datasets/**/_derived/**`, `datasets/archived/**`
- Ask before committing: changed `records.parquet`, large datasets/logs, any bulk sync outputs/caches.

### Commands
```bash
uv run usr --help
uv run usr ls
uv run usr ls --format json
uv run usr info <dataset>
uv run usr head <dataset> -n 5
uv run usr schema <dataset> --tree
uv run usr validate <dataset> --strict
uv run usr delete <dataset> --id <id>
uv run usr restore <dataset> --id <id>
uv run usr namespace list
uv run usr namespace register <name> --columns <col:type,...>

# Attach namespaced columns
uv run usr attach <dataset> --path <file.csv> --namespace <tool> --key <id|sequence|sequence_norm|sequence_ci> --key-col <input_col> --columns <col1,col2>
uv run usr materialize <dataset>

# Maintenance subapp
uv run usr maintenance dedupe <dataset> --key sequence --keep keep-first
uv run usr maintenance merge --dest <dataset> --src <dataset>

# densegen subapp
uv run usr densegen repair --dedupe keep-first

# Export
uv run usr export <dataset> --fmt csv --out /tmp/out.csv
uv run usr materialize <dataset>

# Remote sync (see docs/operations/sync.md)
uv run usr --remotes-config <remotes.yaml> diff <dataset-or-path> <remote-name>
uv run usr --remotes-config <remotes.yaml> pull <dataset-or-path> <remote-name> -y
uv run usr --remotes-config <remotes.yaml> push <dataset-or-path> <remote-name> -y
# Fallback for a shell session:
export USR_REMOTES_PATH=<remotes.yaml>
```

### Notes
- macOS: PyArrow sysctl warnings are suppressed by default. Set `USR_SHOW_PYARROW_SYSCTL=1` to re-enable.
- The `datasets/demo` dataset is tracked. Copy it before running attach/materialize/snapshot if you want a scratch run.
- Update policy: base records are append-only; overlays are the only supported update path. Base rewrites are maintenance operations. In the library, use `with ds.maintenance(reason=...): ds.materialize(...)`.
- Reserved state overlay: `usr_state__masked`, `usr_state__qc_status`, `usr_state__split`, `usr_state__supersedes`, `usr_state__lineage` are standardized and registry-governed.
- Registry auto-freeze: when a registry exists, the first dataset mutation snapshots `_registry/registry.<hash>.yaml`.
- Tombstones: `usr__deleted_at` is stored as `timestamp[us, UTC]`.

### Tests

If you modify `usr`, run:

```bash
uv run pytest -q
```
