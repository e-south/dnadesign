## `seqledger` for agents

Supplement to repo-root `AGENTS.md` with SeqLedger-specific contracts + navigation.

### Key paths
- Code (CLI + library): `src/dnadesign/seqledger/src/`
  - Package-root Python should stay limited to `__init__.py` and `__main__.py`; public API exports belong on the package root, but implementation modules belong under `src/`.
  - `seqledger/src` root should contain package directories only plus `__init__.py`; do not reintroduce flat implementation files at that level.
  - Coordinator packages are reserved for high-level roots such as `api/`, `cli/`, `dataset/`, `maintenance/`, `sync/`, and `version/`.
  - Helper families should live under explicit subpackages: `cli/commands/`, `cli/support/`, `contracts/`, `datasets/`, `events/`, `genbank/`, `overlays/support/`, `overlays/`, `regulondb/`, `registry/`, `runtime/`, `sequence_views/`, `storage/`, and `sync/remote/`.
  - `contracts/` owns shared error classes, schema constants, response dataclasses, and sequence normalization/id rules; do not reintroduce sibling root files for those concerns.
  - `api/` owns the internal public-library export facade that feeds `dnadesign.seqledger`.
  - `events/` owns event actor normalization, redaction, fingerprinting, payload defaults, and append-only logging; keep event helper concerns out of coordinator packages.
  - `genbank/` owns GenBank parsing, import manifests, source-hash fidelity, and optional feature extraction into SeqLedger overlays; keep source-annotation import concerns out of `dataset/` and `cli/`.
  - `regulondb/` owns RegulonDB-specific overlay enrichment helpers such as source-backed functional annotation projection; keep source-specific enrichment logic out of generic dataset, registry, and sequence-view packages.
  - `dataset/` owns the main dataset coordinator surface and should stay as a package root over the `datasets/*` implementation families.
  - `cli/support/` owns CLI-only helper families and should stay split into `resolution/`, `wiring/`, and `presentation/` rather than re-accumulating a flat helper bucket.
  - `cli/support/resolution/` owns root/path resolution, dataset-target heuristics, and merge-policy lookup.
  - `cli/support/wiring/` owns dependency assembly, command bindings, Typer surface construction, and registration glue.
  - `cli/support/presentation/` owns output shaping, pretty formatting, rich/plain rendering, and stderr-noise suppression.
  - `cli/` owns the Typer coordinator surface and registration glue over `cli/commands/*` and `cli/support/*`.
  - `maintenance/` owns the maintenance-only mutation gate/context surface, even though the implementation is small.
  - `overlays/` owns overlay path discovery, part caching, and overlay schema/metadata helpers; `overlays/support/` remains the home for higher-level overlay digest/projection helpers that compose those primitives.
  - `sync/remote/` owns remote endpoint config/loading plus SSH stat, diff, execution orchestration, sidecar, and transfer helpers used by `sync/`.
  - `registry/` owns namespace registry models, YAML/cache/hash IO, and registry type/schema validation; keep registry mechanics out of `dataset/` and `cli/`.
  - `runtime/` owns reusable runtime helpers such as DuckDB session initialization and timezone contracts; keep runtime stateful helpers out of the root coordinator layer.
  - `sequence_views/` owns semantic sequence-view identity, stable `view_id` derivation, sidecar IO, and selector/store helpers; keep view-ontology mechanics out of `datasets/views/`.
  - `storage/` owns low-level parquet IO, snapshotting, and dataset write-lock primitives; do not reintroduce sibling root files for those concerns.
  - `sync/` owns the sync facade and runtime wiring over `sync/remote/*`.
  - `version/` owns the version singleton so even tiny constants do not leak back into the root.
  - When a helper family grows into a closed internal cluster, nest it again instead of re-flattening `usr/src/`; current sanctioned second-level packages are `cli/commands/datasets/`, `cli/commands/genbank/`, `cli/commands/lifecycle/`, `cli/commands/maintenance/`, `cli/commands/namespace/`, `cli/commands/query/`, `cli/commands/read_views/`, `cli/commands/remotes/`, `cli/commands/sync/`, `cli/commands/tooling/`, `cli/support/presentation/`, `cli/support/resolution/`, `cli/support/wiring/`, `datasets/core/`, `datasets/demo/`, `datasets/lifecycle/`, `datasets/maintenance/`, `datasets/merge/`, `datasets/overlay/`, `datasets/query/`, `datasets/state/`, `datasets/validate/`, `datasets/views/`, `overlays/support/`, `regulondb/`, and `sync/remote/`.
  - Tests should stay organized by family under `src/dnadesign/seqledger/tests/cli/`, `datasets/`, `overlays/`, and `sync/`; keep root `tests/` for cross-cutting contracts only.
  - `tests/cli/` is allowed one more owned layer when a command family becomes a closed internal cluster; the current sanctioned CLI test buckets are `tests/cli/commands/`, `tests/cli/support/`, and `tests/cli/sync/`.
  - `tests/cli/commands/` must mirror command namespaces instead of becoming another flat bucket; current sanctioned command test buckets are `datasets/`, `lifecycle/`, `maintenance/`, `namespace/`, `query/`, `read_views/`, `remotes/`, and `tooling/`.
  - `tests/datasets/` is allowed one more owned layer when mirroring stable source families; the current sanctioned dataset test buckets are `tests/datasets/core/`, `tests/datasets/lifecycle/`, `tests/datasets/merge/`, `tests/datasets/overlay/`, `tests/datasets/query/`, `tests/datasets/state/`, `tests/datasets/validate/`, and `tests/datasets/views/`.
  - `tests/overlays/support/` mirrors higher-level overlay support helpers; low-level overlay primitive tests stay in `tests/overlays/`.
  - `tests/sync/remote/` mirrors remote endpoint, diff, transfer, and sidecar helpers; `tests/sync/` stays focused on the sync facade.
  - Shared test fixtures consumed outside SeqLedger belong under `src/dnadesign/devtools/tests/support/`, not under `dnadesign.seqledger.tests`.
- Ops integration: `src/dnadesign/seqledger/ops/`
  - Keep only Ops-facing provider glue, stable ops entrypoints, and status registries here; dataset/sync implementation stays under `src/`.
  - `src/dnadesign/seqledger/ops/sync_audit_drill.py` owns the stable deterministic sync drill entrypoint exposed as `uv run seqledger-sync-audit-drill`.
- Internal drills and harness helpers: `src/dnadesign/seqledger/scripts/`
  - These scripts may import `dnadesign.seqledger.src.*` because they are SeqLedger-owned internal tooling, not a public cross-tool API surface.
- Default datasets root: `./seqledger_datasets/`; `--root` or `DNADESIGN_SEQLEDGER_ROOT` selects existing storage.
  - The checkout-era `src/dnadesign/seqledger/datasets/` root remains accessible explicitly during migration.
  - Dataset layout (recommended):
    - `datasets/<dataset_id>/records.parquet` (canonical base table)
    - `datasets/<dataset_id>/_derived/` (derived overlays)
    - `datasets/<dataset_id>/meta.md` (hand-edited notes/snippets)
    - `datasets/<dataset_id>/.events.log` (append-only; generated)
    - `datasets/<dataset_id>/.events.lock` (local event-log coordination; generated, never synced)
    - `datasets/<dataset_id>/_snapshots/` (generated)
    - `datasets/archived/<dataset_id-or-qualified-path>/...` (canonical dataset archive location)
  - Archive handling:
    - `datasets/archived/**` is the sanctioned dataset archive root.
    - Archived datasets are not part of the default live dataset-id namespace under `datasets/`; target them by explicit path or by setting `--root` to `datasets/archived/` when operating inside the archive root.
- Demo fixtures: `src/dnadesign/seqledger/assets/demo_material/`
  - Stable example inputs for notebooks, mock generation, and CLI docs.
- Docs and media assets: `src/dnadesign/seqledger/assets/`
  - Static documentation/supporting assets only.
- Notebooks: `src/dnadesign/seqledger/notebooks/`
- Remote sync config: copy `remotes.example.yaml` to an untracked local path,
  replace its example values, and pass that path with
  `uv run seqledger --remotes-config <remotes.yaml> ...`; `SEQLEDGER_REMOTES_PATH` is the
  fallback for shell-scoped sessions
- Namespace registry: `registry.yaml` under the datasets root
- Sync details: `src/dnadesign/seqledger/docs/operations/sync/README.md`
- Repo-local BU SCC sync skill: `.agents/skills/bu-scc-seqledger-sync/SKILL.md`
  - Do not add tool-local skills under `src/dnadesign/seqledger/skills/`; the canonical skill root for this repo is `.agents/skills/`.
- External study integration: `docs/integrations/study-workspaces.md`
- Dataset naming ontology: active shared dataset ids must be flat owner-first ids such as `usr_mg1655_promoter_controls`, `usr_pdual10_plasmid_template`, or `densegen_prom_eth_cip_source`; use `root_kind`, `owner_tool`, overlays, and study metadata for provenance instead of nested tool-routing paths.
- Human-readable record names belong in `usr_label__primary` / `usr_label__aliases`.

### Generated vs hand-edited
- Hand-edited: `datasets/**/meta.md`, an untracked local `remotes.yaml`, `src/dnadesign/seqledger/docs/operations/sync/README.md`
- Generated / run artifacts: `datasets/**/.events.log`, `datasets/**/.events.lock`, `datasets/**/_snapshots/**`, `datasets/**/_derived/**`, `datasets/archived/**`
- Ask before committing: changed `records.parquet`, large datasets/logs, any bulk sync outputs/caches.

### Commands
```bash
uv run seqledger --help
uv run seqledger ls
uv run seqledger ls --format json
uv run seqledger info <dataset>
uv run seqledger head <dataset> -n 5
uv run seqledger schema <dataset> --tree
uv run seqledger validate <dataset> --strict
uv run seqledger delete <dataset> --id <id>
uv run seqledger restore <dataset> --id <id>
uv run seqledger namespace list
uv run seqledger namespace register <name> --columns <col:type,...>

# Attach namespaced columns
uv run seqledger attach <dataset> --path <file.csv> --namespace <tool> --key <id|sequence|sequence_norm|sequence_ci> --key-col <input_col> --columns <col1,col2>
uv run seqledger materialize <dataset>

# Maintenance subapp
uv run seqledger maintenance dedupe <dataset> --key sequence --keep keep-first
uv run seqledger maintenance merge --dest <dataset> --src <dataset>


# Export
uv run seqledger export <dataset> --fmt csv --out /tmp/out.csv
uv run seqledger materialize <dataset>

# Remote sync (see docs/operations/sync/README.md)
uv run seqledger --remotes-config <remotes.yaml> diff <dataset-or-path> <remote-name>
uv run seqledger --remotes-config <remotes.yaml> pull <dataset-or-path> <remote-name> -y
uv run seqledger --remotes-config <remotes.yaml> push <dataset-or-path> <remote-name> -y
# Fallback for a shell session:
export SEQLEDGER_REMOTES_PATH=<remotes.yaml>
```

### Notes
- macOS: PyArrow sysctl warnings are suppressed by default. Set `SEQLEDGER_SHOW_PYARROW_SYSCTL=1` to re-enable.
- Demo inputs are tracked under `assets/demo_material/`. Import them into a scratch dataset before running attach, materialize, or snapshot commands.
- Update policy: base records are append-only; overlays are the only supported update path. Base rewrites are maintenance operations. In the library, use `with ds.maintenance(reason=...): ds.materialize(...)`.
- Reserved state overlay: `usr_state__masked`, `usr_state__qc_status`, `usr_state__split`, `usr_state__supersedes`, `usr_state__lineage` are standardized and registry-governed.
- Registry auto-freeze: when a registry exists, the first dataset mutation snapshots `_registry/registry.<hash>.yaml`.
- Tombstones: `usr__deleted_at` is stored as `timestamp[us, UTC]`.

### Tests

Start with the affected test family and the matching static checks in the
[repository change gate](../../../docs/repo-change-gate.md). For example, changes
to CLI presentation use:

```bash
uv run pytest -q src/dnadesign/seqledger/tests/cli/support
```

Use the complete SeqLedger suite for changes spanning multiple SeqLedger subsystems. Reserve
the repository-wide suite for cross-tool behavior changes or merge verification.
