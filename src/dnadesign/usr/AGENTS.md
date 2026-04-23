## `usr` for agents

Supplement to repo-root `AGENTS.md` with `usr`-specific contracts + navigation.

### Key paths
- Code (CLI + library): `src/dnadesign/usr/src/`
  - Package-root Python should stay limited to `__init__.py` and `__main__.py`; public API exports belong on the package root, but implementation modules belong under `src/`.
  - Root coordinators are reserved for high-level modules such as `api.py`, `cli.py`, `dataset.py`, `sync.py`, and `convert_legacy.py`.
  - Helper families should live under explicit subpackages: `cli_commands/`, `cli_support/`, `datasets/`, `legacy/`, `overlay_support/`, `remote_sync/`, and `storage/`.
  - `cli_support/` owns CLI-only wiring and presentation helpers such as root/path resolution, bindings, output shaping, pretty formatting, rich/plain rendering, and stderr-noise suppression.
  - `remote_sync/` owns remote endpoint config/loading plus SSH stat, diff, execution orchestration, sidecar, and transfer helpers used by `sync.py`.
  - `storage/` owns low-level parquet IO, snapshotting, and dataset write-lock primitives; do not reintroduce sibling root files for those concerns.
  - When a helper family grows into a closed internal cluster, nest it again instead of re-flattening `usr/src/`; current sanctioned second-level packages are `cli_commands/datasets/`, `cli_commands/lifecycle/`, `cli_commands/maintenance/`, `cli_commands/namespace/`, `cli_commands/query/`, `cli_commands/read_views/`, `cli_commands/remotes/`, `cli_commands/sync/`, `cli_commands/tooling/`, `datasets/lifecycle/`, `datasets/merge/`, `datasets/overlay/`, `datasets/query/`, `datasets/state/`, `datasets/validate/`, and `datasets/views/`.
  - Tests should stay shallowly organized by family under `src/dnadesign/usr/tests/cli/`, `datasets/`, `sync/`, `remote_sync/`, and `legacy/`; keep root `tests/` for cross-cutting contracts and shared helpers.
- Ops integration: `src/dnadesign/usr/ops/`
  - Keep only Ops-facing provider glue and status registries here; dataset/sync implementation stays under `src/`.
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
- Dataset naming ontology: prefer the least-coupled semantic dataset ids, usually flat ids such as `mg1655_promoters`, `plasmids`, or `anchor_template_slot_a_window_1kb_demo`; use namespace-qualified ids only when they genuinely improve disambiguation. Keep tool provenance in overlay namespaces, not dataset ids.
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
