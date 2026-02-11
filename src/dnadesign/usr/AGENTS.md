## `usr` for agents

Supplement to repo-root `AGENTS.md` with `usr`-specific contracts + navigation.

### Key paths
- Code (CLI + library): `src/dnadesign/usr/src/`
- Datasets root: `src/dnadesign/usr/datasets/`
  - Dataset layout (recommended):
    - `datasets/<namespace>/<name>/records.parquet` (canonical base table)
    - `datasets/<namespace>/<name>/_derived/` (derived overlays)
    - `datasets/<namespace>/<name>/meta.md` (hand-edited notes/snippets)
    - `datasets/<namespace>/<name>/.events.log` (append-only; generated)
    - `datasets/<namespace>/<name>/_snapshots/` (generated)
    - `datasets/_archive/<namespace>/<name>/...` (canonical archive location)
  - Legacy archive roots are not operational:
    - `datasets/archived/**` and `usr/archived/**` should be treated as historical only.
- Notebooks: `src/dnadesign/usr/notebooks/`
- Remote sync config: set `USR_REMOTES_PATH` to your remotes YAML
- Namespace registry: `registry.yaml` under the datasets root
- Sync details: `src/dnadesign/usr/docs/operations/sync.md`
- Historical artifacts: `src/dnadesign/usr/archived/` (treat as generated)

### Generated vs hand-edited
- Hand-edited: `datasets/**/meta.md`, `remotes.yaml`, `docs/operations/sync.md`
- Generated / run artifacts: `datasets/**/.events.log`, `datasets/**/_snapshots/**`, `datasets/**/_derived/**`, `archived/**`
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
uv run usr diff <dataset-or-path> <remote-name>
uv run usr pull <dataset-or-path> <remote-name> -y
uv run usr push <dataset-or-path> <remote-name> -y
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
