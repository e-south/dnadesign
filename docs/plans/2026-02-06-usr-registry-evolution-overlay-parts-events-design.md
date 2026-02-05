# USR Registry Evolution, Overlay Parts, and Event Semantics

## Intent
Strengthen USR for long-running, write-heavy tools (densegen) while keeping the core contract intact: base table is canonical, overlays are derived, events are audit records, and validation is strict. This design adds registry evolution support, append-only overlay parts, and structured events suitable for webhook adapters without adding top-level CLI sprawl or heavyweight imports to `dnadesign.usr`.

## Scope
- In:
  - Registry hashing and dataset-local registry snapshots with validation modes.
  - Append-only overlay parts for write-heavy namespaces, with compaction as maintenance.
  - Structured events with actor fields and stable JSON output for tails.
  - Densegen adapter update to write Arrow-native complex types and overlay parts.
  - Notify adapter that consumes USR events (outside USR core).
- Out:
  - Changes to the dataset on-disk contract (records.parquet remains canonical).
  - Reintroduction of pandas overlay merge paths.
  - New top-level CLI commands.

## Current Constraints
- `import dnadesign.usr` must remain light (no Typer/Rich, no DuckDB import unless used).
- No silent fallbacks; errors must be explicit.
- Registry enforcement remains strict and repo-wide.

## Registry Evolution
### Metadata
- Write `usr:registry_hash` into Parquet metadata for:
  - `records.parquet`
  - every overlay (file or part)
- Hash is `sha256` of the normalized registry YAML bytes (sorted keys, stable encoding).

### Freeze
- Add dataset-local registry snapshots under `_registry/registry.<hash>.yaml`.
- Add `Dataset.freeze_registry()` (maintenance-gated) to write the snapshot and update base metadata.

### Validation modes
- `current`: validate against repo registry.
- `frozen`: validate against dataset’s frozen registry hash (error if none).
- `either`: accept if either current or frozen matches.

## Overlay Parts (Append-Only)
### Layout
- Support either:
  - `_derived/<namespace>.parquet` (single file)
  - `_derived/<namespace>/part-*.parquet` (directory overlay)
- Directory overlays take precedence if present; simultaneous file + directory is an error.

### Write path
- Add `Dataset.write_overlay_part(...)`:
  - validate schema against registry once per write
  - write a new immutable `part-<timestamp>-<uuid>.parquet`
  - record a single event with rows/bytes and registry hash

### Read path
- DuckDB joins read from either file or part glob.
- Overlay metadata is derived from the Parquet file metadata in each part.

### Compaction
- Add `Dataset.compact_overlay(namespace, maintenance=True)`:
  - merge parts into a single file (or fewer parts)
  - archive old parts under `_derived/_archived/<namespace>/<ts>/...`

## Registry Type Support for Complex Dtypes
Extend registry parsing to include Arrow nested types:
- `fixed_size_list<float32>[D]`
- `struct<field1:type,field2:type>`
- nested compositions such as `list<struct<...>>`

Type parsing is strict and canonicalized. `arrow_type_str()` must round-trip to registry strings deterministically.

## Events (Structured, Integration-Ready)
- Add `event_version: 1`.
- Add `actor`: `tool`, `run_id`, `host`, `pid` (optional fields).
- Include `dataset` identifiers and `registry_hash`.
- Include action metrics (rows, bytes, namespace).
- Add `usr events tail --format json --follow` with one JSON object per line and no rich formatting.

## Densegen Adapter
Update `densegen/src/adapters/outputs/usr_writer.py`:
- Use `write_overlay_part` for derived columns.
- Emit Arrow-native complex types; do not JSON stringify.
- For large or irregular arrays, allow explicit `.npz` artifact refs with typed metadata columns.
- Populate actor fields (`tool=densegen`, `run_id=<run_id>`).

## Notify Adapter
Add a small adapter in `notify/` that tails `.events.log` JSONL and calls `notify send`. This keeps notify optional and decoupled from heavy HPC runs (can be executed on login nodes or as a post-run step).

## Error Handling and Invariants
- No file/dir overlay ambiguity: error if both exist.
- Overlay parts must be registry-valid and key-unique across all parts.
- Validation fails if `frozen` mode is requested but no frozen registry exists.
- Maintenance-gated operations require an explicit maintenance context.
- Event write failures fail the mutation (auditability requirement).

## Tests
- Registry hash written to base and overlay metadata.
- Validate `current|frozen|either` modes.
- Write multiple overlay parts; ensure scan/materialize sees combined outputs.
- Compaction preserves data and archives parts.
- Event schema includes `event_version`, `actor`, `registry_hash`.
- `usr events tail --format json` emits valid JSON lines.
- Densegen adapter writes registry-valid schemas for complex types.

## Risks and Mitigations
- Registry evolution risk: frozen mode prevents historic breakage.
- Overlay growth: compaction provides maintenance path.
- Large nested types: enforce explicit registry typing and fail fast.
- HPC notify: keep networking out of compute nodes via external adapter.

## Rollout
- Implement steps in small commits with tests.
- Update USR docs and densegen docs to reflect registry hash, overlay parts, and event schema.
- Provide migration note for tools writing overlays to switch to `write_overlay_part` when write-heavy.
