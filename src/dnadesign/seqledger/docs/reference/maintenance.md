<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../README.md)

# SeqLedger maintenance patterns

Use maintenance commands when you intend to change dataset storage or consolidate records. Routine reads and exports already include overlays; they do not require materialization. Run the examples from a DNADesign checkout with the intended root selected through `--root` or `DNADESIGN_SEQLEDGER_ROOT`. Dataset and namespace names below are examples.

## Materialize annotations into the base table

Materialization rewrites `records.parquet` with effective overlay values. Use it when a consumer needs those columns in the base file itself; use export when the consumer only needs a separate annotated table.

```bash
# Snapshot the base table before merging annotations into it.
uv run seqledger materialize example_sequences --snapshot-before
```

The command prompts before rewriting. Overlay removal and archival are separate options; inspect `seqledger materialize --help` before choosing them.

## GenBank-backed import and sequence-view upkeep

```bash
# Import GenBank-backed native records plus seq_annot overlays and native sequence views.
uv run seqledger genbank import --manifest path/to/genbank_import.yaml
```

Operational rules:

- `seqledger genbank import` is the owning write surface for `seq_annot` and native-record sequence views.
- Feature extraction during GenBank import may also write `derived` overlays and child sequence views when the manifest requests it.
- Do not hand-edit `_views/sequence_views.parquet`; rerun the owning import or realization step with the intended conflict policy.
- Dataset-local `_views/sequence_views.parquet` sidecars are additive metadata and are not compacted via overlay maintenance commands.

## Registry and overlay maintenance

```bash
# Freeze active registry into dataset artifact.
uv run seqledger maintenance registry-freeze example_sequences

# Compact overlay parts for one namespace.
uv run seqledger maintenance overlay-compact example_sequences --namespace demo

# Refresh a compact overlay's registry metadata after registry-only drift.
uv run seqledger maintenance overlay-refresh-metadata example_sequences --namespace demo

# Project one namespace from a source dataset onto a downstream dataset by join key.
uv run seqledger maintenance overlay-project \
  --src example_sequences \
  --dest derived_sequences \
  --namespace demo \
  --src-join id \
  --dest-join id \
  --allow-missing

# Remove or archive one derived namespace.
uv run seqledger maintenance overlay-remove example_sequences --namespace demo --mode archive
```

Compaction retention contract:

- Compaction rewrites active parts into one compact overlay file.
- Compaction fails fast before registry freeze or file rewrites when the namespace has no overlay parts or is already compact.
- Previous part snapshots are dropped by default.
- Overlay archive retention is bounded: `overlay-remove --mode archive` keeps only the latest archived snapshot.
- Reserved system namespaces such as `usr_state` are only mutated through dedicated command groups such as `uv run seqledger state ...`.
- `overlay-project` is the safe repair path when downstream handoff datasets must inherit authoritative overlay metadata after merge, construct, or infer without rewriting `records.parquet` or disturbing unrelated namespaces such as `infer`.

Registry-metadata refresh contract:

- Use `overlay-refresh-metadata` only for compact overlays at `_derived/<namespace>.parquet`.
- The command streams the existing overlay rows through the current registry metadata without changing base rows, overlay values, or join keys.
- It fails fast for part-directory overlays at `_derived/<namespace>/`; run `overlay-compact` there because compaction changes artifact layout.
- Before refreshing, confirm that namespace-level compatibility still holds:

```bash
uv run seqledger validate <dataset> --strict --registry-mode namespace-current
```

If namespace-current validation passes but default strict validation reports an `Overlay registry_hash mismatch`, the overlay data and namespace contract are compatible; the compact overlay is carrying a stale full-registry stamp and can be refreshed.

## Event-log gardening

SeqLedger `.events.log` files are append-only operational ledgers while a dataset is
active. Do not truncate or hand-edit them. When a long-lived dataset has moved
through a reviewed maintenance boundary, archive the full log and keep only an
operational tail with the offline gardening command:

```bash
# Dry-run; reports how many lines would be archived and retained.
uv run seqledger maintenance event-log-garden <dataset> --retain-last 1000

# Write only after Notify watchers are stopped and cursors will be reseeded.
uv run seqledger maintenance event-log-garden <dataset> \
  --retain-last 1000 \
  --write \
  --acknowledge-notify-cursor-reset \
  --reason "retaining the latest 1000 events for active use"
```

Event-log gardening contract:

- The full pre-garden log is copied to `.events.archive/events-<timestamp>-<sha>.jsonl`.
- The live `.events.log` keeps the most recent `--retain-last` records and appends an `event_log_garden` audit event.
- Writes require `--acknowledge-notify-cursor-reset`; this is intentional because byte-offset Notify cursors become invalid after log rewriting.
- Run this as an offline maintenance operation only. Stop active Notify watchers first, reseed cursors to the new live log tail, and coordinate SeqLedger sync/publish so local and remote event sidecars do not drift.

## De-duplication

```bash
# Preview the selected deduplication rule before applying it.
uv run seqledger maintenance dedupe example_sequences \
  --key sequence_ci --keep keep-first --dry-run

# Apply the same rule (dna_4 only).
uv run seqledger maintenance dedupe example_sequences \
  --key sequence_ci --keep keep-first
```

`sequence_ci` is valid only for `dna_4` datasets.

## Merge datasets

```bash
uv run seqledger maintenance merge \
  --dest candidate_library \
  --src additional_candidates \
  --union-columns \
  --if-duplicate skip
```

Merge controls:

- `--require-same-columns` or `--union-columns`
- `--if-duplicate {error|skip|prefer-src|prefer-dest}`
- `--coerce-overlap {to-dest|none}`
- `--carry-namespace <namespace>` to explicitly carry one compact, `id`-keyed overlay namespace from `src` onto rows that actually survive the merge
- `--no-avoid-casefold-dups` to disable default case-fold duplicate avoidance
- plain `merge` still rewrites canonical `records.parquet` only; it does not implicitly copy source overlay namespaces or `_derived` sidecars
- `--carry-namespace` is fail-fast and narrow by design:
  - the source namespace must exist
  - source and destination overlays must be compact files, not overlay-parts directories
  - only `id`-keyed overlays are supported
  - only rows that survive the merge are carried
- if a needed namespace is not `id`-keyed or still lives in overlay parts, compact or reattach it explicitly before the merge
- plain merge does not copy `_views/sequence_views.parquet`; downstream sequence views must be regenerated or explicitly rewritten by the owning tool because view ids, bounds, and lineage may change across derived products

## Snapshots and export

```bash
# Write timestamped snapshot under _snapshots/.
uv run seqledger snapshot example_sequences

# Export the combined base-plus-overlay view to explicit files.
uv run seqledger export example_sequences --fmt parquet --out annotated_sequences.parquet
uv run seqledger export example_sequences --fmt csv --out annotated_sequences.csv
```

## Next steps

- End-to-end command chains: [SeqLedger workflow map](../operations/routes/workflow-map.md)
- Quickstart path: [Create your first dataset](../getting-started/cli-quickstart.md)
