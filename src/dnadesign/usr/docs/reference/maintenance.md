# USR maintenance patterns

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-26


This page captures common maintenance commands that mutate or package dataset state.

## GenBank-backed import and sequence-view upkeep

```bash
# Import GenBank-backed native records plus seq_annot overlays and native sequence views.
uv run usr genbank import --manifest path/to/genbank_import.yaml
```

Operational rules:

- `usr genbank import` is the owning write surface for `seq_annot` and native-record sequence views.
- Feature extraction during GenBank import may also write `derived` overlays and child sequence views when the manifest requests it.
- Do not hand-edit `_views/sequence_views.parquet`; rerun the owning import or realization step with the intended conflict policy.
- Dataset-local `_views/sequence_views.parquet` sidecars are additive metadata and are not compacted via overlay maintenance commands.

## Registry and overlay maintenance

```bash
# Freeze active registry into dataset artifact.
uv run usr maintenance registry-freeze densegen_demo

# Compact overlay parts for one namespace.
uv run usr maintenance overlay-compact densegen_demo --namespace densegen

# Project one namespace from a source dataset onto a downstream dataset by join key.
uv run usr maintenance overlay-project \
  --src densegen_demo \
  --dest promoter/demo_anchor_set \
  --namespace densegen \
  --src-join id \
  --dest-join id \
  --allow-missing

# Remove or archive one derived namespace.
uv run usr maintenance overlay-remove densegen_demo --namespace densegen --mode archive
```

Compaction retention contract:

- Compaction rewrites active parts into one compact overlay file.
- Compaction fails fast before registry freeze or file rewrites when the namespace has no overlay parts or is already compact.
- Previous part snapshots are dropped by default (no lingering compact archives).
- Overlay archive retention is bounded: `overlay-remove --mode archive` keeps only the latest archived snapshot.
- Reserved system namespaces such as `usr_state` are only mutated through dedicated command groups such as `uv run usr state ...`.
- `overlay-project` is the safe repair path when downstream handoff datasets must inherit authoritative overlay metadata after merge, construct, or infer without rewriting `records.parquet` or disturbing unrelated namespaces such as `infer`.

## De-duplication

```bash
# Case-insensitive dedupe (dna_4 only).
uv run usr maintenance dedupe <dataset> --key sequence_ci --keep keep-first

# Case-preserving dedupe.
uv run usr maintenance dedupe <dataset> --key sequence --keep keep-last

# Preview dedupe impact.
uv run usr maintenance dedupe <dataset> --dry-run
```

`sequence_ci` is valid only for `dna_4` datasets.

## Merge datasets

```bash
uv run usr maintenance merge \
  --dest 60bp_dual_promoter_cpxR_LexA \
  --src 60bp_dual_promoter_cpxR_LexA_v2 \
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
uv run usr snapshot densegen_demo

# Export canonical data.
uv run usr export densegen_demo --fmt parquet --out /tmp/usr_exports
uv run usr export densegen_demo --fmt csv --out /tmp/usr_exports
```

## Next steps

- End-to-end command chains: [../operations/workflow-map.md](../operations/workflow-map.md)
- Quickstart path: [../getting-started/cli-quickstart.md](../getting-started/cli-quickstart.md)
