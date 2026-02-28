# USR maintenance patterns

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This page captures common maintenance commands that mutate or package dataset state.

## Registry and overlay maintenance

```bash
# Freeze active registry into dataset artifact.
uv run usr maintenance registry-freeze densegen/demo

# Compact overlay parts for one namespace.
uv run usr maintenance overlay-compact densegen/demo --namespace densegen
```

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
- `--no-avoid-casefold-dups` to disable default case-fold duplicate avoidance

## Snapshots and export

```bash
# Write timestamped snapshot under _snapshots/.
uv run usr snapshot densegen/demo

# Export canonical data.
uv run usr export densegen/demo --fmt parquet --out /tmp/usr_exports
uv run usr export densegen/demo --fmt csv --out /tmp/usr_exports
```

## Next steps

- End-to-end command chains: [../operations/workflow-map.md](../operations/workflow-map.md)
- Quickstart path: [../getting-started/cli-quickstart.md](../getting-started/cli-quickstart.md)
