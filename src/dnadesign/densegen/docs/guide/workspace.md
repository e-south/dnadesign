# Workspace layout (recommended)

DenseGen is most ergonomic when each run is **self‑contained** and uses config‑relative paths.
That keeps inputs, outputs, logs, and plots together and makes runs easy to archive or move.

## Suggested directory layout

```
densegen/
  runs/
    2026-01-14_sigma70_demo/
      config.yaml
      inputs/                # optional local copies
      outputs/
        parquet/             # part-*.parquet
        usr/                 # USR datasets (if used)
      plots/
      logs/
    _archive/
      legacy_run_name/       # older runs or artifacts kept out of the active list
    _campaigns/
      template/              # staged templates or demo campaigns
```

## Why this layout

- **Decoupled**: moving a run directory preserves everything needed to reproduce it.
- **No fallbacks**: all paths are explicit and resolve relative to `config.yaml`.
- **Scalable**: large runs don’t collide in shared output directories.
- **Predictable logs**: default logs land in `logs/<run_id>.log` within the run directory.

## Config snippet (run-scoped paths)

```yaml
densegen:
  run:
    id: 2026-01-14_sigma70_demo
    root: "."

output:
  targets: [parquet]
  schema:
    bio_type: dna
    alphabet: dna_4
  parquet:
    path: outputs/parquet
    deduplicate: true

logging:
  log_dir: logs

plots:
  out_dir: plots
```

When a run is complete, archive or sync the run directory as a unit.

Tip: use `dense stage --id <run_name>` to scaffold a new run directory.
Tip: use `dense ls-runs --root runs/_archive` to inspect archived runs.
