## Workspace layout

DenseGen is most ergonomic when each run is self-contained and uses config-relative paths. That
keeps inputs, outputs, logs, and plots together and makes runs easy to archive or move.

### Contents
- [Suggested directory layout](#suggested-directory-layout) - a run-scoped structure.
- [Why this layout](#why-this-layout) - portability and reproducibility benefits.
- [Config snippet (run-scoped paths)](#config-snippet-run-scoped-paths) - minimal run wiring.

---

### Suggested directory layout

```
densegen/
  runs/
    demo/                   # canonical demo config + inputs
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
```

---

### Why this layout

- **Decoupled**: moving a run directory preserves everything needed to reproduce it.
- **No fallbacks**: all paths are explicit and resolve relative to `config.yaml`.
- **Scalable**: large runs do not collide in shared output directories.
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

Tip: use `dense stage --id <run_name>` to scaffold a new run directory. Use
`dense ls-runs --root runs/_archive` to inspect archived runs.

@e-south
