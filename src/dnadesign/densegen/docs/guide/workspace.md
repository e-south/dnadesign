## Workspace layout

DenseGen is most ergonomic when each run is self-contained and uses config-relative paths. That
keeps inputs, outputs, and logs together and makes runs easy to archive or move.

### Contents
- [Suggested directory layout](#suggested-directory-layout) - a run-scoped structure.
- [Why this layout](#why-this-layout) - portability and reproducibility benefits.
- [Config snippet (run-scoped paths)](#config-snippet-run-scoped-paths) - minimal run wiring.

---

### Suggested directory layout

```
densegen/
  workspaces/
    demo_meme_two_tf/        # canonical demo config + inputs
    2026-01-14_sigma70_demo/
      config.yaml
      inputs/                # optional local copies
      outputs/               # data parquet, reports, plots, library artifacts
        logs/
        meta/
    _archive/
      legacy_run_name/       # older workspaces or artifacts kept out of the active list
```

---

### Why this layout

- **Decoupled**: moving a workspace preserves everything needed to reproduce it.
- **No fallbacks**: all paths are explicit and resolve relative to `config.yaml`.
- **Scalable**: large runs do not collide in shared output directories.
- **Predictable logs**: default logs land in `outputs/logs/<run_id>.log` within the workspace.
- **Resume‑safe**: if `outputs/dense_arrays.parquet` already exists, DenseGen resumes from existing sequences
  (only when the config hash and run_id match), so interrupted runs can continue without manual cleanup.

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
    path: outputs/dense_arrays.parquet
    deduplicate: true

logging:
  log_dir: outputs/logs

plots:
  out_dir: outputs
```

When a run is complete, archive or sync the workspace as a unit.

Tip: use `dense workspace init --id <run_name>` to scaffold a new workspace. Use
`dense inspect run --root workspaces/_archive` to inspect archived workspaces.
If your config references local motif files, add `--copy-inputs` so the workspace
remains self-contained (or update paths in `config.yaml` after staging).

@e-south
