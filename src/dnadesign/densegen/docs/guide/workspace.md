## Workspace layout

DenseGen is most ergonomic when each run is self-contained and uses config-relative paths.

### Contents
- [Suggested directory layout](#suggested-directory-layout) - a run-scoped structure.
- [Why this layout](#why-this-layout) - portability and reproducibility benefits.
- [Config snippet (run-scoped paths)](#config-snippet-run-scoped-paths) - minimal run wiring.

---

### Suggested directory layout

```
workspace/
  config.yaml
  inputs/
  outputs/
    meta/
    logs/
    pools/
    libraries/
    tables/
    plots/
    report/
    pools/candidates/  # optional Stage‑A debug artifacts
```

---

### Why this layout

- **Decoupled**: moving a workspace preserves everything needed to reproduce it.
- **Explicit config resolution**: CLI checks `./config.yaml`, then parent directories, then a single auto-detected
  workspace; it prints the chosen path and exits if ambiguous. Use `-c` to pin a workspace.
- **Scalable**: large runs do not collide in shared output directories.
- **Predictable logs**: default logs land in `outputs/logs/<run_id>.log` within the workspace.
- **Resume‑safe (explicit)**: if run outputs already exist (e.g., `outputs/tables/attempts.parquet` or
  `outputs/meta/run_state.json`), you must choose `dense run --resume` (continue in‑place) or
  `dense run --fresh` (clear outputs and start over). Stage‑A/Stage‑B artifacts in
  `outputs/pools` or `outputs/libraries` do not trigger this guard.
- **Candidate mining artifacts**: `outputs/pools/candidates/` is overwritten by
  `stage-a build-pool --fresh` or `dense run --rebuild-stage-a`; copy it elsewhere if you want
  to keep prior candidates. Use `dense run --fresh` to clear outputs when restarting a workspace.

---

### Config snippet (run-scoped paths)

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
      path: outputs/tables/dense_arrays.parquet
      deduplicate: true

  logging:
    log_dir: outputs/logs

plots:
  out_dir: outputs/plots
```

When a run is complete, archive or sync the workspace as a unit. If you rerun in the same
workspace and run outputs already exist, DenseGen requires an explicit choice: use
`dense run --resume` to continue from existing outputs or `dense run --fresh` to clear
`outputs/` and start over.

Tip: use `dense workspace init --id <run_name> --template-id <template>` to scaffold a new workspace.

---

@e-south
