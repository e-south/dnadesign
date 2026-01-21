Run-scoped workbenches live here. Each workspace should contain:

- `config.yaml`
- `inputs/`
- `outputs/` (Parquet data, run reports, plots, library artifacts)
  - `logs/` (optional; defaults to outputs/logs)
  - `meta/` (run manifests + run state)

Keep real production runs out of version control; use this directory for demo artifacts only.

Archived or legacy artifacts live under `_archive/` so the active workspace list stays clean.
The canonical demo lives under `demo_meme_two_tf/` and uses MEME motif files copied from
the basic Cruncher demo workspace (`inputs/local_motifs`). DenseGen reads these with the
shared Cruncher MEME parser to keep parsing DRY and consistent.
Use `dense inspect run --root workspaces/_archive` if you want to inspect archived workspaces.
Only `demo_meme_two_tf/` is tracked in git; any other workspace directories here are ignored
and intended for local experiments.
