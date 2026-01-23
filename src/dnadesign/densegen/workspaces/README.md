Run-scoped workbenches live here. Each workspace should contain:

- `config.yaml`
- `inputs/`
- `outputs/` (run artifacts)
  - `tables/` (dense_arrays + attempts/solutions/composition)
  - `plots/` (plot images)
  - `report/` (report.md/.json/.html + assets/)
  - `pools/` (Stage‑A TFBS pools + optional candidates/)
  - `libraries/` (Stage‑B library artifacts)
  - `logs/` (optional; defaults to outputs/logs)
  - `meta/` (run manifests + run state + id index)

Keep real production runs out of version control; use this directory for demo artifacts only.

Archived or legacy artifacts live under `_archive/` so the active workspace list stays clean.
The canonical demo lives under `demo_meme_two_tf/` and uses MEME motif files stored in the
workspace `inputs/` directory (packaged for reproducibility).
Use `dense inspect run --root workspaces/_archive` if you want to inspect archived workspaces.
Only `demo_meme_two_tf/` is tracked in git; any other workspace directories here are ignored
and intended for local experiments.
