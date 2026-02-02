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

Archived or legacy artifacts live under `_archived/` so the active workspace list stays clean.
The canonical demo lives under `demo_meme_three_tfs/` and uses DenseGen PWM artifacts stored in the
workspace `inputs/motif_artifacts/` directory (packaged for reproducibility).
Use `dense inspect run --root workspaces/_archive` if you want to inspect archived workspaces.
Only `demo_meme_three_tfs/` is tracked in git; any other workspace directories here are ignored
and intended for local experiments.
