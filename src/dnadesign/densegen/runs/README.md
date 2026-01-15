Run-scoped workbenches live here. Each run directory should contain:

- `config.yaml`
- `inputs/`
- `outputs/` (Parquet + optional USR datasets)
- `logs/`
- `plots/`

Keep real production runs out of version control; use this directory for demo artifacts only.

Archived or legacy artifacts live under `_archive/` so the active run list stays clean.
The canonical demo lives under `demo/`.
Use `dense ls-runs --root runs/_archive` if you want to inspect archived runs.
