DenseGen keeps packaged demos and local run workspaces here.

- `demo_*` directories are tracked templates with didactic inputs/config.
- `runs/` is the centralized local runtime root for generated workspaces.
- `_archived/` stores historical local artifacts that are not part of active runs.

Each run workspace should contain:

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

Keep real production runs out of version control. Runtime workspaces under `runs/` are ignored.

Tracked demos:

- `demo_binding_sites_vanilla/` is the unconstrained baseline demo for didactic run/solver behavior.
- `demo_meme_three_tfs/` is the canonical Cruncher PWM handoff demo with LexA/CpxR/BaeR artifacts.

The canonical PWM demo uses DenseGen motif artifacts stored in
`demo_meme_three_tfs/inputs/motif_artifacts/` (packaged for reproducibility).
Use `dense inspect run --root src/dnadesign/densegen/workspaces/runs` to inspect active local runs.
