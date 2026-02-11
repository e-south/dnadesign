DenseGen stores tracked demo workspaces and local generated workspaces here.

- `demo_*` directories are representative workspaces with didactic configs/inputs.
- New workspaces created by `dense workspace init` default to this same root.
- `archived/` stores historical local artifacts that are not part of active work.

Each workspace should contain:

- `config.yaml`
- `inputs/`
- `outputs/` (run artifacts)
  - `tables/` (dense_arrays + attempts/solutions/composition)
  - `plots/` (plot images)
  - `report/` (report.md/.json/.html + assets/)
  - `pools/` (Stage-A TFBS pools + optional candidates/)
  - `libraries/` (Stage-B library artifacts)
  - `logs/` (optional; defaults to outputs/logs)
  - `meta/` (run manifests + run state + id index)

Keep real production outputs out of version control. Generated workspaces under
`workspaces/` are ignored by default unless explicitly whitelisted.

Tracked demos:

- `demo_binding_sites/` is the unconstrained baseline demo for didactic run/solver behavior.
- `demo_meme_three_tfs/` is the canonical Cruncher PWM handoff demo with LexA/CpxR/BaeR artifacts.

The canonical PWM demo uses DenseGen motif artifacts stored in
`demo_meme_three_tfs/inputs/motif_artifacts/` (packaged for reproducibility).
Use `dense inspect run --root src/dnadesign/densegen/workspaces` to inspect local workspaces.
