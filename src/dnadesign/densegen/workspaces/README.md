DenseGen stores tracked packaged workspaces and local generated workspaces here.

User-facing template guidance lives in:
`src/dnadesign/densegen/docs/concepts/workspace-templates.md`

- `demo_*` directories are fast baseline workspaces for guided onboarding and CI pressure tests.
- `study_*` directories are larger template workspaces for real run planning.
- New workspaces created by `dense workspace init` default to this same root.
- `dense workspace init` requires exactly one source option: `--from-workspace` or `--from-config`.
- `archived/` stores historical local artifacts that are not part of active work.

Each workspace should contain:

- `config.yaml`
- `inputs/`
- `outputs/` (run artifacts)
  - `tables/` (`records.parquet` + attempts/solutions/composition/run metrics)
  - `plots/` (plot images)
  - `pools/` (Stage-A TFBS pools + optional candidates/)
  - `libraries/` (Stage-B library artifacts)
  - `logs/` (optional; defaults to outputs/logs)
  - `meta/` (run manifests + run state + id index)

Keep real production outputs out of version control. Generated workspaces under
`workspaces/` are ignored by default unless explicitly whitelisted.

Packaged workspace naming convention:

- `demo_<intent>`: didactic baseline with low runtime cost.
- `study_<intent>`: extended template for non-demo workloads.

Packaged workspaces:

- `demo_tfbs_baseline/` is the unconstrained baseline demo for didactic run/solver behavior.
- `demo_sampling_baseline/` is the canonical Cruncher PWM handoff demo with LexA/CpxR/BaeR artifacts.
- `study_constitutive_sigma_panel/` templates constitutive promoter generation via motif-set and plan-template matrix expansion (six baseline -35/-10 pairs).
- `study_stress_ethanol_cipro/` templates larger stress-response generation for ethanol/ciprofloxacin with high-budget Stage-A sampling and global final-sequence motif constraints.

The sampling-baseline workspace stores DenseGen motif artifacts in
`demo_sampling_baseline/inputs/motif_artifacts/` and the stress-study workspace carries the
same curated artifact set under its own `inputs/motif_artifacts/` directory.
Use `dense inspect run --root src/dnadesign/densegen/workspaces` to inspect local workspaces.
