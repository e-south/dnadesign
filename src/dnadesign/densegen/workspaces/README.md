## DenseGen workspaces

This directory contains packaged workspace templates and local run directories created with `dense workspace init`.

List the packaged workspaces and their current output state with `uv run dense workspace list`.

### Choose a packaged workspace
- [`demo_tfbs_baseline`](demo_tfbs_baseline/README.md): smallest local baseline without PWM mining.
- [`demo_dense_array_showcase`](demo_dense_array_showcase/README.md): local CBC showcase for dense TFBS packing with no, one, and two fixed-anchor regimes.
- [`demo_sampling_baseline`](demo_sampling_baseline/README.md): PWM sampling baseline with ethanol/ciprofloxacin plans.
- [`study_constitutive_sigma_panel`](study_constitutive_sigma_panel/README.md): constitutive σ70 panel study.
- [`study_stress_ethanol_cipro`](study_stress_ethanol_cipro/README.md): stress-condition study with GUROBI defaults.

### Run it
From inside a packaged workspace directory:

```bash
# Run a clean generation pass.
./runbook.sh --mode fresh
# Rebuild plots/notebook from existing outputs only.
./runbook.sh --mode analysis
```

Runbook mode is explicit: `fresh|resume|analysis`. Wrappers default to `fresh` and also read `DENSEGEN_RUNBOOK_MODE` for non-interactive runs.

### Choose execution surface
- Use `runbook.sh` when you want the test-backed default sequence from `runbook.md`.
- Use direct `dense` CLI commands when you need partial flows (`run`, `inspect`, `plot`, `notebook`) or custom resume/extend behavior.
- Use `dense workspace init --output-mode local|usr|both` when you need a separate run root with explicit output placement.
- Use `dense workspace list --format json` when you need a machine-readable inventory with output-file counts and latest output timestamps.

### Directory policy
- `demo_*`: small baseline templates used for onboarding and checks.
- `study_*`: larger campaign templates.
- `archived/`: preserved historical local runs.
- Local workspaces created by `dense workspace init` are expected under this root unless you set `DENSEGEN_WORKSPACE_ROOT`.

### USR root semantics

- DenseGen `usr` workspaces should write to an explicit shared USR root in the
  config, typically `src/dnadesign/usr/datasets/` for repo-local study work.
- Use a workspace-local export root only when a study record or runbook says
  the dataset is intentionally not yet the shared cross-tool copy.
- The shared study copy is the root used by downstream status, infer, cluster,
  or OPAL routes.

### Expected packaged workspace shape
- `README.md`
- `config.yaml`
- `runbook.md`
- `runbook.sh`
- `inputs/`
- `outputs/` (generated at runtime)

### References
- Template behavior model: [workspace templates](../docs/concepts/workspace/templates.md)
- Workspace layout contract: [workspace model](../docs/concepts/workspace/layout.md)
- Output artifact contract: [outputs reference](../docs/reference/outputs.md)

### Stage-B Showcase Video
When `dense_array_showcase_video` is included in `plots.default` or passed through
`dense plot --only`, DenseGen writes:
`outputs/plots/stage_b/all_plans/showcase.mp4`.

Enable video rendering in a workspace `config.yaml` under `plots`:

```yaml
plots:
  video:
    enabled: true  # opt in to video rendering
    mode: all_plans_round_robin_single_video  # single round-robin MP4 across plans
```

Then run `dense plot` (or `dense plot --only dense_array_showcase_video`).
