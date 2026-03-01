## DenseGen workspaces

This directory contains packaged workspace templates and local run directories created with `dense workspace init`.

### Choose a packaged workspace
- [`demo_tfbs_baseline`](demo_tfbs_baseline/README.md): smallest local baseline without PWM mining.
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

### Directory policy
- `demo_*`: small baseline templates used for onboarding and checks.
- `study_*`: larger campaign templates.
- `archived/`: preserved historical local runs.
- Local workspaces created by `dense workspace init` are expected under this root unless you set `DENSEGEN_WORKSPACE_ROOT`.

### Expected packaged workspace shape
- `README.md`
- `config.yaml`
- `runbook.md`
- `runbook.sh`
- `inputs/`
- `outputs/` (generated at runtime)

### References
- Template behavior model: [workspace templates](../docs/concepts/workspace-templates.md)
- Workspace layout contract: [workspace model](../docs/concepts/workspace.md)
- Output artifact contract: [outputs reference](../docs/reference/outputs.md)

### Optional Stage-B Showcase Video
Enable this in a workspace `config.yaml` under `plots`:

```yaml
plots:
  video:
    enabled: true  # opt in to video rendering
    mode: all_plans_round_robin_single_video  # single round-robin MP4 across plans
```

Then run `dense plot` (or `dense plot --only dense_array_video_showcase`) to produce:
`outputs/plots/stage_b/all_plans/showcase.mp4`.
