## Workspace templates

This page is the catalog for packaged DenseGen workspace templates.

Use this page to choose a starting template and run it end-to-end with minimal
command repetition. Demo templates keep full walkthrough prose in `docs/demo/`.
Study templates stay concise here.

### Template classes

- `demo_*`: didactic onboarding templates with full walkthrough docs.
- `study_*`: non-demo templates for real study execution; no full walkthrough docs by default.

### Packaged templates

| Template id | Class | Use case | Extra requirements |
|---|---|---|---|
| `demo_tfbs_baseline` | demo | Fast baseline run with mock TFBS + sigma70 variant | Solver only (`CBC`/`GUROBI`) |
| `demo_sampling_baseline` | demo | Canonical motif-artifact sampling baseline (LexA/CpxR/BaeR) | Solver + MEME Suite (`fimo`) |
| `study_constitutive_sigma_panel` | study | Constitutive promoter panel across six explicit -35/-10 hexamer pairs | Solver only (`CBC`/`GUROBI`) |
| `study_stress_ethanol_cipro` | study | Larger stress-response generation with legacy demo-3TF Stage-A sampling profile | Solver + MEME Suite (`fimo`) |

### Demo walkthroughs

- `demo_tfbs_baseline`: [../demo/demo_tfbs_baseline.md](../demo/demo_tfbs_baseline.md)
- `demo_sampling_baseline`: [../demo/demo_sampling_baseline.md](../demo/demo_sampling_baseline.md)
- Full stack (`DenseGen -> USR -> Notify`): [../demo/demo_usr_notify.md](../demo/demo_usr_notify.md)

### Study template quickstart (single runbook pattern)

```bash
# Choose one packaged study template.
TEMPLATE_ID="study_constitutive_sigma_panel"   # or study_stress_ethanol_cipro
RUN_ID="study_trial"

# Create a workspace from the template.
uv run dense workspace init --id "$RUN_ID" --from-workspace "$TEMPLATE_ID" --copy-inputs --output-mode both
cd "src/dnadesign/densegen/workspaces/$RUN_ID"
CONFIG="$PWD/config.yaml"

# Validate config + solver.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Stage-A build is required for pwm_artifact inputs.
uv run dense stage-a build-pool --fresh -c "$CONFIG"

# Run, inspect, and render outputs.
uv run dense run --no-plot -c "$CONFIG"
uv run dense inspect run --library --events -c "$CONFIG"
uv run dense plot -c "$CONFIG"
uv run dense notebook generate -c "$CONFIG"
```

`study_constitutive_sigma_panel` enforces strict final-sequence motif exclusions. If
`dense run` stops with `Exceeded max_seconds_per_plan`, inspect run state and resume:

```bash
uv run dense inspect run --events --library -c "$CONFIG"
uv run dense run --resume --no-plot -c "$CONFIG"
```

Then tune `densegen.runtime.max_seconds_per_plan` or lower plan quotas if needed.
`dense plot` reads the sink selected by `plots.source`; if dual sinks are enabled,
ensure that sink has records before plotting.

If your template uses `pwm_artifact` inputs and `fimo` is not on `PATH`, use
`pixi run dense ...`.

Notebook compatibility:
- local/parquet-only workspaces: notebook reads `output.parquet.path`
- usr-only workspaces: notebook reads `<output.usr.root>/<output.usr.dataset>/records.parquet`
- dual-sink workspaces: notebook reads the sink declared by `plots.source`

### Sensitive/non-tracked study notes

Keep study-specific notes in generated workspace directories (ignored by default)
or in private operator docs outside git. Keep this catalog limited to:

- template id
- intent
- requirements
- command pattern

### When adding a new packaged workspace

Use this rule to keep docs decoupled and minimal:

1. Add template config under `workspaces/<template_id>/`.
2. Add one row to this catalog.
3. If it is a demo template (`demo_*`), add a dedicated walkthrough in `docs/demo/`.
4. If it is a study template (`study_*`), do not add a walkthrough unless there is
   new operator behavior not covered by this page.
