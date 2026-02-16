## DenseGen Demo Flows

This page helps you pick the right starting demo.

DenseGen ships two packaged workspaces plus one full-stack integration walkthrough.
Packaged demo configs use DenseGen schema `2.9`.
For non-demo packaged templates (`study_*`), use:
- [../guide/workspace-templates.md](../guide/workspace-templates.md)

### Recommended order

1. Start with the binding-sites baseline demo (fastest path).
2. Move to the sampling baseline demo (canonical production path).
3. Finish with the DenseGen -> USR -> Notify stack demo.

### Sequential operator arc

Use this same intent-driven sequence in every demo:

1. `workspace init`: stage a reproducible workspace from a packaged template.
2. `validate-config` + `inspect ...`: confirm wiring before runtime spend.
3. `run` (or `stage-a build-pool` + `run`): generate arrays and manifests.
4. `inspect run --library --events`: verify outcomes and failure modes.
5. `plot` + `notebook generate`: materialize analysis artifacts.
6. `campaign-reset` + rerun: restart from clean outputs with the same inputs/config.

Notebook contract note:
- `dense notebook generate` works for both local parquet and USR-backed runs.
- source selection is explicit: single sink -> that sink, dual sink -> `plots.source`.

### Contents
- [1) Binding-sites baseline demo (start here)](#1-binding-sites-baseline-demo-start-here)
- [2) Sampling baseline demo (canonical workflow)](#2-sampling-baseline-demo-canonical-workflow)
- [3) DenseGen -> USR -> Notify demo (end-to-end stack)](#3-densegen-usr-notify-demo-end-to-end-stack)
- [Reset pattern for any demo](#reset-pattern-for-any-demo)

### 1) Binding-sites baseline demo (start here)

- Workspace template: `workspaces/demo_tfbs_baseline/`
- Purpose: compare default `baseline` behavior against `baseline_sigma70` using mock TFBS inputs
- Guide: [demo_tfbs_baseline.md](demo_tfbs_baseline.md)

### 2) Sampling baseline demo (canonical workflow)

- Workspace template: `workspaces/demo_sampling_baseline/`
- Purpose: run the Cruncher -> motif-artifact -> DenseGen flow and write to a local USR dataset
- Guides:
  - [demo_sampling_baseline.md](demo_sampling_baseline.md)
  - [../workflows/cruncher_pwm_pipeline.md](../workflows/cruncher_pwm_pipeline.md)

### 3) DenseGen -> USR -> Notify demo (end-to-end stack)

- Purpose: validate the integration boundary end-to-end
- Outcome: DenseGen writes to USR and Notify consumes USR `.events.log`
- Guide: [demo_usr_notify.md](demo_usr_notify.md)

### Reset pattern for any demo

Use this whenever you want a clean rerun from the same workspace.

```bash
# Set config path for the workspace you want to reset.
CONFIG="$PWD/config.yaml"

# Remove outputs/run state but keep config + inputs.
uv run dense campaign-reset -c "$CONFIG"

# Re-run generation from scratch.
uv run dense run --fresh --no-plot -c "$CONFIG"

# Re-render plots.
uv run dense plot -c "$CONFIG"

# Re-generate and launch the workspace notebook.
uv run dense notebook generate -c "$CONFIG"
uv run dense notebook run -c "$CONFIG"
uv run dense notebook run --headless -c "$CONFIG"
```

---

@e-south
