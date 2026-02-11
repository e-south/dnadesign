## DenseGen Demo Flows

This page helps you pick the right starting demo.

DenseGen ships two packaged workspaces plus one full-stack integration walkthrough.
Packaged demo configs use DenseGen schema `2.9`.

### Recommended order

1. Start with the binding-sites baseline demo (fastest path).
2. Move to the three-TF PWM demo (canonical production path).
3. Finish with the DenseGen -> USR -> Notify stack demo.

### Contents
- [1) Binding-sites baseline demo (start here)](#1-binding-sites-baseline-demo-start-here)
- [2) Three-TF PWM demo (canonical workflow)](#2-three-tf-pwm-demo-canonical-workflow)
- [3) DenseGen -> USR -> Notify demo (end-to-end stack)](#3-densegen-usr-notify-demo-end-to-end-stack)
- [Reset pattern for any demo](#reset-pattern-for-any-demo)

### 1) Binding-sites baseline demo (start here)

- Workspace template: `workspaces/demo_binding_sites/`
- Purpose: compare default `baseline` behavior against `baseline_sigma70` using mock TFBS inputs
- Guide: [demo_binding_sites.md](demo_binding_sites.md)

### 2) Three-TF PWM demo (canonical workflow)

- Workspace template: `workspaces/demo_meme_three_tfs/`
- Purpose: run the Cruncher -> motif-artifact -> DenseGen flow and write to a local USR dataset
- Guides:
  - [demo_pwm_artifacts.md](demo_pwm_artifacts.md)
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
```

---

@e-south
