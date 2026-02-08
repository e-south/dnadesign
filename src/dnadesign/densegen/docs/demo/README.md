## DenseGen Demo Flows

DenseGen includes two packaged workspaces plus one stack integration walkthrough.

### 1) Binding-sites baseline demo (start here)

- Workspace: `workspaces/demo_binding_sites/`
- Purpose: compare default `baseline` behavior against `baseline_sigma70` (fixed promoter spacer 16-18 bp) using mock TFBS inputs in the 16–20 bp range.
- Guide: [demo_binding_sites.md](demo_binding_sites.md)

### 2) Three-TF PWM demo (canonical workflow)

- Workspace: `workspaces/demo_meme_three_tfs/`
- Purpose: run the Cruncher -> motif-artifact -> DenseGen pipeline used in main workflows, writing outputs into a workspace-local USR dataset.
- Guides:
  - [demo_pwm_artifacts.md](demo_pwm_artifacts.md)
  - [../workflows/cruncher_pwm_pipeline.md](../workflows/cruncher_pwm_pipeline.md)

Recommended order: run the binding-sites baseline demo first, then the three-TF PWM workflow, then the stack demo.

Reset + rerun pattern for either demo:

```bash
dense campaign-reset -c "$CONFIG"
dense run --fresh --no-plot -c "$CONFIG"
dense plot -c "$CONFIG"
```

### 3) DenseGen -> USR -> Notify demo (end-to-end stack)

- Purpose: understand the integration boundary:
  DenseGen writes sequences plus derived overlay parts into USR, and Notify consumes USR `.events.log`.
- Guide: [demo_usr_notify.md](demo_usr_notify.md)

---

@e-south
