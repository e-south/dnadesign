## DenseGen Demo Flows

DenseGen includes two packaged workspaces plus one stack integration walkthrough.

### 1) Vanilla binding-sites demo (start here)

- Workspace: `workspaces/demo_binding_sites_vanilla/`
- Purpose: understand default `dense run` behavior with no added plan constraints.
- Guide: [demo_binding_sites.md](demo_binding_sites.md)

### 2) Three-TF PWM demo (canonical workflow)

- Workspace: `workspaces/demo_meme_three_tfs/`
- Purpose: run the Cruncher -> motif-artifact -> DenseGen pipeline used in main workflows, writing outputs into a workspace-local USR dataset.
- Guides:
  - [demo_pwm_artifacts.md](demo_pwm_artifacts.md)
  - [../workflows/cruncher_pwm_pipeline.md](../workflows/cruncher_pwm_pipeline.md)

Recommended order: run the vanilla demo first, then the three-TF PWM workflow, then the stack demo.

### 3) DenseGen -> USR -> Notify demo (end-to-end stack)

- Purpose: understand the integration boundary:
  DenseGen writes sequences plus derived overlay parts into USR, and Notify consumes USR `.events.log`.
- Guide: [demo_usr_notify.md](demo_usr_notify.md)

---

@e-south
