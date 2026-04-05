## YIU Workspace Demo

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-03
**Last updated by:** cruncher-maintainers on 2026-04-05

Use this walkthrough to run the checked-in YIU reference workspace, publish its payload bundle, and inspect the three payload views. Keep this page runnable-first; use [YIU Artifacts](../reference/yiu_artifacts.md) for the emitted-file contract and [YIU Visual System](../reference/yiu_visual_system.md) for the named visual directions.

<!-- docs:toc:off -->

### Route map

- Use this workspace when you want the checked-in user-sequence demo.
- Use a monotypic workspace when you need a sample-hit-backed YIU example.
- Use `init-workspace` when you want a disposable scratch workspace with the same schema and a starter `center_locked` or `optimize` junction policy.

```bash
# Use the checked-in YIU demo workspace in the repo.
DEMO_WORKSPACE=src/dnadesign/cruncher/workspaces/demo_yiu_payload

# Use the checked-in payload-centric YIU spec.
USER_SPEC="$DEMO_WORKSPACE/configs/yiu/example_payload.yiu.yaml"

# Confirm the checked-in workspace is discoverable.
uv run cruncher workspaces list --root src/dnadesign/cruncher/workspaces

# Run the checked-in machine runbook.
uv run cruncher workspaces run --workspace demo_yiu_payload --runbook configs/runbook.yaml

# Validate the user-sequence payload spec.
uv run cruncher yiu validate --spec "$USER_SPEC"

# Publish and render the user-sequence payload bundle.
uv run cruncher yiu render --spec "$USER_SPEC" --force-overwrite --emit-renders

# Inspect the published user-sequence payload bundle.
uv run cruncher yiu show --bundle "$DEMO_WORKSPACE/outputs/example_payload"
```

The published bundle and mirrored operator PDF live under:

```text
outputs/example_payload/
outputs/example_payload__payload_views.pdf
```

Use [YIU Artifacts](../reference/yiu_artifacts.md) for the exact emitted files, render-status semantics, and shared `render`/`show` inspection surface.

The checked-in workspace is intentionally user-sequence-only. Sample-hit YIU demos now live beside their source Sample outputs in the monotypic workspaces such as `demo_monotypic_tetr`, where the whole YIU bundle is consolidated under `outputs/plots/yiu__tetr_monotypic_hit/`.
The workspace still includes a generic local PWM context sidecar under `motifs/example_pwm_context.yaml` for extra experimentation.

If you want a disposable scratch copy instead of the checked-in repo workspace:

```bash
# Create a fresh YIU workspace with the same payload-centric schema.
uv run cruncher yiu init-workspace yiu_lab_demo

# Or seed a scratch workspace with your own payload and a center-locked starter junction.
uv run cruncher yiu init-workspace yiu_lab_center --sequence AACCGGTTGGTT --junction-mode center_locked
```

Next:

- [YIU Workflow](../guides/yiu_workflow.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
- [YIU Visual System](../reference/yiu_visual_system.md)
