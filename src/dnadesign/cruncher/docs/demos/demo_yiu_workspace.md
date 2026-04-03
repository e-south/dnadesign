## YIU Workspace Demo

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-03

Use this walkthrough to run the checked-in YIU reference workspace, publish its payload bundle, and inspect the three payload views.

```bash
# Use the checked-in YIU demo workspace in the repo.
DEMO_WORKSPACE=src/dnadesign/cruncher/workspaces/demo_yiu_payload

# Use the checked-in payload-centric YIU specs.
USER_SPEC="$DEMO_WORKSPACE/configs/yiu/example_payload.yiu.yaml"
TETR_SPEC="$DEMO_WORKSPACE/configs/yiu/tetr_monotypic_hit.yiu.yaml"

# Confirm the checked-in workspace is discoverable.
uv run cruncher workspaces list --root src/dnadesign/cruncher/workspaces

# Run the checked-in machine runbook.
uv run cruncher workspaces run --workspace demo_yiu_payload --runbook configs/runbook.yaml

# Validate the user-sequence payload spec.
uv run cruncher yiu validate --spec "$USER_SPEC"

# Publish and render the user-sequence payload bundle.
uv run cruncher yiu render --spec "$USER_SPEC" --force-overwrite --emit-renders

# Inspect the published user-sequence payload bundle.
uv run cruncher yiu show --bundle "$DEMO_WORKSPACE/bundles/example_payload"

# Validate and render the checked-in sample-hit payload bundle.
uv run cruncher yiu validate --spec "$TETR_SPEC"
uv run cruncher yiu render --spec "$TETR_SPEC" --force-overwrite --emit-renders
uv run cruncher yiu show --bundle "$DEMO_WORKSPACE/bundles/tetr_monotypic_hit"
```

The published bundles live under:

```text
bundles/example_payload/
bundles/tetr_monotypic_hit/
```

Key YIU publication paths:

- `bundle_manifest.json`
- `normalized_payload.json`
- bundle-root render truth: `visual_inventory.json`
- published view contracts: `payload_view.json`, `split_payload_view.json`, `assembled_payload_view.json`
- one composite operator PDF: `payload_views.pdf`
- optional debug jobs: `baserender_jobs/*.job.yaml` when `emit_render_jobs_debug: true`

The sample-hit spec is self-contained. The workspace also includes a local PWM context sidecar under `motifs/example_pwm_context.yaml` as an advanced example.

If you want a disposable scratch copy instead of the checked-in repo workspace:

```bash
# Create a fresh YIU workspace with the same payload-centric schema.
uv run cruncher yiu init-workspace yiu_lab_demo
```

Next:

- [YIU Workflow](../guides/yiu_workflow.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
