## YIU Workspace Demo

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-29

Use this walkthrough to run the checked-in YIU reference workspace, inspect the explicit and solve bundles, and rerender their published visuals.

The checked-in solve fixture is small but nontrivial. On 2026-03-29 the default run was exhaustive, found 2 satisfying solutions, and selected 1 deterministic solution for publication.

```bash
# Use the checked-in YIU demo workspace in the repo.
DEMO_WORKSPACE=src/dnadesign/cruncher/workspaces/demo_yiu_circularized

# Resolve the checked-in explicit spec without hard-coding the filename.
EXPLICIT_SPEC="$(find "$DEMO_WORKSPACE/configs/yiu" -maxdepth 1 -name '*.yiu.yaml' ! -name '*.solve.yaml' | head -n 1)"

# Resolve the checked-in solve spec without hard-coding the filename.
SOLVE_SPEC="$(find "$DEMO_WORKSPACE/configs/yiu" -maxdepth 1 -name '*.yiu.solve.yaml' | head -n 1)"

# Confirm the checked-in workspace is discoverable.
uv run cruncher workspaces list --root src/dnadesign/cruncher/workspaces

# Run the checked-in machine runbook.
uv run cruncher workspaces run --workspace demo_yiu_circularized --runbook configs/runbook.yaml

# Validate the explicit spec.
uv run cruncher yiu validate --spec "$EXPLICIT_SPEC"

# Materialize and render the explicit bundle.
uv run cruncher yiu trace --spec "$EXPLICIT_SPEC" --force-overwrite --emit-renders

# Run the paired solve spec and render the selected solution bundle.
uv run cruncher yiu solve --spec "$SOLVE_SPEC" --force-overwrite --emit-renders

# Derive the workflow directory name from the explicit spec path.
WORKFLOW_NAME="$(basename "${EXPLICIT_SPEC%.yiu.yaml}")"

# Resolve the newest explicit run id for that workflow.
TRACE_ID="$(ls -1 "$DEMO_WORKSPACE/outputs/yiu/explicit/$WORKFLOW_NAME" | tail -n 1)"

# Resolve the newest solve run id for that workflow.
SOLVE_ID="$(ls -1 "$DEMO_WORKSPACE/outputs/yiu/solve/$WORKFLOW_NAME" | tail -n 1)"

# Show the explicit bundle summary.
uv run cruncher yiu show --run "$DEMO_WORKSPACE/outputs/yiu/explicit/$WORKFLOW_NAME/$TRACE_ID"

# Show the solve bundle summary.
uv run cruncher yiu show --run "$DEMO_WORKSPACE/outputs/yiu/solve/$WORKFLOW_NAME/$SOLVE_ID"

# Rerender the explicit bundle from its persisted view contracts.
uv run cruncher yiu render --run "$DEMO_WORKSPACE/outputs/yiu/explicit/$WORKFLOW_NAME/$TRACE_ID"

# Rerender the solve bundle from its persisted view contracts.
uv run cruncher yiu render --run "$DEMO_WORKSPACE/outputs/yiu/solve/$WORKFLOW_NAME/$SOLVE_ID"
```

After `trace`, the explicit bundle lives under:

```text
outputs/yiu/explicit/<workflow>/<trace_id>/
```

After `solve`, the solve bundle lives under:

```text
outputs/yiu/solve/<workflow>/<solve_id>/
```

Key YIU publication paths:

- bundle-root render truth: `visual_inventory.json`
- persisted view contracts: `contracts/visuals/*.json`
- explicit rendered PDFs: `visuals/*.pdf`
- solve rendered PDFs: `solution/visuals/*.pdf`

If you want a disposable scratch copy instead of the checked-in repo workspace:

```bash
# Create a fresh YIU workspace with the same reference inputs and catalogs.
uv run cruncher yiu init-workspace yiu_lab_demo
```

Next:

- [YIU Workflow](../guides/yiu_workflow.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
