## YIU Workspace Demo

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-26

Use this walkthrough to scaffold a runbook-only YIU workspace, validate the shipped spec, materialize the explicit state bundle, and inspect the published contracts.

```bash
# Use the checked-in Cruncher workspaces root.
WORKSPACES_ROOT=src/dnadesign/cruncher/workspaces
# Pick one YIU workspace name under that root.
DEMO_WORKSPACE=yiu_lab_demo

# Scaffold the runbook-only YIU workspace.
uv run cruncher yiu init-workspace "$DEMO_WORKSPACE"
# Confirm the new workspace is discoverable.
uv run cruncher workspaces list --root "$WORKSPACES_ROOT"
# Validate the shipped example spec before materializing outputs.
uv run cruncher yiu validate --spec "$WORKSPACES_ROOT/$DEMO_WORKSPACE/configs/yiu/example.yiu.yaml"
# Write the deterministic explicit YIU bundle.
uv run cruncher yiu design --spec "$WORKSPACES_ROOT/$DEMO_WORKSPACE/configs/yiu/example.yiu.yaml"
# Re-materialize the state graph for QA-oriented inspection.
uv run cruncher yiu trace --spec "$WORKSPACES_ROOT/$DEMO_WORKSPACE/configs/yiu/example.yiu.yaml" --force-overwrite
```

The scaffold keeps the YIU lane beside other Cruncher families:

- `configs/runbook.yaml` makes the workspace discoverable without adding `configs/config.yaml`
- `configs/yiu/example.yiu.yaml` is the default explicit spec
- `catalogs/*.yaml` holds optional protocol catalogs; the scaffolded example spec references all three
- `outputs/yiu/explicit/<spec.name>/<design_id>/` holds deterministic bundles
- `published/views/` contains per-state neutral JSON contracts

After `design`, inspect the bundle with:

```bash
# Inspect the explicit YIU bundle after design finishes.
uv run cruncher yiu show --run "$WORKSPACES_ROOT/$DEMO_WORKSPACE/outputs/yiu/explicit/example_yiu/<design_id>"
```

Key files inside the run directory:

- `yiu_report.json`
- `yiu_status.json`
- `yiu_manifest.json`
- `yiu_trace.jsonl`
- `yiu_parts.csv`
- `yiu_annotations.csv`
- `yiu_fragments.csv`
- `published/views/source_oligo_ssdna.json`
- `published/views/downstream_amplifiable_product.json`

Next:

- [YIU Workflow](../guides/yiu_workflow.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
