## YIU Workspace Demo

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-27

Use this walkthrough to run the checked-in split-payload circularized YIU demo workspace, inspect the explicit and solve bundles, and render the emitted QA views.

```bash
# Use the checked-in YIU demo workspace in the repo.
DEMO_WORKSPACE=src/dnadesign/cruncher/workspaces/demo_yiu_circularized

# Confirm the checked-in workspace is family-discoverable from the workspace registry.
uv run cruncher workspaces list --root src/dnadesign/cruncher/workspaces
# Run the standard machine runbook from the checked-in workspace.
uv run cruncher workspaces run --workspace demo_yiu_circularized --runbook configs/runbook.yaml

# Validate the explicit YIU spec before materializing any artifacts.
uv run cruncher yiu validate \
  --spec "$DEMO_WORKSPACE/configs/yiu/example_split_payload_circularized.yiu.yaml"

# Materialize the explicit YIU bundle plus published views and jobs.
uv run cruncher yiu design \
  --spec "$DEMO_WORKSPACE/configs/yiu/example_split_payload_circularized.yiu.yaml" \
  --force-overwrite

# Re-materialize the same explicit bundle under trace intent.
uv run cruncher yiu trace \
  --spec "$DEMO_WORKSPACE/configs/yiu/example_split_payload_circularized.yiu.yaml" \
  --force-overwrite

# Run the paired solve spec and materialize the top hit bundles.
uv run cruncher yiu solve \
  --spec "$DEMO_WORKSPACE/configs/yiu/example_split_payload_circularized.yiu.solve.yaml" \
  --force-overwrite

# Validate the emitted render job for the ligated hairpin view.
uv run cruncher visuals validate \
  --job "$DEMO_WORKSPACE/outputs/yiu/explicit/example_split_payload_circularized/<design_id>/published/baserender_jobs/ligated_ssdna_hairpin.job.yaml"

# Render the ligated hairpin QA view from the emitted job file.
uv run cruncher visuals run \
  --job "$DEMO_WORKSPACE/outputs/yiu/explicit/example_split_payload_circularized/<design_id>/published/baserender_jobs/ligated_ssdna_hairpin.job.yaml"
```

The checked-in demo workspace ships input and runbook material only:

- `runbook.md`
- `configs/runbook.yaml`
- `configs/yiu/example_split_payload_circularized.yiu.yaml`
- `configs/yiu/example_split_payload_circularized.yiu.solve.yaml`
- `configs/yiu/compat/example_adapter_hairpin.yiu.yaml`
- `configs/yiu/compat/example_legacy_v1.yiu.yaml`
- `catalogs/enzymes.yaml`
- `catalogs/oligo_parts.yaml`
- `catalogs/backbones.yaml`

Runtime outputs are generated under `outputs/yiu/...` only after you run the explicit or solve commands above. The checked-in workspace does not version control explicit bundles, rendered outputs, matplotlib caches, or desktop clutter.

If you want a disposable scratch copy instead of the checked-in repo workspace, generate one with:

```bash
# Create a scratch YIU workspace outside the checked-in demo.
uv run cruncher yiu init-workspace yiu_lab_demo
```

After `design`, inspect the explicit bundle with:

```bash
# Inspect the explicit bundle and published visual surface.
uv run cruncher yiu show \
  --run "$DEMO_WORKSPACE/outputs/yiu/explicit/example_split_payload_circularized/<design_id>"
```

After `solve`, inspect the solve bundle with:

```bash
# Inspect the solve bundle, solve-level views, and top-hit path.
uv run cruncher yiu show \
  --run "$DEMO_WORKSPACE/outputs/yiu/solve/example_split_payload_circularized/<solve_id>"
```

One emitted BaseRender job from the explicit bundle looks like:

```text
published/baserender_jobs/ligated_ssdna_hairpin.job.yaml
```

Published view contracts are written under:

```text
published/views/
```

Running the two `cruncher visuals` commands above writes rendered QA output under:

```text
published/renders/
```

Next:

- [YIU Workflow](../guides/yiu_workflow.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
