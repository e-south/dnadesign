## DenseGen dense array showcase tutorial

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-14

Use this tutorial to run the local dense-array showcase. The workspace uses toy `binding_sites` input, CBC, parquet-only output, and fixed-length 100 bp sequences.

### Runbook command

Use the workspace runbook sequence from [demo_dense_array_showcase/runbook.md](../../workspaces/demo_dense_array_showcase/runbook.md).

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_dense_array_showcase
# Run the packaged flow in explicit fresh mode.
./runbook.sh --mode fresh
```

Use `--mode resume` to continue generation, or `--mode analysis` when you only need plots, notebook, and video refresh.

### What this demo shows

- `dense_free`: dense TFBS packing without fixed anchors.
- `single_anchor`: the same toy TFBS panel with one variant fixed-element pair.
- `dual_anchor`: the same panel with one variant fixed-element pair plus one static fixed-element pair.

The input table contains 40 short TFBS entries across eight generic regulators, `TF_A` through `TF_H`.

### Step-by-step commands

```bash
# Enter the demo workspace.
cd src/dnadesign/densegen/workspaces/demo_dense_array_showcase
# Keep the config path stable across later commands.
CONFIG="$PWD/config.yaml"
```

```bash
# Validate config schema and probe local CBC availability.
uv run dense validate-config --probe-solver -c "$CONFIG"
# Start a fresh run from a clean output state.
uv run dense run --fresh --no-plot -c "$CONFIG"
# Inspect accepted rows, events, and per-plan library progress.
uv run dense inspect run --events --library -c "$CONFIG"
```

```bash
# Render static plots and the Stage-B showcase video.
uv run dense plot -c "$CONFIG"
# Generate the run-overview marimo notebook artifact.
uv run dense notebook generate -c "$CONFIG"
# Run notebook validation before opening or sharing it.
uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"
```

### Expected outputs

- `outputs/tables/records.parquet`
- `outputs/plots/`
- `outputs/plots/stage_b/all_plans/showcase.mp4`
- `outputs/notebooks/densegen_run_overview.py`

### Related docs

- [Generation concept](../concepts/generation/model.md)
- [Outputs reference](../reference/outputs.md)
- [Workspaces directory](../../workspaces/README.md)
