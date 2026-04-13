## study_stress_ethanol_cipro workspace

This workspace writes its USR sink directly to the shared USR root
`src/dnadesign/usr/datasets/`. Treat that shared root as the DenseGen producer
surface for this study and as the cross-tool handoff source for downstream
status.

Run from this directory:

```bash
# Start a clean generation pass (default mode if omitted).
./runbook.sh --mode fresh
# Continue generation without wiping prior outputs.
./runbook.sh --mode resume
# Rebuild plots/notebook from existing outputs only.
./runbook.sh --mode analysis
```

Read-only local analysis over the shared DenseGen source dataset uses the same
workspace through the public DenseGen CLI:

```bash
# Render the workspace's default analysis plots from the shared dataset inputs.
uv run dense plot -c "$PWD/config.yaml"
# Regenerate the read-only marimo notebook that browses the persisted artifacts.
uv run dense notebook generate -c "$PWD/config.yaml"
# Launch the generated notebook locally without recomputing upstream datasets.
uv run dense notebook run -c "$PWD/config.yaml"
```

Those commands read `densegen/study_stress_ethanol_cipro` through the existing
workspace config and only write local plot/notebook artifacts under `outputs/`.
The workspace resolves `output.usr.root` against the git common repo root, so
the same config works from a normal checkout and from an isolated worktree.

This workspace now defaults `dense plot` to the dataset-native
`dataset_source_inventory` and `dataset_metadata_heatmap` views plus the core
local diagnostics. The Stage-B showcase video remains explicit-only through
`uv run dense plot --only dense_array_video_showcase`.

- Runbook: [runbook.md](runbook.md)
- Config: [config.yaml](config.yaml)
- All workspaces: [../README.md](../README.md)
