# Runbook

This workspace has one visual-only Snapback step. It validates the explicit sequence decomposition and renders a
precursor-sites, post-release-fragments, and foldback triptych.

```bash
# Run the visual-only MSD-HOPV5 Snapback render from its scoped workspace.
uv run cruncher workspaces run --workspace src/dnadesign/cruncher/workspaces/msd-HOPV5_snapback
```

### Step-by-Step Commands

    cruncher snapback visual --spec configs/snapback/msd-HOPV5.visual.snapback.yaml --force-overwrite --json

The runbook writes only ignored generated artifacts under `outputs/msd-HOPV5_visual`.
