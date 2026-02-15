## Workspace layout

This page explains what a healthy DenseGen workspace looks like and why.

DenseGen works best when each run is self-contained: one `config.yaml`, one `inputs/`, one `outputs/` tree.
Packaged demos use schema `2.9`, and `config.yaml` is the contract anchor for all subprocesses.

### Expected layout

```text
workspace/
  config.yaml
  inputs/
  notebooks/
  outputs/
    meta/
    logs/
    pools/
    libraries/
    tables/
    plots/
```

### Why this layout matters

- path resolution is predictable (`config.yaml` is the anchor)
- run state is easy to reset without touching inputs
- diagnostics stay with the run they came from
- Stage-A/Stage-B/solver artifacts stay co-located for one-run debugging

### Practical notes

- Config resolution policy: [../reference/cli.md#config-resolution](../reference/cli.md#config-resolution)
- Outputs live under `densegen.run.root` (typically `.`)
- If outputs already exist, `dense run` resumes by default; use `dense run --fresh` for a clean run
- `dense workspace init` defaults to `src/dnadesign/densegen/workspaces` (or `$DENSEGEN_WORKSPACE_ROOT`)

### Useful commands

```bash
# Create a workspace from a packaged demo template.
uv run dense workspace init --id my_run --from-workspace demo_binding_sites --copy-inputs --output-mode local

# Show the effective workspace roots DenseGen will use.
uv run dense workspace where --format json
```

---

@e-south
