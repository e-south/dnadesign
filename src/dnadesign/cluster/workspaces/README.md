## Cluster workspaces

`cluster` workspaces are the canonical local operating surface for reusable runs.
Each workspace owns one `config.yaml`, its local inputs, and its generated outputs under `outputs/cluster/`.
List the packaged workspaces and their current output state with `uv run cluster workspace list`.
The singular `workspace` form matches the other workspace-owning tools. `uv run cluster workspaces ...` still works.

Expected layout:

```text
src/dnadesign/cluster/workspaces/
  <workspace-id>/
    config.yaml
    outputs/
      cluster/
```

Workspace contract:

- `config.yaml` is the machine-facing source of truth for `fit`, `umap`, and `analyze`.
- Checked-in demo workspaces may point at repo-local datasets or files using relative paths.
- Generated plots, embeddings, labels, summaries, and analysis artifacts stay under the selected workspace's `outputs/cluster/`.
- This workspace-owned output root is the one allowed runtime location inside the built-in `cluster/` tree.
- Built-in presets may still be referenced from workspace config, but workspace behavior does not depend on any separate legacy job layer.

Typical lifecycle:

```bash
# Inspect the packaged cluster workspaces and their current output state.
uv run cluster workspace list
# Show the active built-in workspace root.
uv run cluster workspace where
# Scaffold a new workspace under an explicit writable root.
uv run cluster workspace init --id my_run --root /tmp
# Fit one clustering run from the checked-in demo workspace.
uv run cluster fit --workspace promoter_clusters_v1
# Render UMAP artifacts for that fitted workspace run.
uv run cluster umap --workspace promoter_clusters_v1
# Generate analysis outputs for the same workspace run.
uv run cluster analyze --workspace promoter_clusters_v1
```

For ad hoc standalone runs, pass explicit CLI inputs plus `--results-root` instead of relying on workspace config.
