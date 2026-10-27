## `cluster` for agents

Supplement to repo-root `AGENTS.md` with `cluster`-specific structure + run conventions.

### Key paths
- README: `src/dnadesign/cluster/README.md`
- Docs index by workflow: `src/dnadesign/cluster/docs/README.md`
- Docs index by type: `src/dnadesign/cluster/docs/index.md`
- Workspaces: `src/dnadesign/cluster/workspaces/`
- Presets: `src/dnadesign/cluster/presets/` (`method`/`umap`/`plot`/`analysis`)
- Source: `src/dnadesign/cluster/src/`
- Run store: `workspaces/<workspace-id>/outputs/cluster/` or an explicit `--results-root`

### Contract: presets vs workspaces
Workspace `config.yaml` is the canonical reusable config surface.
If workspace config references a preset, do NOT duplicate the same method keys in the workspace section; the CLI errors.
For `fit` and method-scoped `sweep`, put reusable method knobs in the preset or under `method_params`, and use `--method-param key=value` for ad hoc overrides.

### Generated vs hand-edited
- Hand-edited: `workspaces/**/config.yaml`, `presets/**`, code, docs
- Generated: `workspaces/**/outputs/**` and standalone `--results-root` artifacts (plots, embeddings, indices, records.md)

### OPAL joins
If a hue references `obj__/pred__/sel__` columns missing from the dataset, cluster can join from an OPAL campaign (requires `--opal-*` flags). Don’t guess—use `--help`.

### Commands
```bash
uv run cluster --help
uv run cluster fit --help
uv run cluster umap --help
uv run cluster analyze --help

# Run built-in workspaces
uv run cluster workspaces where
uv run cluster workspaces init --help
uv run cluster workspaces list
uv run cluster fit --workspace promoter_clusters_v1
uv run cluster umap --workspace promoter_clusters_v1
uv run cluster analyze --workspace promoter_clusters_v1

# Optional helpers (confirm flags via --help)
uv run cluster intra-sim --help
uv run cluster sweep --help
uv run cluster delete-columns --help
```

### Environment variables (common)

* `DNADESIGN_USR_ROOT` (resolve datasets without passing full roots)
* `DNADESIGN_OPAL_CAMPAIGNS_ROOT` (resolve OPAL campaign names for joins)

### Tests

If you modify `cluster`, run:

```bash
uv run pytest -q
```
