# Cruncher Integration Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This page defines Cruncher schema mappings used by `baserender`.

## Contract intent

Cruncher-specific semantics stay in adapters and transform wiring.
`baserender` consumes those contracts and renders canonical `Record` instances.

## Primary runtime path (Cruncher analyze)

Cruncher `analyze` builds in-memory `Record` primitives for `elites_showcase` and calls BaseRender public APIs directly:

- `dnadesign.baserender.render_record_grid_figure(...)`
- `dnadesign.baserender.cruncher_showcase_style_overrides()`

Record semantics in this path:
- `features[*].id`: `<elite_id>:best_window:<tf>:<tf_index>`
- `features[*].attrs`: `{"tf": "<tf>"}`
- `effects[*].kind`: `motif_logo` with `target.feature_id` and PWM `params.matrix`
- `display.overlay_text`: `Elite #<rank>`

## BaseRender workspace demo mapping

`demo_cruncher_render` snapshots the same normalized record primitives into parquet so BaseRender can be exercised standalone with the same effective render contract used by Cruncher analysis plots.

### Normalized-record path

Adapter kind: `generic_features`

Expected source columns:
- `id`
- `sequence`
- `features`
- `effects` (optional)
- `display` (optional)

Use this path when upstream already emits canonical feature/effect structures.

### Elite-window path

Adapter kind: `cruncher_best_window`

Required config columns:
- `sequence`
- `id`
- `hits_path`
- `config_path`

This path maps Cruncher window/hit data into rendered features/effects when working from window-level artifacts instead of normalized records.

## Demo workspace

Validate and run the curated Cruncher demo workspace:

```bash
uv run baserender job validate --workspace demo_cruncher_render --workspace-root src/dnadesign/baserender/workspaces
uv run baserender job run --workspace demo_cruncher_render --workspace-root src/dnadesign/baserender/workspaces
```

For workspace-level operations, see `docs/demos/workspaces.md`.
