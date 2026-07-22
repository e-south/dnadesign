# Cruncher Integration Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-29


This page defines Cruncher schema mappings used by `baserender`.
The YIU family now has its own contract page at `docs/integrations/yiu.md`.

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

## Cassette visual-contract path

Cassette solve/design runs publish shared JSON and JSONL view contracts plus sibling job files directly into the owning Cruncher workspace:

- per-hit duplex view: `hits/hit_<rank>_<solution_id>/views/linear_duplex.v1.json`
- per-hit hairpin view: `hits/hit_<rank>_<solution_id>/views/ssdna_hairpin.v1.json`
- solve-level duplex sheet rows: `views/top_hits.linear_duplex.v1.jsonl`
- solve-level hairpin sheet rows: `views/top_hits.ssdna_hairpin.v1.jsonl`

Adapter kinds for this path:

- `duplex_sequence_v1` consumes `linear_duplex_v1`
- `hairpin_topology_v1` consumes `ssdna_hairpin_v1`

Input kinds for cassette handoff:

- `json` for single-view jobs
- `jsonl` for top-hit contact sheets

The generated `baserender_jobs/*.job.yaml` files are self-contained and render back into sibling `renders/` directories inside the same Cruncher workspace.

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
