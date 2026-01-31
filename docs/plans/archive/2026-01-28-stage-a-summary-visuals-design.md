# Stage-A Summary Visual Redesign

Date: 2026-01-28
Status: accepted

## Context
Stage-A summary plots are informative but suffer from overlapping titles/annotations,
noisy tier markers, and legends that can obscure data. The goal is publication-quality
visuals without changing sampling semantics.

## Goals
- Eliminate title/subtitle/annotation collisions across Stage-A summary figures.
- Replace tier-line labels with a compact, non-overlapping summary box.
- Improve scatter encoding by faceting by regulator and removing legends.
- Reduce axis noise with shared x-axes and lighter tick density.
- Use a colorblind-safe categorical palette with stable TF color mapping.
- Make yield/dedupe a stepwise survival story (counts + stepwise %).

## Non-goals
- Change Stage-A sampling semantics, thresholds, or tier boundaries.
- Rename outputs or plot file names.
- Introduce new data dependencies beyond existing Stage-A manifests/pools.

## Decisions
- Use a dedicated GridSpec header row for title + subtitle; no manual `fig.text` y-positions.
- Update tier markers to short dashed vlines capped at ~0.58 axes height.
- Render tier thresholds and retained counts in a single anchored box per subplot.
- Replace yield/dedupe heatmap with per-regulator funnel/step plots showing counts and
  stepwise conversion percentages.
- Facet score vs length scatter by regulator; remove scatter legends entirely.
- Add palette alias `colorblind2` (Okabe-Ito) and stabilize TF colors:
  lexA → #0072B2, cpxR → #009E73.

## Implementation Notes
- Modify Stage-A plotting helpers in `plotting.py`:
  - `_draw_tier_markers` (ymax cap + boxed labels).
  - `_build_stage_a_strata_overview_figure` (header row + anchored box).
  - `plot_stage_a_summary` (funnel layout + shared colorbar column).
- Add `_add_anchored_box` helper using `matplotlib.offsetbox.AnchoredText`.
- Extend palette helper to accept `colorblind2` alias.
- Use shared x-axes with `MaxNLocator` and hide top tick labels.

## Testing Plan
- Add tests for `colorblind2` alias and tier marker vline height + anchored box.
- Update Stage-A plot tests to reflect header axis and layout changes.
