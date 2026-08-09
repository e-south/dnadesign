# Expanded Reference and View Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-09

This spec defines the LatentDNA contract for promoter-reference visibility after
the stress ethanol/ciprofloxacin study expanded from three carried controls to a
larger reference inventory and from two candidate geometries to the completed
7B sequence-view matrix.

## Problem

The old stress-study LatentDNA surface treated `spyp`, `sulAp`, and `J23105` as
the only reference set, and it projected only the merged anchor-source insert
mean plus the forward 1 kb anchor-mean view. That was enough for the first
reference smoke check, but it is no longer enough for study status. Current
study records include native
reference rows, Anderson iGEM rows, W-collection rows, Native MG1655 controls,
and derived core60 rows.

The new contract is:

```text
reference inventory x geometry inventory x plot primitive
```

Every plot family must either cover this matrix or declare why it is limited.

## Reference Inventory

Workspace configs should expose named `reference_sets` for these slices:

- `reference_spyp_sulap`: small static label set for callout-heavy plots.
- `reference_spyp_sulap_core60`: core60 windows for the small spyP/sulAp control set.
- `reference_native_mg1655`: Native MG1655 source-length controls.
- `reference_native_mg1655_core60`: core60 windows derived from Native MG1655 controls.
- `reference_anderson_igem`: Anderson iGEM standards.
- `reference_anderson_igem_core60`: core60 windows derived from Anderson iGEM standards.
- `reference_w_collection`: W-collection standards.
- `reference_w_collection_core60`: core60 windows derived from W-collection standards.

Internal sampling may keep a non-notebook-exposed all-labeled preserve set, but
it should not appear as a user-facing annotation mode. Static callouts should
stay small. If a selected annotation set has more than five matched sequences,
the renderer should draw point markers only and keep complete membership in the
manifest/control payload.

## Geometry Inventory

The stress study should surface these 7B intermediate geometries:

- Mixed-length anchor-source insert sequence mean.
- Forward 1 kb context sequence mean.
- Forward 1 kb context anchor mean.
- Reverse-complement 1 kb context sequence mean.
- Reverse-complement 1 kb context anchor mean.
- Reference core60.
- Reference-context forward 1 kb sequence mean.
- Reference-context forward 1 kb anchor mean.
- Reference-context reverse-complement 1 kb sequence mean.
- Reference-context reverse-complement 1 kb anchor mean.

Output-layer views may remain `role: planned` until they are promoted as
evidence surfaces.

## Primitive Coverage

| Primitive | Contract |
| --- | --- |
| `projection_scatter`, `projection_grid` | Resolve selector-backed reference sets per panel, record expected/matched ids, draw small sets as callouts, and draw large sets as highlight-only overlays. |
| `curve`, `curve_grid` | Scree/collapse curves should be emitted for every promoted geometry group that has a reducer. Missing reducers are a regeneration gap, not an implicit pass. |
| `metric_panel_grid` | Reference alignment must not require only spyP/sulAp. It must emit selector-backed collapse metrics when config-declared `reference_sets` are configured, while `reference_group_columns` remain a broad metadata-audit fallback. |
| `xy_scatter_grid` | Strength/collapse scatter must facet or group by collection when strength scales differ. |
| `distribution_grid` | Within/between reference-distance distributions should group by `source_family`, `selection_basis`, and collection. |
| `heatmap_grid` | Pairwise reference-distance heatmaps should support full-set and per-collection subsets. |
| `categorical_count` | Dataset counts should see the expanded reference categories and sequence-view products. |
| Notebook/browser | Controls must expose promoted geometries and reference-set modes, even when some appendix views still need generated projections. |

## Implementation Slices

1. Reference-set schema supports explicit ids and selector-backed membership.
2. Sampling preserves selector-backed reference rows without expanding whole
   row ledgers through `Table.to_pylist()`.
3. Projection renderers resolve selector-backed reference sets and persist
   per-panel completeness in manifests.
4. Highlight-only reference sets draw visible overlay markers without dense
   labels.
5. Reference alignment emits collection-aware collapse rows keyed by
   `reference_group_column` and `reference_group`.
6. Notebook controls publish promoted geometry rows and reference-set modes.
7. The stress workspace promotes the completed intermediate sequence-view
   inventory to appendix visibility and keeps output-layer views planned.

## Validation

Use this minimum gate after edits:

```sh
uv run pytest -q \
  src/dnadesign/latentdna/tests/test_sampling_contracts.py::test_sample_build_stratified_preserves_selector_reference_set_rows \
  src/dnadesign/latentdna/tests/integrations/test_runtime_progress.py::test_projection_grid_resolves_selector_reference_set_in_every_panel \
  src/dnadesign/latentdna/tests/test_scalar_build.py::test_reference_alignment_summary_emits_collection_collapse_without_spyp_sulap \
  src/dnadesign/latentdna/tests/contracts/test_study_workspace_contracts.py::test_live_study_browser_controls_expose_sidecar_geometry_inventory

uv run latentdna validate workspace --workspace /path/to/workspace --deep
uv run latentdna workspace snapshot --workspace /path/to/workspace --json
```

Heavy regeneration of missing appendix projections is intentionally separate
from the schema/control-plane gate because those UMAPs require materializing
additional 157k-row matrices.
