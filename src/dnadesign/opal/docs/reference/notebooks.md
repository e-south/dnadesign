<!--
---
id: opal-reference-notebooks
title: OPAL notebooks
owner: dnadesign-maintainers
status: active
last_verified: 2026-10-28
audience:
  - operator
  - maintainer
  - agent
entrypoints:
  cli: uv run opal notebook
  api: dnadesign.opal.notebooks.api.generated
---

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[OPAL docs](../index.md)

# Review campaigns in notebooks

OPAL generates marimo notebooks for browsing saved predictions, selections,
plots and campaign progress. Use the CLI to import measurements and run a
selection round; use the notebook to inspect the results.

## Open a completed campaign

Install the [notebook dependencies](../index.md#try-a-complete-selection-round),
activate that environment, and complete at least one
[campaign round](../workflows/campaign-round.md). From your campaign directory:

```bash
# Create a notebook from the latest saved round.
opal notebook generate -c configs/campaign.yaml \
  --round latest --out notebooks/campaign.py --json
# Open it in read-only app mode.
opal notebook run -c configs/campaign.yaml --path notebooks/campaign.py
```

The generation command reports the notebook path and follow-up commands. In the
app, choose a selection view and a review section to inspect its saved results.
Shared model diagnostics appear once; view-specific plots and selections retain
their view identity. Close the server with Ctrl+C in the terminal when finished.

The generation response includes these fields (excerpt):

```json
{
  "ok": true,
  "notebook_path": "notebooks/campaign.py",
  "round_selector": "latest",
  "overwritten": false
}
```

Use `--round all` during generation to include the campaign's recorded history.
When a round has several runs, inspect `opal runs list` and pass `--run-id` to
pin the notebook to the intended run. Add `--force` only when you want to
replace an existing generated notebook.

To change the notebook code, open editable marimo mode:

```bash
# Edit the generated notebook rather than opening the read-only app.
opal notebook edit -c configs/campaign.yaml --path notebooks/campaign.py
```

## Compare campaigns

Pass each campaign explicitly to review them together:

```bash
# Replace these paths with the campaign directories you want to inspect.
opal notebook generate \
  --campaign /path/to/campaign-a \
  --campaign /path/to/campaign-b \
  --out notebooks/campaign-comparison.py
```

For grouped comparisons, add `--collection /path/to/campaign_collection.yaml`.
The collection manifest declares which campaigns can be compared and how their
results are grouped; its schema is `opal.campaign_collection.v2`. By default,
generation writes comparison CSV, PNG and manifest files under
`collection_visuals/` beside the notebook. The notebook reads these saved
results through `opal.collection_visual_manifest_index.v1`.

Use `--no-materialize-collection-visuals` only when testing a template without
rebuilding its comparison outputs.

## Notebook data and integration reference

Generated notebooks import public helpers from
`dnadesign.opal.notebooks.api.generated`. Single-campaign and multi-campaign notebooks use the same template;
a single-campaign notebook supplies one configuration. Generated notebooks embed
`__opal_notebook_template_schema__ = "opal.generated_campaign_review_notebook.v6"` so
non-current local notebooks can be distinguished from current templates during review.
`opal notebook generate --json` emits schema `opal.notebook_generate.v1` with the
written notebook path, config paths, resolved round selector, optional pinned
run ID, optional `collection_manifest_path`, optional
`collection_visual_index_path`, and follow-up `opal notebook run` / `marimo
check` commands. App review uses `opal notebook run`, which delegates to
`marimo run`. Editable authoring uses the separate `opal notebook edit` command.

Each campaign entry is a manifest-backed `NotebookViewModel` with schema
`opal.notebook_view_model.v1`; the enclosing campaign set uses
`opal.notebook_campaign_set_view_model.v1`.

| field | purpose |
| --- | --- |
| `progress` | campaign progress JSON from `build_campaign_progress` |
| `review_manifests` | map of selection-view ID to review manifest |
| `review_manifest_paths` | map of selection-view ID to canonical manifest path |
| `plot_manifests` | per-plot manifests referenced by the index |
| `artifact_garden` | local artifact-root inventory, stale sibling list, byte counts, and dry-run prune plan |
| `selection_batch` | final deduplicated proposal for the resolved run |
| `stale_artifacts` | stale review or plot files not referenced by active manifests |
| `warnings` | missing manifests, stale files, or other nonfatal states |

Generated notebooks import `build_campaign_set_notebook_view_model(...)`. The
payload contains one `NotebookViewModel` per campaign plus aggregate warnings,
an optional `collection`, and optional materialized
`collection_visuals`. The builder accepts one or more distinct campaign configs
and rejects duplicates. `--run-id` pinning requires exactly one campaign,
because run IDs identify executions within a campaign. Collection manifests and
collection visual indexes require at least two explicit campaign configs.

The generated notebook presents:

- campaign state as a compact table;
- validity state for progress, review, plot, warning, and artifact-garden
  contracts;
- progress-derived change rows for visible rounds and run scope;
- campaign selector when two or more explicit campaign configs are loaded;
- selection-view selector for the selected campaign; the chosen view controls
  objective parameters, target mask, ranks, selected rows, and plot manifests;
- round selector for progress and manifest-backed plot scope;
- one human-readable campaign title. The campaign description, observed-label
  evidence status and claim boundary, and active objective target stay in a
  closed `Campaign context` accordion immediately below the navigation controls;
- a top-level `Review scope` control only when there is a real campaign versus
  cross-campaign choice: `Campaign` for one selected campaign's plot
  deliverables, and `Cross-campaign comparison` for manifest-backed comparisons
  between genuinely independent campaigns. Selection views never activate or
  stand in for a cross-campaign comparison;
- a `Comparison group` selector for explicit matched campaign groups, such as
  one positive/null control pair for a target/family/split;
- a consolidated, wrapped `mo.hstack` that keeps campaign, selection view,
  review section, deliverable, and any scoped follow-on control together before
  the selected media. `Review section` is the
  stable progressive-disclosure layer: `Decision review`, `Assay evidence`,
  `EDA comparisons`, `Model diagnostics`, `Method diagnostics`, and `Handoff`.
  The deliverable selector owns OPAL plot artifacts, selected-sequence renders,
  collection comparison visuals, and Reader evidence plot types within the
  active section. Follow-on controls are scoped to the selected deliverable:
  plot scope and manifest-declared layer controls for OPAL plots, selection
  round/run/sequence for selected-sequence renders, and Reader plot instance or
  time controls for Reader deliverables. Campaign, selection-view,
  review-section, deliverable, and plot-scope choices preserve valid state when
  another control changes. Manifest-declared layered scatters resolve their layer
  controls from the concrete plot scope, so observed-batch choices change with
  the selected round or run. Each exact batch keeps one marker across toggle
  states. The adapter fails before rendering when a plot scope contains more
  than 12 exact batches, rather than cycling or reassigning markers.
  Prediction-pool, selected-candidate, observed-batch, and annotation controls
  remain independent. The prediction-pool base layer
  contains every prediction; `Selected` adds or removes the active-view
  highlight without deleting those candidates from the base layer. Control
  memory is scoped by campaign and plot identity, annotation scope is limited
  to visible layers, and hiding every layer produces a compact empty state
  rather than a notebook error;
- layered-scatter plot manifests declare their own x/y reference lines and
  color-center interpretation. The notebook renders only those declared lines
  and uses the declared color text. It does not assume that zero means a
  feasibility boundary; threshold-free objectives may declare no reference
  lines and describe zero only as a reference direction;
- one selected media viewport above secondary tables, so app-mode review can
  iterate OPAL and Reader deliverables through one selector without opening
  detail sections. When a campaign has Reader evidence but no OPAL plot
  manifest, the media region shows Reader evidence without a misleading empty
  OPAL plot panel;
- an EDA comparison section for value-reference views such as effect-scaled
  response versus logic fidelity, fold-change response versus logic fidelity,
  and pooled campaign-set selection overlap. These views support review of
  selection behavior and pooled-build pressure; they do not claim measured
  stress response before follow-up labels exist;
- plot metric/data-shape definitions from plot-manifest metadata;
- plot-local method, math, failure-mode, and evidence tables in a compact
  progressively disclosed evidence section;
- Reader evidence manifests, artifact tables, and renderable Reader plot
  artifacts when a campaign stages measured Reader evidence. Current
  four-state event-window diagnostic records retain their Reader record ID,
  reduction identity, and selected event-relative window;
- artifact garden rows with local-only status, stale siblings, byte counts, and
  prune plans that require explicit apply outside the notebook;
- limitations and evidence rows in the campaign status detail section.

`campaign_collection.v2` is a semantic manifest, not a plot file. It declares
collection dimensions, relationship lenses, and explicit `comparison_views`.
OPAL supports `metric_over_rounds_comparison` views over validated
campaign relationships, `vector_reference_mse_over_rounds_comparison` views
from `vector_summary_heatmap` tidy rows, `vector_heatmap_comparison` views
that render side-by-side group heatmaps plus a shared MSE trajectory panel, and
`paired_plot_gallery` views from manifest-backed source plot images. Each
comparison view declares its source plot, grouping dimension,
`comparison_scope`, optional `match_filters`, and the estimator/interval fields
required by its kind. A comparison view may also
carry an `interpretation_note`; OPAL preserves that note in the materialized
visual manifest and notebook caption without interpreting study-specific
biology. `comparison_scope:
comparison_set` materializes one visual per matched set, avoiding accidental
IQR bands across unrelated targets or label families. OPAL may render generic
IQR bands or Student-t mean confidence intervals only when the manifest
explicitly declares the interval kind, replicate unit, mean estimator, and
confidence level. Materialized visual manifests may also carry a generic
`axis_scale` object derived from source plot params, including comparable
y-limits and reference lines. OPAL applies that scale mechanically; the study
owns the scientific meaning of scale classes such as negative MSE or target
count. DenseGen targets, oracle meaning, split logic, and suite-level
interpretation remain study-owned.

Relationship pairs use campaign-member identity, not display names. If a
campaign set contains duplicate campaign slugs, such as the same DenseGen
campaign repeated across seed roots, OPAL disambiguates members with
`config_path` or `workdir` before building collection rows. This prevents
seed-replicate comparisons from cross-joining rows that happen to share a
slug.

## Extend a generated notebook

Use [`opal record-show`](cli.md#record-show) to inspect an individual record's
measurements, predictions and ranks. Notebook displays read saved plot manifests
and the public `dnadesign.opal.notebooks.api.generated` helpers.

The generator composes cells from `src/analysis/notebook_template/` and shared
components in `src/analysis/notebook_components/`. Add reusable display
components there. Keep review controls and the selected plot visible near the
top; use accordions for secondary status and evidence tables. Interactive
controls must remain reactive when their values change.

Define marimo controls in one cell and read their `.value` in a downstream cell.
Generated notebooks include a regression check for this rule.

For a multi-view campaign, the top hierarchy is `Campaign | Selection view |
Review section | Deliverable | optional plot scope and layer controls`. View-specific plots live under
`outputs/plots/selection_views/<view_id>/`; shared model diagnostics appear
once. A selection-batch deliverable reports the final deduplicated proposal and
view memberships. Do not create a campaign-set notebook merely to compare
setpoints that share one learning lifecycle.

A plot that needs interactive layers declares a notebook adapter in its plot
metadata and publishes one canonical media/tidy manifest. The notebook derives
controls from that concrete manifest scope rather than pairing independently
rendered variants. This keeps one deliverable identity while allowing prediction,
selection, observed-batch, and annotation visibility to change independently.

## External evidence and plots

OPAL notebooks combine saved campaign results with explicitly supplied
external evidence manifests: records contracts, configured X provenance,
ledgers, progress, review manifests, selection behavior, labels, predictions,
plot artifacts, Reader evidence artifacts, and limitations. Reader artifacts
remain Reader-authored; OPAL displays their manifest-backed outputs and does not
rebuild assay reductions or composite figures.

They must not render LatentDNA geometry, UMAP atlases, or representation-browser
content. They may expose a BaseRender record view only when the records schema
satisfies a public adapter contract such as generic feature annotations or
sequence-feature annotations. That contract is detected from records columns and
is optional; OPAL notebooks must not require producer-specific browser state to
render.

Study-specific visuals may appear in OPAL notebooks only through the OPAL plot
registry and artifact-manifest contract. A study can provide a plugin plot kind,
but generated notebooks consume it the same way as built-ins: `PlotMeta`
metadata, configured plot entries, written media/tidy outputs, and
`opal.plot_artifact.v1` manifests. Arbitrary study report images remain a
separate report layer unless they are produced through that API.

## Check notebook changes

Use `smoke_check_notebook(path)` for lightweight generated-file checks. When
marimo is available, run `marimo check` as part of the validation gate for
changed notebook templates.
