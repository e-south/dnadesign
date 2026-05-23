## OPAL Notebooks

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-22


OPAL notebooks are generated marimo campaign analysis surfaces. They summarize
campaign state, progress, ledgers, and visual artifacts for inspection; mutation
and long-running execution remain in the CLI.
Checked-in operator notebooks and generated campaign notebooks use the public
`dnadesign.opal.notebooks.api.generated` adapter for notebook-specific helpers;
generated notebooks should import only that notebook API plus general
third-party packages.

Generate one with:

```bash
uv run opal notebook generate --config /path/to/campaign --round latest --force --json
```

Pin a single-campaign notebook to a specific rerun when a round has multiple
`run_id` values:

```bash
uv run opal notebook generate --config /path/to/campaign --round latest --run-id <run-id> --force --json
```

Generate a campaign-set review notebook with repeated `--campaign` options:

```bash
uv run opal notebook generate \
  --campaign /path/to/campaign-a \
  --campaign /path/to/campaign-b \
  --out /path/to/opal_campaign_set_analysis.py
```

### Notebook View Model

Generated notebooks import public helpers from
`dnadesign.opal.notebooks.api.generated`. The canonical generated surface is the
campaign-set template, and a single-campaign notebook is the same template with
one campaign config. Generated notebooks embed
`__opal_notebook_template_schema__ = "opal.generated_campaign_notebook.v2"` so
old local notebooks can be distinguished from current templates during review.
`opal notebook generate --json` emits schema `opal.notebook_generate.v1` with the
written notebook path, config paths, resolved round selector, optional pinned
run ID, and follow-up `opal notebook run` / `marimo check` commands.

Each campaign entry is a manifest-backed `NotebookViewModel` with schema
`opal.notebook_view_model.v1`; the enclosing campaign surface uses
`opal.notebook_campaign_set_view_model.v1`.

| field | purpose |
| --- | --- |
| `progress` | campaign progress JSON from `build_campaign_progress` |
| `review_manifest` | latest or explicitly provided review manifest, when present |
| `plot_manifest_index` | aggregate `outputs/plots/plot_manifest.json`, when present |
| `plot_manifests` | per-plot manifests referenced by the index |
| `artifact_garden` | local artifact-root inventory, stale sibling list, byte counts, and dry-run prune plan |
| `stale_artifacts` | stale review or plot files not referenced by active manifests |
| `warnings` | missing manifests, stale files, or other nonfatal states |

Campaign surfaces import `build_campaign_set_notebook_view_model(...)`. The
payload contains one `NotebookViewModel` per campaign plus aggregate warnings.
The builder accepts one or more distinct campaign configs and fails fast on
duplicates. `--run-id` pinning is supported only when the surface has exactly
one campaign, because a single run ID is not portable across a campaign set.

The generated notebook renders the view model with progressive disclosure:

- campaign state as a compact table;
- validity state for progress, review, plot, warning, and artifact-garden
  contracts;
- progress-derived change rows for visible rounds and run scope;
- campaign selector, even for one-campaign notebooks;
- round selector for progress and manifest-backed plot scope;
- a single visual-surface selector for manifest-backed plots;
- a plot-scope selector when the active plot has multiple manifest-backed
  scopes, such as `all rounds`, `latest`, or per-round artifacts emitted by
  `round_variants`;
- plot metric/data-shape definitions from plot-manifest metadata;
- plot-local method, math, failure-mode, and evidence tables inside
  progressively disclosed accordions;
- artifact garden rows with local-only status, stale siblings, byte counts, and
  prune plans that require explicit apply outside the notebook;
- limitations and evidence rows.

The canonical generated notebook no longer has a separate single-campaign
record/table drilldown path. Records, labels, predictions, and selected-record
inspection remain CLI/API concerns unless they are promoted through a
manifest-backed OPAL plot or another public notebook component. This keeps the
single-campaign and multi-campaign surfaces from drifting.

Heavy sections should use marimo accordions with lazy loading. Reusable
generated-cell builders and public component primitives live in
`src/analysis/notebook_components/`. Current reusable primitives cover
campaign summary rows, at-a-glance rows, validity lines, change summary lines
and rows, distrust/limitations lines, warning and stale-artifact evidence rows,
metric definition rows, artifact garden rows, manifest-backed visual-surface
models, compact path labels, plot detail rows, plot method rows, and optional
BaseRender record-render contracts. Keep the generated
source renderer in `src/analysis/notebook_template/` as thin composition over
small semantic cell fragment modules. The reusable component surface lives in the
`src/analysis/notebook_components/` package; add new notebook UX as small
semantic modules there instead of growing a single component file.
Define marimo UI controls in one cell and read their `.value` in a downstream
cell; generated notebooks include a regression guard for this rule.

Campaign surfaces are intentionally overview-first: they provide campaign and
visual controls, manifest-backed plot-scope controls, status and provenance
summary, visible manifest-backed plot surfaces, validity panels, change rows,
metric definitions, artifact garden rows, warnings, and stale-artifact evidence.
The checked-in `src/dnadesign/opal/notebooks/campaign_progress.py` notebook is
now a project-scoped campaign surface over discovered campaign configs, not a
separate progress-only UI.

### Boundaries

Canonical OPAL notebooks show OPAL campaign evidence only: records contract,
configured X column provenance, ledgers, progress, review manifests, selection
behavior, labels, predictions, plot artifacts, and limitations.

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

### Smoke Checks

Use `smoke_check_notebook(path)` for lightweight generated-file checks. When
marimo is available, run `marimo check` as part of the validation gate for
changed notebook templates.
