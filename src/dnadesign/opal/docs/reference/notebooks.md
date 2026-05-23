## OPAL Notebooks

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-22


OPAL notebooks are generated marimo campaign analysis surfaces. They summarize
campaign state, records, ledgers, and visual artifacts for inspection; mutation
and long-running execution remain in the CLI.
Checked-in operator notebooks and generated campaign notebooks use the public
`dnadesign.opal.notebooks.api` adapter for notebook-specific helpers; generated
notebooks should import only that notebook API plus general third-party packages.

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

Generated notebooks import public helpers from `dnadesign.opal.notebooks.api` and
build a `NotebookViewModel` through `build_notebook_view_model(...)`. The view
model is manifest-backed and uses schema `opal.notebook_view_model.v1`.
Generated single-campaign notebooks also embed
`__opal_notebook_template_schema__ = "opal.generated_campaign_notebook.v1"` so
old local notebooks can be distinguished from current templates during review.
`opal notebook generate --json` emits schema `opal.notebook_generate.v1` with the
written notebook path, config paths, resolved round selector, optional pinned
run ID, and follow-up `opal notebook run` / `marimo check` commands.

| field | purpose |
| --- | --- |
| `progress` | campaign progress JSON from `build_campaign_progress` |
| `review_manifest` | latest or explicitly provided review manifest, when present |
| `plot_manifest_index` | aggregate `outputs/plots/plot_manifest.json`, when present |
| `plot_manifests` | per-plot manifests referenced by the index |
| `artifact_garden` | local artifact-root inventory, stale sibling list, byte counts, and dry-run prune plan |
| `stale_artifacts` | stale review or plot files not referenced by active manifests |
| `warnings` | missing manifests, stale files, or other nonfatal states |

Campaign-set notebooks import `build_campaign_set_notebook_view_model(...)` and
build schema `opal.notebook_campaign_set_view_model.v1`. The payload contains
one `NotebookViewModel` per campaign plus aggregate warnings. Campaign-set
notebooks require at least two distinct campaign configs and fail fast on
duplicates.

The generated notebook renders the view model with progressive disclosure:

- campaign state as a compact table;
- validity state for progress, review, plot, warning, and artifact-garden
  contracts;
- progress-derived change rows for visible rounds and run scope;
- selected round/run scope, with the run dropdown initialized to the pinned
  `--run-id` when one was provided;
- records and X provenance;
- ledgers, labels, predictions, and selected records;
- a single visual-surface selector for manifest-backed plots and optional
  record renders;
- a plot-scope selector when the active plot has multiple manifest-backed
  scopes, such as `all rounds`, `latest`, or per-round artifacts emitted by
  `round_variants`;
- plot metric/data-shape definitions from plot-manifest metadata;
- plot-local method, math, failure-mode, and evidence tables inside
  progressively disclosed accordions;
- artifact garden rows with local-only status, stale siblings, byte counts, and
  prune plans that require explicit apply outside the notebook;
- limitations and handoff commands.

The records preview is schema-pruned. It loads identity/metadata columns needed
for inspection and record selection, but it does not materialize the configured
X payload into the notebook preview. X is reported from the records schema and
kept as provenance unless a runtime command explicitly needs the matrix.

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

Campaign-set notebooks are intentionally overview-first: they provide campaign
and visual controls, manifest-backed plot-scope controls, status and provenance
summary, visible manifest-backed plot surfaces, validity panels, change rows,
metric definitions, artifact garden rows, warnings, and stale-artifact evidence.
Single-campaign notebooks remain the record/table drill-down surface.
Single-campaign and campaign-set notebooks use the same public visual-surface,
plot-card, plot-method, validity, change-row, evidence-row, metric-definition, and
artifact-garden primitives.

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
