## OPAL Notebooks

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-21


OPAL notebooks are generated marimo campaign viewers. They are inspection
artifacts, not runtime control planes and not study-specific visual browsers.
Checked-in operator notebooks may use the public `dnadesign.opal.notebooks.api`
adapter for notebook discovery helpers; generated notebooks should continue to
import only `dnadesign.opal` public helpers plus general third-party packages.

Generate one with:

```bash
uv run opal notebook generate --config /path/to/campaign --round latest --force
```

Generate a campaign-set review notebook with repeated `--campaign` options:

```bash
uv run opal notebook generate \
  --campaign /path/to/campaign-a \
  --campaign /path/to/campaign-b \
  --out /path/to/opal_campaign_set_analysis.py
```

### Notebook View Model

Generated notebooks import public helpers from `dnadesign.opal` and build a
`NotebookViewModel` through `build_notebook_view_model(...)`. The view model is
manifest-backed and uses schema `opal.notebook_view_model.v1`.
Generated single-campaign notebooks also embed
`__opal_notebook_template_schema__ = "opal.generated_campaign_notebook.v1"` so
old local notebooks can be distinguished from current templates during review.

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

- campaign state at a glance;
- validity state for progress, review, plot, warning, and artifact-garden
  contracts;
- progress-derived change rows for visible rounds and run scope;
- records and X provenance;
- ledgers, labels, predictions, and selected records;
- manifest-backed plot cards;
- plot metric/data-shape definitions from plot-manifest metadata;
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
campaign summary rows, at-a-glance lines, validity lines, change summary lines
and rows, distrust/limitations lines, warning and stale-artifact evidence rows,
metric definition rows, artifact garden rows, manifest-backed plot-gallery
models, and plot card detail lines. Keep the generated source renderer in
`src/analysis/notebook_template/` as thin composition over small semantic cell
fragment modules. The reusable component surface lives in the
`src/analysis/notebook_components/` package; add new notebook UX as small
semantic modules there instead of growing a single component file.
Define marimo UI controls in one cell and read their `.value` in a downstream
cell; generated notebooks include a regression guard for this rule.

Campaign-set notebooks are intentionally overview-first: they provide campaign
and plot dropdowns, status and provenance summary, manifest-backed plot cards,
validity panels, change rows, metric definitions, artifact garden rows,
warnings, and stale-artifact evidence.
Single-campaign notebooks remain the record/table drill-down surface.
Single-campaign and campaign-set notebooks use the same public plot-gallery,
plot-card, validity, change-row, evidence-row, metric-definition, and
artifact-garden primitives.

### Boundaries

Canonical OPAL notebooks show OPAL campaign evidence only: records contract,
configured X column provenance, ledgers, progress, review manifests, selection
behavior, labels, predictions, plot artifacts, and limitations.

They must not render LatentDNA geometry, UMAP atlases, DenseGen-specific
visuals, or representation-browser content. Study/probe notebooks may link to
OPAL review artifacts, but OPAL-generated notebooks should remain
campaign-agnostic.

### Smoke Checks

Use `smoke_check_notebook(path)` for lightweight generated-file checks. When
marimo is available, run `marimo check` as part of the validation gate for
changed notebook templates.
