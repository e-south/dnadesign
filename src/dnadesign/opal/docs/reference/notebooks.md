## OPAL Notebooks

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-20


OPAL notebooks are generated marimo campaign viewers. They are inspection
artifacts, not runtime control planes and not study-specific visual browsers.

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

| field | purpose |
| --- | --- |
| `campaign_state` | campaign progress JSON from `build_campaign_progress` |
| `review_manifest` | latest or explicitly provided review manifest, when present |
| `plot_manifest_index` | aggregate `outputs/plots/plot_manifest.json`, when present |
| `plot_manifests` | per-plot manifests referenced by the index |
| `stale_artifacts` | stale review or plot files not referenced by active manifests |
| `warnings` | missing manifests, stale files, or other nonfatal states |

Campaign-set notebooks import `build_campaign_set_notebook_view_model(...)` and
build schema `opal.notebook_campaign_set_view_model.v1`. The payload contains
one `NotebookViewModel` per campaign plus aggregate warnings. Campaign-set
notebooks require at least two distinct campaign configs and fail fast on
duplicates.

The generated notebook renders the view model with progressive disclosure:

- campaign state at a glance;
- records and X provenance;
- ledgers, labels, predictions, and selected records;
- manifest-backed plot cards;
- limitations and handoff commands.

Heavy sections should use marimo accordions with lazy loading. Reusable
generated-cell builders live in `src/analysis/notebook_components.py`; keep
`notebook_template.py` as thin wiring and move independently testable panels out
of the template before adding new notebook UX.

Campaign-set notebooks are intentionally overview-first: they provide campaign
and plot dropdowns, status and provenance summary, manifest-backed plot cards,
warnings, and stale-artifact evidence. Single-campaign notebooks remain the
record/table drill-down surface.

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
