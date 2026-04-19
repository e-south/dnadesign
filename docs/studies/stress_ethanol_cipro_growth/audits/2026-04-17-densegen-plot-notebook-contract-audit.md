## DenseGen Plot And Notebook Contract Audit

Date: 2026-04-17
Study: `stress_ethanol_cipro_growth`
DenseGen workspace: `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro`
Dataset basis: overlay-aware `usr:densegen/study_stress_ethanol_cipro` at `157,160` accepted arrays

### Purpose

This audit package is meant to support a follow-on development specification for the DenseGen promoter-study analysis surface: plot generation, plot manifest behavior, notebook gallery behavior, Stage A to Stage B bridge semantics, and study-record alignment.

### Swarm Run Contract

- Objective: identify contract drift, silent fallbacks, missing invariants, and missing artifacts that materially affect how the DenseGen study is interpreted.
- Execution mode: `platform_subagent`
- Topology: `star`
- Context carry: `continue`
- Bounds: `max_workers=3`, `max_depth=1`
- Worker surfaces:
- `plot/default-surface alignment`
- `stage_a_stage_b_bridge semantics`
- `notebook/loader/test coverage`
- Local verification: targeted DenseGen pytest slice

### Prioritized Findings

1. High: the DenseGen default plot surface is not one contract.
`docs/studies/stress_ethanol_cipro_growth/pipeline.yaml` declares `dataset_metadata_heatmap` in the default plot set and treats the video as optional, but the executable workspace config defaults to `[dataset_source_inventory, stage_a_summary, placement_map, run_health, tfbs_usage]` and `run_plots_from_config()` auto-appends `dense_array_video_showcase` whenever `plots.video.enabled` is true. The notebook gallery then hides `dataset_metadata_heatmap`. The study docs, workspace defaults, and notebook-visible surface therefore describe different “defaults”.

2. High: the Stage-A companion plot is not honoring its own bridge contract.
The `sampling_vs_length_ridgeline` variant is described as showing accepted TFBS length counts by regulator, but the normal `stage_a_summary` render path only declares `pools` as required input. The companion receives the Stage-A dataframe and only opportunistically uses output records if they were loaded for some other selected plot. When accepted-output annotations are missing, the right panel silently falls back to retained Stage-A pool lengths. That means the same chart can represent different populations while keeping the same title and legend.

3. High: the ridgeline left-panel “retained cutoff” is semantically wrong.
The dashed line is the minimum score among the MMR-retained set, not a simple “top-500” score boundary. Retention is decided after MMR diversification from a `5000`-candidate prepool, so there is no single scalar score threshold that defines inclusion. The current label overstates certainty.

4. Medium: plot-manifest behavior is cumulative, while the notebook surface behaves like a current snapshot.
`_write_plot_manifest()` keeps prior manifest entries whose files still exist, and the notebook gallery also walks `outputs/plots/` directly. Old artifacts can therefore remain visible after config or default changes unless they are explicitly pruned. The current docs do not state whether `plot_manifest.json` is authoritative, append-only, or stale-tolerant.

5. Medium: Stage-B usage panels use hidden denominator fallbacks.
`tfbs_usage` and `placement_map` clamp `available_unique` to at least `1` when library-members data is absent, which can print a coverage-looking ratio even when the denominator was never observed. This is a silent fallback where the operator-facing output should instead say “availability unavailable” or fail.

6. Medium: the run_health TFBS-length panel overlaps the ridgeline right panel but does not define a stronger invariant.
`run_health/tfbs_length_by_regulator` uses `composition.parquet` and can compute candidate-pool sizes, but it drops the denominator and renders raw counts only. The ridgeline right panel uses `densegen__used_tfbs_detail` from the output-records source. These are two related panels fed by different data planes, with no explicit statement of which one is canonical for bridge auditing.

7. Medium: the notebook USR preview path is real behavior, but only partially documented and only lightly tested.
The scaffold generates `outputs/notebooks/records_with_overlays.parquet` and `.baserender_preview_cache/`, but the workspace runbook and tutorial mostly describe the visible notebook and plots. DenseGen tests verify template wiring and small synthetic fixtures, not an end-to-end USR-backed notebook generation path against the shared analysis flow.

8. Medium: study-level downstream attention surfaces are overstated.
The promoter-study `status.md` and `routes.md` present `appendix_umap_gallery` as part of the current attention surface, but the LatentDNA notebook only exposes that preset when all eight canonical views are materialized, and its actual notebook surface is `latent_geometry_browser`.

### Deep Introspection

#### Decision Summary

- Target scope: DenseGen study analysis surface spanning `dense plot`, `plot_manifest.json`, the generated notebook, and the study record that advertises those outputs.
- Depth: deep
- Audience: maintainer and spec author
- In scope: plot and notebook runtime behavior, config-to-surface mapping, Stage A to Stage B bridge semantics, artifact visibility, and verification coverage.
- Out of scope: changing the figures or implementing the dev spec.

#### Intent And Use-Case Map

- Primary intent: provide an operator-facing analysis layer that explains what DenseGen generated, why Stage A pools look the way they do, and how accepted Stage B outputs reflect or distort those pools.
- Primary use cases:
- audit Stage A pool quality
- inspect Stage B occupancy and TFBS usage
- browse a study run in the notebook
- hand off a shared 60 bp anchor dataset into infer and construct workflows
- Secondary use cases:
- artifact export for reports
- study-status reporting
- operator triage when runs are partial or stalled
- Non-goals:
- the DenseGen plots are not the downstream `full_context_1kb` comparison surface
- the notebook is not a full provenance database

#### Core Functionality And Behavior Contract

- `dense plot` resolves a selected plot set from workspace defaults, optionally auto-adds the video artifact, loads only the sources each plot family claims to require, renders artifacts, and writes/merges `plot_manifest.json`.
- `dense notebook generate` scaffolds a marimo notebook that reads run artifacts, resolves plot entries from the manifest and filesystem, filters hidden plot types from the operator-facing gallery, and renders BaseRender previews from a records source.
- The Stage-A companion is the only built-in cross-stage plot. Its left panel is driven by Stage A pool statistics; its right panel is intended to summarize deployed TFBS lengths from Stage B output annotations.
- The records loader is overlay-aware for USR sources and can recover missing `densegen__plan` / `densegen__input_name` from stable `source` labels.

#### Lifecycle Model

1. Stage A mines PWM/background candidates, deduplicates and collapses by core, then retains `500` sites per pool through MMR from a `5000`-candidate prepool.
2. Stage B samples from those retained pools across `20` expanded plans, writes accepted sequences to parquet and USR, and records placement detail in `densegen__used_tfbs_detail`.
3. Plot generation loads pools, outputs, composition, attempts, and config according to selected plot families, renders artifacts, and merges manifest state.
4. Notebook generation opens the records source, exports merged USR records to a hidden parquet when needed, reconstructs a plot inventory from manifest plus filesystem, and exposes a filtered gallery.
5. Study docs advertise a higher-level interpretation surface over the DenseGen outputs and downstream LatentDNA comparison.

#### Architecture View Stack

- Context view:
DenseGen writes the shared source dataset for the promoter study. Study docs point at that dataset, DenseGen artifacts, and downstream LatentDNA/infer/construct surfaces.
- Module view:
`adapters/outputs/loader.py` handles parquet versus USR record loading and overlay recovery.
`viz/plotting.py` is the plot orchestrator and manifest writer.
`viz/plot_stage_a*.py`, `plot_run.py`, `plot_run_panels.py`, and `plot_stage_b_placement.py` implement figure families.
`viz/plot_inventory.py` defines notebook-facing plot typing, gallery visibility, and text contracts.
`cli/notebook*_template*.py` scaffolds the notebook behavior.
- Runtime scenario:
config -> source resolution -> dataframe loading -> figure rendering -> manifest merge -> notebook plot discovery -> notebook gallery filtering -> operator interpretation

#### Config To Behavior And Architecture Mapping

- `plots.default`
Controls the nominal default plot selection, but is not the whole default surface because video can be auto-added and hidden plot types can still exist on disk.
- `plots.video.enabled`
When true and `--only` is not used, `run_plots_from_config()` auto-appends `dense_array_video_showcase`.
- `plots.options.stage_a_summary.include_sampling_length_companion`
Enables the ridgeline companion, but does not itself force output-record loading.
- `plots.options.{placement_map,tfbs_usage}.scope` and `max_plans`
Cause the study’s 20 expanded plans to collapse into 4 grouped Stage-B plot scopes when `scope:auto` and `max_plans=12`.
- `plots.source`
Determines whether output-record-backed plots and the notebook use parquet or USR. For this study it is `usr`.
- `output.targets`
Both parquet and USR are emitted, but analysis uses the source chosen by `plots.source`.

#### Interaction Map

- Upstream:
Cruncher motif artifacts and Stage A pools feed DenseGen.
DenseGen output rows feed dataset plots, Stage-B plots, and the notebook.
- Lateral:
`plot_manifest.json` is shared between plot generation and notebook discovery.
`densegen__used_tfbs_detail` is the bridge contract across output rows, notebook preview, and some plots.
- Downstream:
study docs, promoter-study status, and LatentDNA routes use DenseGen artifacts and dataset counts as a study-facing interpretation layer.

#### Math And Operations Notes

- Stage A selection is not “top N by score”; it is MMR over a bounded prepool.
- The current study has `157,160` accepted arrays, `471,480` variable TFBS placements, and `314,320` fixed sigma70 parts.
- The 20 expanded plans are currently summarized into 4 grouped Stage-B plot families because of `scope:auto` and `max_plans: 12`.
- Cross-stage compression is real and spec-worthy:
- `lexA` retained pool is `61.2%` at `19-20 bp`, but deployed placements are only `24.7%` at `19-20 bp`
- `lexA` deployed `17-18 bp` share rises to `69.0%`
- similar shortening occurs for `cpxR`, `baeR`, and `background`

### Missing Visuals And Missing First-Class Outputs

- `accepted arrays by plan` barplot, ideally both grouped by base plan and expanded across all `20` variants
- `retained vs deployed length shift` per regulator
- `used unique sites / 500 retained` and `placements / unique used`
- `tier mix retained vs deployed`
- `plan x regulator` presence or length-composition heatmap
- `upstream evidence and MEME quality` panel with source-site counts, motif widths, and E-values
- an explicit `coverage unavailable` visual state when library-members denominators are absent

### Dev-Spec Requirements

1. Define one operator-facing default plot surface and one internal-but-generated surface.
2. Define whether `plot_manifest.json` is a current snapshot or cumulative inventory.
3. Make the Stage-A companion fail closed when accepted-output annotations are missing or incomplete.
4. Rename the ridgeline cutoff marker to match the actual statistic, or add the true prepool/retained markers needed to support a threshold interpretation.
5. Define which plot is canonical for TFBS-length bridge auditing: the ridgeline right panel, the run_health length panel, or a new dedicated bridge plot.
6. Remove denominator fabrication from Stage-B usage panels.
7. Specify hidden notebook artifacts and their refresh lifecycle.
8. Add at least one end-to-end USR-backed notebook test and one manifest-pruning regression test.
9. Separate study-record attention surfaces from conditional LatentDNA notebook presets.

### Verification Evidence

- Targeted verification command passed:
`uv run pytest -q src/dnadesign/densegen/tests/stage_a/test_stage_a_sampling_length_ridgeline.py src/dnadesign/densegen/tests/plotting/test_plot_manifest.py src/dnadesign/densegen/tests/plotting/test_plot_inventory.py src/dnadesign/densegen/tests/cli/test_notebook_records_projection.py src/dnadesign/densegen/tests/runtime/test_records_loader_compat.py src/dnadesign/densegen/tests/plotting/test_dataset_plots.py`
- Result: `48` tests passed.

### Key Evidence Ledger

- Study pipeline default plot claim: `docs/studies/stress_ethanol_cipro_growth/pipeline.yaml`
- Study status and current attention surface: `docs/studies/stress_ethanol_cipro_growth/status.md`
- Study downstream route ladder: `docs/studies/stress_ethanol_cipro_growth/routes.md`
- DenseGen workspace runtime defaults: `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml`
- Plot runner and manifest merge behavior: `src/dnadesign/densegen/src/viz/plotting.py`
- Plot inventory hidden-surface behavior: `src/dnadesign/densegen/src/viz/plot_inventory.py`
- Notebook gallery manifest-plus-filesystem discovery: `src/dnadesign/densegen/src/cli/notebook_cells_template_gallery.py`
- Notebook artifact freshness gate: `src/dnadesign/densegen/src/cli/notebook.py`
- Stage-A companion rendering path: `src/dnadesign/densegen/src/viz/plot_stage_a.py`
- Ridgeline semantics and fallback logic: `src/dnadesign/densegen/src/viz/plot_stage_a_sampling_length_ridgeline.py`
- Run-health TFBS length panel: `src/dnadesign/densegen/src/viz/plot_run_panels.py`
- Overlay-aware record loading and metadata recovery: `src/dnadesign/densegen/src/adapters/outputs/loader.py`, `src/dnadesign/densegen/src/core/record_metadata_recovery.py`
- LatentDNA notebook default and conditional appendix preset: `src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/config.yaml`, `src/dnadesign/latentdna/src/services/notebook_controls_service.py`, `src/dnadesign/latentdna/src/services/notebook_service.py`

### Recommended Next Input To Spec Synthesis

Hand the next synthesis pass three explicit questions:

1. What is the canonical operator-facing DenseGen analysis surface for this study?
2. What are the hard failure conditions versus degraded states for bridge plots and notebook artifacts?
3. Which quantitative invariants must be visible in plots rather than left implicit in code or tables?
