## Cruncher architecture

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-23


**Last updated by:** cruncher-maintainers on 2026-05-14

### Contents
- [Cruncher architecture](#cruncher-architecture)
- [Workflow families](#workflow-families)
- [Fixed-length optimization lifecycle](#fixed-length-optimization-lifecycle)
- [Cassette lifecycle](#cassette-lifecycle)
- [YIU lifecycle](#yiu-lifecycle)
- [Snapback lifecycle](#snapback-lifecycle)
- [Scar-nick lifecycle](#scar-nick-lifecycle)
- [Study lifecycle](#study-lifecycle)
- [Portfolio lifecycle](#portfolio-lifecycle)
- [Layers and responsibilities](#layers-and-responsibilities)
- [On-disk layout](#on-disk-layout)
- [Run artifacts](#run-artifacts)
- [Cassette artifacts](#cassette-artifacts)
- [YIU artifacts](#yiu-artifacts)
- [Scar-nick artifacts](#scar-nick-artifacts)
- [Study artifacts](#study-artifacts)
- [Portfolio artifacts](#portfolio-artifacts)
- [Reproducibility boundaries](#reproducibility-boundaries)
- [Extensibility points](#extensibility-points)
- [Related docs](#related-docs)

This doc describes the Cruncher run lifecycle, module boundaries, and on-disk artifacts.

#### Workflow families

Registered family ids: `sample`, `cassette`, `yiu`, `snapback`, `scar_nick`, `study`, `portfolio`

Cruncher is organized as seven peer workflow families, not one monolithic run shape:

- **Sample workspaces** (fixed-length optimization) use `fetch -> lock -> parse -> sample -> analyze -> export` and publish sequence-centric run artifacts for downstream operators.
- **Cassette workspaces** use `cassette init-workspace|validate|design|solve|show` and publish cassette-specific artifacts plus optional baserender job files.
- **YIU workspaces** use `yiu init-workspace|validate|render|show` and publish one payload bundle with three BaseRender-ready views.
- **Snapback workspaces** use `snapback init-workspace|validate|design|target-search|solve|show` plus the released-product `released-design|released-target-search|released-solve|released-show` lane and publish snapback-specific reports, candidate tables, and QA views.
- **Scar-nick workspaces** use `scar-nick validate|design|show` and publish retained-scar terminal-nick candidate tables, geometry audits, and BaseRender job handoffs.
- **Study workspaces** use `study list|run|summarize|show` to orchestrate sweep-owned manifests, trial tables, and aggregate plots over explicit source-family artifacts without redefining those source contracts.
- **Portfolio workspaces** use `portfolio run|show` to aggregate explicit source runs into one handoff package without implicit latest-run fallback.

These families deliberately keep separate workspace contracts, output trees, and orchestration seams. New families should add their own lane-specific artifacts rather than overload `sample`, `cassette`, `yiu`, `snapback`, `scar_nick`, `study`, or `portfolio`.

#### Fixed-length optimization lifecycle

1. **fetch** -> cache motifs/sites and update `catalog.json`
2. **lock** -> resolve TFs to exact cached artifacts (`<workspace>/.cruncher/locks/<config>.lock.json`)
3. **parse** *(optional)* -> validate locked PWMs and refresh the parse cache in workspace state (no logo rendering)
4. **sample** -> run MCMC and write sequences/trace + manifests
5. **analyze** -> curated `plots/*` and `tables/table__*` artifacts + report from sample artifacts (offline, written into the run directory)
6. **export** -> sequence-centric contract tables for wrappers/operators (`cruncher export sequences`)

#### Cassette lifecycle

The cassette workflow is a peer lane, not a variant of `sample`:

1. optional **cassette init-workspace** -> scaffold a runbook-only cassette workspace with shipped solve profiles
2. author `<workspace>/configs/cassettes/<name>.cassette.yaml` or `<workspace>/configs/cassettes/<name>.cassette.solve.yaml`
3. author or select a local nickase catalog (for example `<workspace>/inputs/nickases/*.yaml`) or use a built-in solve preset
4. **cassette validate** -> strict schema + invariant check plus deterministic planning report
5. **cassette design** -> write cassette-specific manifest, status, report, provenance snapshots, views, and optional baserender jobs
6. **cassette solve** -> bounded search, selected-hit materialization, shared view publication, and solve-level/per-hit baserender jobs
7. **cassette show** -> inspect status and artifact paths for one explicit cassette run

This workflow does not currently use `core/evaluator.py`, `gibbs_anneal`, `study`/`portfolio` orchestration, or workspace `run_index.json`.

#### YIU lifecycle

The YIU workflow is a peer lane, not a cassette submode:

1. optional **yiu init-workspace** -> scaffold a runbook-only YIU workspace with one payload example spec
2. author `<workspace>/configs/yiu/<name>.yiu.yaml`
3. **yiu validate** -> strict schema + payload normalization check under `split_yiu_payload_rendering_v4`
4. **yiu render** -> normalize the payload, exhaustively optimize the junction/mismatch plan, write the payload bundle, and optionally render the three payload views
5. **yiu show** -> inspect the bundle contract, provenance, selected junction window, PWM state, and available renders for one YIU bundle

This workflow does not use `sample`, `gibbs_anneal`, `run_index.json`, cassette-specific render contracts, or any legacy state graph.

#### Snapback lifecycle

The snapback workflow is a peer lane, not a cassette or YIU submode:

1. optional **snapback init-workspace** -> scaffold a runbook-only snapback workspace with one explicit v2 example and one solve example
2. author `<workspace>/configs/snapback/<name>.snapback.yaml`, `<workspace>/configs/snapback/<name>.snapback.solve.yaml`, or `<workspace>/configs/snapback/<name>.released.snapback.yaml`
3. author or select local nickase and release-enzyme catalogs or use preset-only resolved catalogs
4. **snapback validate** -> strict schema + invariant check plus deterministic explicit report
5. **snapback design** -> materialize one preserved-site explicit candidate bundle with reports, provenance snapshots, candidate table, and QA views
6. **snapback released-design** -> project a two-stage precursor into a retained top plus exposed-bottom post-release product and materialize one released-product bundle
7. **snapback solve** -> bounded preserved-site search, deterministic hit ranking, selected-hit materialization, and solve-level reports
8. **snapback released-target-search** -> target-first paired nickase plus release-enzyme search in exposed-bottom geometry space
9. **snapback show** / **snapback released-show** -> inspect the appropriate bundle type and fail fast on drift without guessing

This workflow does not use `sample`, `gibbs_anneal`, `run_index.json`, cassette baserender jobs, or YIU payload render contracts.

For operator-facing usage, start with [`../guides/snapback_workflow.md`](../guides/snapback_workflow.md) and [`../guides/snapback_released_workflow.md`](../guides/snapback_released_workflow.md). For the file-by-file output contracts, use [`snapback_artifacts.md`](snapback_artifacts.md) and [`released_snapback_artifacts.md`](released_snapback_artifacts.md).

#### Scar-nick lifecycle

The scar-nick workflow is a peer lane, not a Snapback or YIU submode:

1. author `<workspace>/configs/scar_nick/<name>.scar_nick.yaml`
2. **scar-nick validate** -> strict schema, release-enzyme, nickase, and pair-profile checks
3. **scar-nick design** -> deterministic run bundle with candidate tables, geometry audit, provenance snapshots, QA views, and BaseRender jobs
4. **scar-nick show** -> inspect one explicit run and fail fast on missing or drifted artifacts

This workflow does not solve cap geometry, assemble Retron MSD products, run
YIU payload rendering, or create one workspace per hit.

For operator-facing usage, start with [`../guides/scar_nick_workflow.md`](../guides/scar_nick_workflow.md) and the checked-in [`scar_nick_teto` runbook](../../workspaces/scar_nick_teto/runbook.md). For module ownership, use the [`scar_nick` package map](../../src/scar_nick/README.md).

#### Study lifecycle

The study workflow is a peer family that orchestrates source-family artifacts rather than redefining their geometry:

1. **study list** -> inspect checked-in study specs under `<workspace>/configs/studies/` before choosing a run surface
2. author `<workspace>/configs/studies/<name>.study.yaml`
3. **study run** -> freeze the study spec, expand explicit trials and/or `trial_grids`, and write deterministic study/trial outputs under `outputs/studies/`
4. **study summarize** -> compute aggregate tables, selection summaries, and replay-aware study status from the recorded trial artifacts
5. **study show** -> inspect deterministic study identity, trial inventory, aggregate outputs, and source-run provenance

This workflow does not rewrite source manifests, append trial sampling to workspace `run_index.json`, or hide trial/source provenance behind "latest run" fallbacks.

#### Portfolio lifecycle

The portfolio workflow is a peer family that packages explicit source runs into one handoff bundle:

1. author `<portfolio_workspace>/configs/<name>.portfolio.yaml`
2. **portfolio run** -> resolve explicit source runs, aggregate handoff tables/plots, and write one portfolio-scoped output package
3. **portfolio show** -> inspect deterministic portfolio identity, source provenance, and handoff artifact paths

This workflow does not infer source runs implicitly, mutate source workspaces, or collapse portfolio provenance into `sample` run metadata.

---

#### Layers and responsibilities

Core contract:

- **Network access is explicit** (fetch and remote inventory).
- The **store** is the only persistence layer (project-local).
- The **core** (PWM scoring + optimizers) is pure compute (no I/O).
- **Analyze** reads run artifacts only and can run offline.

#### `core/` (pure compute)
- PWM representation and validation
- scoring / evaluator logic
- sequence state and move operators
- optimizer kernels (gibbs annealing)
- No I/O (no filesystem, no network)

#### `ingest/` (ports/adapters)
- source adapters (RegulonDB first)
- normalization into standard records (motifs + sites)
- optional hydration (coordinates -> sequences) via genome providers

#### `store/` (local persistence)
- the on-disk catalog (what's cached)
- lockfiles (what's pinned)
- run index (what runs exist)

#### `analysis/` (analysis + diagnostics)
- plot registry, per-PWM summaries, and analysis helpers
- plot implementations live under `analysis/plots/`
- baserender-backed elites showcase lives in `analysis/plots/elites_showcase.py`
- trajectory score-space elite mapping/sampling helpers live in `analysis/plots/trajectory_score_space.py`
- trajectory score-space panel rendering helpers live in `analysis/plots/trajectory_score_space_panel.py`
- trajectory score-space plot orchestration lives in `analysis/plots/trajectory_score_space_plot.py`
- chain-trajectory video orchestration lives in `analysis/trajectory_video.py`
- trajectory frame/timeline selection helpers live in `analysis/trajectory_video_timeline.py`
- baserender video contract assembly lives in `analysis/trajectory_video_contract.py`

#### `artifacts/` (run layout + manifests)
- run directory layout + status helpers
- manifest + artifact bookkeeping utilities

#### `cassette/` (dual-context cassette domain)
- cassette spec and nickase catalog schemas
- workspace-relative loading and path validation
- deterministic nick-site scanning and bounded-segment planning
- cassette-specific artifact helpers
- no dependency on legacy `sample` optimizer contracts

#### `yiu/` (payload-centric YIU domain)
- YIU spec schema for `split_yiu_payload_rendering_v4`, exposed through the stable `yiu/spec_models.py` facade
- focused input, PWM, and rendering validators live in `yiu/spec_input_models.py`, `yiu/spec_pwm_models.py`, and `yiu/spec_rendering_models.py`
- payload normalization for `user_sequence` and `sample_hit`
- public input-resolution orchestration lives in `yiu/payload_resolution.py`, while sample-hit artifact IO stays isolated in `yiu/sample_hit_sources.py`
- public PWM-resolution orchestration lives in `yiu/pwm_context.py`
- inline/file PWM source dispatch lives in `yiu/pwm_context_sources.py`
- sample-backed PWM-context orchestration lives in `yiu/pwm_context_sample_context.py`
- selected-occurrence parquet loading lives in `yiu/pwm_context_sample_occurrences.py`
- sample-backed motif-instance materialization lives in `yiu/pwm_context_sample_motifs.py`
- exhaustive optimization, split, display-orientation, and junction derivation
- shared view fragments live in `yiu/view_common.py`
- shared manifest/inventory/normalized load-persist helpers live in `yiu/bundle_state.py`
- shared typed render/show bundle-artifact surfaces for app/CLI boundaries live in `yiu/bundle_surface.py`
- `yiu/render.py` is the thin public render facade
- render preflight, runtime loading, and transactional plan assembly live in `yiu/render_plan.py`
- panel execution and output publication live in `yiu/render_execution.py`
- failure-state persistence and cleanup of partial render artifacts live in `yiu/render_state.py`
- `app/yiu_workflow/render.py` owns bundle publication orchestration
- `app/yiu_workflow/show.py` owns read-only inspection and drift checking over the shared bundle surfaces
- payload bundle publication orchestration lives in `yiu/publish.py`
- payload bundle filesystem writes and debug-job emission live in `yiu/publish_io.py`
- bundle layout and artifact-path planning live in `yiu/publish_layout.py`
- view-entry/render-job planning lives in `yiu/view_catalog.py`
- normalized-payload, inventory, and manifest assembly lives in `yiu/publish_inventory.py`
- display-title policy lives in `yiu/view_styles.py`
- producer-owned YIU visual foundations live in `yiu/visual_foundations.py`
- named visual-direction deltas live in `yiu/visual_directions.py`
- view registry and style profiles live in `yiu/visual_system.py`
- the named YIU visual system is `bench_strip`, with `evidence_ribbon` for payload truth and `operator_strip` for assembly-oriented views
- payload mismatch/motif/meta shaping lives in `yiu/view_payload_content.py`
- payload-view contract shells live in `yiu/view_payload_contracts.py`
- split/assembled sequence-contract assembly lives in `yiu/view_sequence_contracts.py`
- split sticky-end and assembled junction metadata policy lives in `yiu/view_sequence_metadata.py`
- bundle-path invariants live in `yiu/bundle_paths.py`
- panel render/load/save helpers live in `yiu/render_panels.py`
- no dependency on legacy `sample` or cassette-specific planner contracts

#### `snapback/` (single-nick foldback domain)
- snapback explicit `single_nick_snapback_v2` and co-design solve `single_nick_snapback_solve_v3` contracts
- authored top-strand nick-relative geometry and bounded search
- protected-region, homology, and extra-nick invariant enforcement
- preserved-site target search orchestration is split by ownership under `snapback/preserved_search/`
- `snapback/target_search.py` is the thin public facade; placement enumeration, candidate construction, ranking, and report shaping now live in `preserved_search/{placements.py,candidate_builder.py,ranking.py,reporting.py,runner.py}`
- released explicit origin inference and typed report shaping live in `snapback/released_explicit_evaluation.py`, keeping `app/snapback_released_workflow.py` on orchestration and publication
- released-product search orchestration is split by ownership under `snapback/released_search/`
- `snapback/released_target_search.py` is the thin public facade; placement generation, precursor construction, pair evaluation, ranking, and report shaping now live in `released_search/{nick_placements.py,release_placements.py,precursor_builder.py,evaluator_adapter.py,ranking.py,reporting.py,runner.py}`
- released precursor projection is split by ownership under `snapback/released_projection_{common,projection,candidate,selection}.py`; `snapback/released_projection.py` is the thin public facade and `snapback/released_models.py` is a model facade, not a legacy compatibility layer
- typed producer-owned QA view models live in `snapback/view_models.py`
- local view-contract assembly and validation live in `snapback/view_contracts.py`
- no dependency on legacy `sample`, cassette render contracts, or YIU payload semantics

#### `viz/` (plotting)
- matplotlib/logomaker setup
- PWM logo rendering + visualization helpers

#### `integrations/` (external tools)
- wrappers for external binaries (e.g., MEME Suite)

#### `app/` (orchestration)
- fetch / lock / parse / sample / analyze coordination
- cassette workflow coordination in `app/cassette_workflow.py`
- study coordination (`study list|run|summarize|show`)
- portfolio coordination (`portfolio run|show`)
- translates CLI intent + config into concrete runs and artifacts
- analyze orchestration is split by concern:
  - run-level assembly/state transitions in `app/analyze_workflow.py`
  - run-level metadata/artifact context resolution in `app/analyze/execution.py`
  - run-level compute/score-space context resolution in `app/analyze/run_context.py`
  - table/metric computation + persistence in `app/analyze/computation.py`
- score-space projection helpers in `app/analyze_score_space.py`
- plot orchestration surface in `app/analyze/plotting.py`
- run-level plot render orchestration in `app/analyze/rendering.py`
- lazy plot callable resolution in `app/analyze/plot_resolver.py`
- plot artifact bookkeeping in `app/analyze/plotting_registry.py`
- trajectory plot/video render paths in `app/analyze/plotting_trajectory.py`
- static and FIMO plot render paths in `app/analyze/plotting_static.py`
- report/manifest/summary publication in `app/analyze/publish.py`
- snapback CLI request construction lives in `app/snapback_cli_requests.py`, which owns typed preserved-site and released-product search/solve invocation factories
- preserved-site snapback catalog loading and source-label normalization shared by explicit validate/design and solve now live in `app/snapback_catalogs.py`
- preserved-site show/readback stays behind `app/snapback_workflow.py`, with bundle loading, drift validation, and payload assembly split across `app/snapback_show_{load,validate,present}.py`
- preserved-site solve stays behind `app/snapback_solve_workflow.py`, with per-hit explicit-spec snapshot assembly, materialization, and report shaping split across `app/snapback_solve_{snapshot,materialize,reporting}.py`
- released-product catalog loading and source-label normalization shared by explicit validate/design, target-search, and solve now live in `app/snapback_released_catalogs.py`
- released-product show/readback stays behind `app/snapback_released_show.py`, with artifact loading, drift validation, and payload assembly split across `app/snapback_released_show_{load,validate,present}.py`
- released-product solve stays behind `app/snapback_released_solve_workflow.py`, with request snapshot assembly, per-hit materialization, and report shaping split across `app/snapback_released_solve_{snapshot,materialize,reporting}.py`

#### `cli/` (UX only)
- Typer commands
- argument parsing, output formatting
- `cli/commands/snapback.py` is command registration only
- snapback command handlers live in `cli/commands/snapback_{workspace,explicit,released,show}.py`
- shared report/path presentation lives in `cli/commands/snapback_presenters.py`
- delegates work to app modules and request factories (no business logic)

#### Baserender integration boundary

Cruncher integrates with baserender through the **public package root only**:

- Allowed: `from dnadesign.baserender import ...`
- Disallowed: `dnadesign.baserender.src.*` deep imports

Current Cruncher handoff for `elites_showcase.*` and `chain_trajectory_video.mp4`:

1. Cruncher resolves run data into rendering primitives:
   - sequence per elite
   - best-window spans/strand per TF
   - locked motif matrices for each TF
2. Cruncher hands baserender only the minimal plotting contract:
   - record-shaped rows (`id`, `sequence`, `features`, `effects`, `display`) and motif primitives
   - or equivalent in-memory `Record` objects through baserender public APIs
3. Baserender validates contracts, performs layout/rendering, and emits assets.

For payload-centric YIU, Cruncher publishes shared `yiu_payload_visual_v1` contracts only. Downstream, baserender keeps the adapter split explicit: `yiu_payload_visual_v1.py` owns public adapter orchestration, `yiu_payload_sequence_projection.py` owns sequence-evidence projection, and `yiu_payload_motif_overlay.py` owns payload motif `Feature`/`Effect` assembly.

For `chain_trajectory_video.mp4`, Cruncher first resolves selected-chain trajectory rows and sampled frame indices, then writes temporary record rows and passes a strict sequence-rows video job contract to baserender.

For cassette runs, Cruncher now publishes shared, file-based visual contracts rather than a Cruncher-owned render payload:

* `views/linear_duplex.v1.json` for the duplex interpretation
* `views/ssdna_hairpin.v1.json` for the folded hairpin interpretation
* `views/views_manifest.v1.json` for grouping and discovery
* sibling `baserender_jobs/*.job.yaml` files that reference those contracts by path

Cassette runs do not call baserender directly; they publish the contracts and jobs for downstream consumers and fail fast on schema violations before write-out.

For snapback runs, Cruncher publishes both producer-owned QA views and shared evidence-map contracts:

* `analysis/views/pre_nick_duplex.v1.json`, `analysis/views/post_nick_exposed.v1.json`, `analysis/views/post_nick_foldback.v1.json`
* `analysis/views/pre_nick_duplex.snapback_visual.v1.json`, `analysis/views/post_nick_exposed.snapback_visual.v1.json`, `analysis/views/post_nick_foldback.snapback_visual.v1.json`
* `analysis/views/snapback_triptych.snapback_visual.v1.jsonl`
* `analysis/views/views_manifest.v1.json`
* sibling `baserender_jobs/snapback_triptych.job.yaml` that references the triptych evidence-map rows by path

These views are a topology and coordinate QA surface, not a biophysical rendering claim. The producer-owned JSON captures snapback-specific semantics such as a nick-anchored retained stem, source-side cap, effective `3 nt` cap loop, active-strand foldback geometry, and the shared nick/origin boundary in the pre/exposed states. The shared `snapback_visual_v1` contracts carry the nucleotide-resolution publication surface for downstream BaseRender rendering.

The showcase/video renderers do not require overlap tables; overlap metrics remain separate analysis artifacts.

This keeps responsibilities decoupled:
- Cruncher owns analysis semantics and motif provenance.
- Baserender owns rendering contracts, geometry, and output encoding.
- Both sides fail fast on schema/contract violations.

---

#### On-disk layout

**cruncher** uses **project-local state** (relative to the workspace root resolved from config). Data artifacts live in the workspace;
tooling caches stay within the repo/workspace unless you override their environment variables.

Recommended workspace layout:

```
<workspace>/
configs/
  config.yaml
  cassettes/             # optional cassette specs
  yiu/                   # optional YIU specs
  studies/               # optional study specs
  portfolios/            # optional portfolio specs
inputs/
  nickases/              # optional local nickase catalogs
.cruncher/
outputs/
```

In this repo, the bundled demo workspaces live at:

- `src/dnadesign/cruncher/workspaces/demo_pairwise/`
- `src/dnadesign/cruncher/workspaces/demo_multitf/`
- `src/dnadesign/cruncher/workspaces/project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs/`

#### Catalog root (`catalog.root`, default: `.cruncher/`)

```
<catalog.root>/
catalog.json
normalized/
motifs/<source>/<motif_id>.json
sites/<source>/<motif_id>.jsonl
genomes/              # if genome hydration is enabled
discoveries/          # MEME/STREME discovery runs
```

- `catalog.json` is the "what do we have cached?" index.
- `catalog.root` can be absolute or relative to the workspace root; relative paths must not include `..`.
- By default the catalog cache is workspace-local (`<workspace>/.cruncher`); locks/run_index also live in each workspace's `.cruncher/`.

#### Workspace state (per workspace `.cruncher/`)

```
<workspace>/.cruncher/
locks/<config>.lock.json
run_index.json
parse/inputs/{lockfile.json,parse_manifest.json,pwm_summary.json}
```

- `locks/<config>.lock.json` pins TF names -> exact cached artifacts + hashes.
- `run_index.json` tracks run folders for `cruncher runs ...` within that workspace.
- `parse/` stores parse-stage validation artifacts outside user-facing sample outputs.

#### Tooling caches

- Matplotlib writes its cache under `.cache/matplotlib/cruncher` unless `MPLCONFIGDIR` is set.
- Numba JIT cache defaults to `<workspace>/.cruncher/numba_cache` unless `NUMBA_CACHE_DIR` is set.

#### Run outputs (`out_dir`, e.g. `outputs/`)

Each regulator set gets one standard run directory:

- single regulator set: `<workspace>/<out_dir>/`
- multiple regulator sets: `<workspace>/<out_dir>/setN_<tf-slug>/`

Within each run directory, Cruncher uses a stable, stage-agnostic subdirectory layout:

```
<run_dir>/
  meta/
  provenance/
  optimize/
  analysis/
  plots/
  export/
```

Cassette runs use a separate deterministic output root:

```
<workspace>/outputs/cassettes/<spec.name>/<design_id>/
```

YIU runs use one family-rooted deterministic payload bundle root:

```
<workspace>/outputs/<spec.name>/
```

---

#### Run artifacts

A typical **sample** run directory contains:

- `meta/run_manifest.json`, `meta/run_status.json`, `meta/config_used.yaml` - run metadata + status
- `provenance/lockfile.json` - pinned input snapshot (reproducibility boundary)
- `optimize/tables/sequences.parquet`, `optimize/tables/elites*`, `optimize/tables/random_baseline*` - sampling tables (`random_baseline*` defaults on with `sample.output.save_random_baseline=true`, `sample.output.random_baseline_n=10000`)
- `optimize/state/trace.nc`, `optimize/state/metrics.jsonl`, `optimize/state/elites.{json,yaml}` - sampling metadata
- `analysis/reports/summary.json` - standard analysis summary
- `analysis/reports/report.json` + `analysis/reports/report.md` - analysis report outputs from `cruncher analyze`
- `analysis/manifests/plot_manifest.json` + `analysis/manifests/table_manifest.json` + `analysis/manifests/manifest.json` - analysis inventories
- `export/table__elites.csv` + `export/table__*.{parquet|csv}` + `export/export_manifest.json` - sequence-export tables from `cruncher export sequences`
- `plots/*` - curated analysis plots and catalog logo renders
- `plots/chain_trajectory_video.mp4` - optional trajectory-video artifact (`analysis.trajectory_video.enabled=true`)
- `analysis/tables/table__*` - curated table outputs

#### Cassette artifacts

A typical **cassette** run directory contains:

- `meta/cassette_manifest.json`, `meta/cassette_status.json` - cassette-stage metadata + status
- `provenance/spec_used.yaml`, `provenance/nickase_catalog.yaml` - frozen input snapshots
- `analysis/reports/report.json`, `analysis/reports/report.md` - deterministic planning report
- `export/table__candidates.csv` - candidate table (one row for the satisfied candidate when present)
- `views/linear_duplex.v1.json`, `views/ssdna_hairpin.v1.json`, `views/views_manifest.v1.json` - shared file-based visual contracts
- `baserender_jobs/linear_duplex.job.yaml`, `baserender_jobs/ssdna_hairpin.job.yaml` - optional downstream baserender jobs

Cassette runs are intentionally isolated from `sample` runs:

- they do not write `meta/run_manifest.json`
- they do not append to workspace `run_index.json`
- they do not share `optimize/`, `plots/`, or `analysis/tables/` sample-stage contracts

---

#### YIU artifacts

A typical **YIU** run directory contains:

- `bundle_summary.json` - operator-facing 5' to 3' run summary with one `views` block for payload, split-left, split-right, and assembled reference-vs-mismatch-present rows
- `bundle_manifest.json` - machine-facing payload bundle metadata under `split_yiu_payload_bundle_v4`
- `normalized_payload.json` - normalized semantic payload object for validation and debug
- `visual_inventory.json` - machine-facing bundle-local visual inventory and render-truth index
- `payload_view.json` - pure payload contract with optional PWM motif layers
- `split_payload_view.jsonl` - JSONL split payload contract rows (`split_payload_left`, then `split_payload_right`)
- `assembled_payload_view.json` - rejoined payload machine contract in original payload order with one explicit `junction_span`
- `payload_views.pdf` - operator-facing composite render listed in `visual_inventory.json`
- `baserender_jobs/*.job.yaml` - optional debug-only jobs when explicitly requested

YIU runs are intentionally isolated from both `sample` and `cassette` runs:

- they do not write `meta/run_manifest.json`
- they do not append to workspace `run_index.json`
- they do not reuse cassette-specific `views/*.v1.json` render contracts

---

#### Snapback artifacts

A typical **snapback explicit** run directory at `<workspace>/outputs/design/` contains:

- `meta/snapback_manifest.json`, `meta/snapback_status.json` - explicit-run metadata + status
- `provenance/spec_used.yaml`, `provenance/nickase_catalog.yaml` - frozen input snapshots
- `analysis/reports/report.json`, `analysis/reports/report.md` - deterministic explicit report
- `export/table__candidates.csv` - candidate table with one row for the explicit candidate when satisfied
- `analysis/views/pre_nick_duplex.v1.json`, `analysis/views/post_nick_exposed.v1.json`, `analysis/views/post_nick_foldback.v1.json` - producer-owned QA views
- `analysis/views/*.snapback_visual.v1.json` - shared snapback visual contracts for the QA triptych
- `analysis/views/snapback_triptych.snapback_visual.v1.jsonl` - ordered shared triptych contracts for one composite render
- `analysis/views/views_manifest.v1.json` - grouped view inventory plus recommended render jobs
- `baserender_jobs/snapback_triptych.job.yaml` - optional downstream render job for the composite QA triptych
- `plots/snapback_triptych.<png|svg|pdf>` - optional rendered QA triptych after downstream BaseRender execution

A typical **snapback solve** run directory at `<workspace>/outputs/solve/` contains:

- `meta/solve_manifest.json`, `meta/solve_status.json` - solve metadata + status
- `provenance/input_solve_spec.yaml`, `provenance/resolved_catalog.yaml` - frozen solve inputs
- `analysis/reports/solve_report.json`, `analysis/reports/solve_report.md` - solve-level report outputs
- `export/table__hits.csv`, `export/table__frontier.csv` - ranked hit table plus compactness frontier
- `analysis/materialized_hits/hit_<rank>/...` - materialized explicit hit bundles under stable rank paths

Snapback runs are intentionally isolated from `sample`, `cassette`, and `yiu` runs:

- they do not write `meta/run_manifest.json`
- they do not append to workspace `run_index.json`
- they do not invoke baserender directly
- they do not reuse cassette or YIU view contracts

---

#### Scar-nick artifacts

A typical **scar-nick** run directory at `<workspace>/outputs/scar_nick/<name>/` contains:

- `meta/scar_nick_manifest.json`, `meta/scar_nick_status.json` - run metadata and status
- `provenance/spec.snapshot.yaml` - frozen spec snapshot
- `analysis/reports/report.md` - deterministic operator report
- `export/table__scar_nick_candidates.csv` - accepted and reserve candidate table
- `export/table__scar_nick_candidate_pair_calls.csv` - per-S-site pair calls
- `export/table__scar_nick_nickase_geometry_audit.csv` - release/nickase geometry audit rows
- `analysis/views/*.scar_nick_visual.v1.json` - producer-owned terminal-nick visual contracts
- `baserender_jobs/scar_nick_terminal_nick.job.yaml` - optional downstream BaseRender job

Scar-nick runs are intentionally isolated from `sample`, `cassette`, `yiu`, and
`snapback` runs:

- they do not write `meta/run_manifest.json`
- they do not append to workspace `run_index.json`
- they do not solve Snapback cap geometry
- they do not assemble Retron MSD products

---

#### Study artifacts

Study runs are aggregate sweep workflows that keep deterministic workspace config separate from sweep intent.

Study specs live under:

```
<workspace>/configs/studies/<name>.study.yaml
```

Study outputs live under:

```
<workspace>/outputs/studies/<study.name>/<study_id>/
  study/
  trials/
  tables/
  manifests/
```

Key points:

- `study_id` is deterministic from frozen spec + base config hash + target descriptor.
- Study specs support both explicit trial lists and cartesian grid expansion (`trial_grids`).
- Trial outputs are nested under `trials/<trial_id>/seed_<seed>/`.
- `tables/` stores aggregate sweep tables (`table__trial_metrics*`, optional `table__mmr_tradeoff_agg`).
- Aggregate study plots are workspace-flat under `outputs/plots/` as namespaced files
  (`study__<study_name>__<study_id>__plot__*`).
- Study trial sampling does **not** write to workspace `run_index.json`.
- Replay sweeps reuse saved run artifacts instead of re-sampling where possible (MMR selection replay).

---

#### Portfolio artifacts

Portfolio runs aggregate selected source runs across workspaces into one handoff package.

Portfolio specs live under:

```
<portfolio_workspace>/configs/<name>.portfolio.yaml
```

Portfolio outputs live under:

```
<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/
  meta/
  tables/
  plots/
```

Key points:

- `portfolio_id` is deterministic from the frozen Portfolio spec payload.
- Source run selection is explicit in spec (`workspace`, `run_dir` per source).
- Source elite count is contract-driven from source run manifest `top_k` and `elites.parquet`.
- Portfolio plots are run-scoped under
  `<portfolio_run_dir>/plots/plot__*`.
- No implicit latest-run fallback is used during portfolio aggregation.
- Handoff tables are source-provenance-first (`table__handoff_windows_long`, `table__handoff_elites_summary`, `table__source_summary`).

---

#### Reproducibility boundaries

- **Lockfiles are mandatory** for `parse` and `sample`.
- If you change inputs that affect TF resolution (e.g., PWM source, site filters, dataset selection),
  **re-lock** so the lockfile hash set matches reality.
- `analyze` validates the lockfile recorded in the run manifest.

---

#### Extensibility points

- **Sources:** add a new adapter under `ingest/adapters/` and register it in the source registry.
- **Local sources:** configure `ingest.local_sources` for local motif directories (no new code required).
- **Parsers:** add a parser under `io/parsers/` or register via `io.parsers.extra_modules`.
- **Optimizers:** add a new kernel and register it in `core/optimizers/registry.py`.
- **Analysis plots:** add a plot implementation and register it in the analysis plot registry.

---

#### Related docs

- [Intent + lifecycle](../guides/intent_and_lifecycle.md)
- [Snapback workflow](../guides/snapback_workflow.md)
- [Scar-nick workflow](../guides/scar_nick_workflow.md)
- [Portfolio aggregation](../guides/portfolio_aggregation.md)
- [Config reference](config.md)
- [Snapback artifacts](snapback_artifacts.md)
