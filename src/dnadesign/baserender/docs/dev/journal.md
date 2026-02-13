# Baserender Dev Journal

## 2026-02-12 - Critical audit for Cruncher substrate refactor

### Critical Audit (Baserender as Cruncher-plot substrate)

1. Critical: baserender is not fully tool-agnostic yet due to sigma70 logic in core rendering path.
   `src/dnadesign/baserender/src/layout.py:85` reserves forward track 0 for sigma tags, `src/dnadesign/baserender/src/palette.py:30` canonicalizes `tf:sigma70_* -> sigma`, and `src/dnadesign/baserender/src/legend.py:18` hard-codes sigma legend semantics.
   This is densegen-specific behavior in core, not plugin-only.

2. High: output.video is implicitly enabled when omitted (UX footgun).
   In `src/dnadesign/baserender/src/config/job_v2.py:368`, `video_raw = m.get("video", {})` means omission still builds default MP4 output.
   Result: images-only configs silently also render video unless user sets `video: null`.

3. High: stale/legacy config surface still present in repo examples.
   `src/dnadesign/baserender/jobs/scratch.yml:1` is v1-style (no `version: 2`, top-level `plugins`, legacy fields), while CLI still supports v1 via `--allow-v1` (`src/dnadesign/baserender/src/cli.py:189`).
   This increases cognitive load and weakens contract clarity.

4. Medium: densegen naming is still the default public surface.
   Defaults in API/CLI still point to `densegen__used_tfbs_detail` (`src/dnadesign/baserender/src/api.py:46`, `src/dnadesign/baserender/src/cli.py:524`), and README contract uses densegen-specific naming (`src/dnadesign/baserender/README.md:65`).

5. Medium: unknown guide kinds are silently ignored.
   `src/dnadesign/baserender/src/render.py:383` intentionally drops unknown guide types. This is extensible, but it can hide typos in guide/effect names.

6. Strong positive: the schema and row parsing are already strict and fail-fast where it matters.
   `src/dnadesign/baserender/src/config/job_v2.py:465` rejects unknown keys; `src/dnadesign/baserender/src/model.py:90` asserts strand/label sequence correctness; parser has explicit ambiguity/missing-kmer policies (`src/dnadesign/baserender/src/io/parquet.py:158`).

---

### Feasibility verdict for Cruncher integration: YES

Validated a direct Cruncher->Baserender mapping using current artifacts:

- `cruncher optimize/elites.parquet`
- `cruncher optimize/elites_hits.parquet`

Converted these to Baserender records and rendered successfully (4/4 images), including forward/reverse best-window placement.

Key Cruncher fields already available:

- best-window start/strand/sequence: `src/dnadesign/cruncher/src/core/scoring.py:307`
- elite hit export: `src/dnadesign/cruncher/src/app/sample/run_set.py:843`

Conclusion: base track/kmer rendering is already compatible.

---

### Pragmatic integration path (pre-implementation)

1. Keep Baserender core generic and move sigma-specific behavior out of core into plugin/effect logic.
2. Add a generic annotation effect payload (example: `motif_logo`) and render it in Baserender as an overlay on the kmer box.
3. In Cruncher analyze, add an adapter that builds `SeqRecord` from elites + hits + PWM matrix and writes a new plot artifact (elites only).
4. Use information-content logo semantics for overlay: faded stack for motif probabilities, opaque highlight for observed base at each position.

---

### Planned execution phases

- Phase 1: core decoupling + schema cleanup in Baserender.
- Phase 2: new Cruncher plot wired into `analyze` and `plot_registry`.

## 2026-02-12 - Architecture hardening audit (post-vNext tree consolidation)

### Findings (ordered by severity)

1. High: external integrations are coupled to internal modules, not a stable public API.
   - `src/dnadesign/baserender/__init__.py:12` exports only `run_job_v3` and `validate_job`.
   - OPAL imports deep internals (`dnadesign.baserender.src.adapters...`, `...src.render...`) at:
     - `src/dnadesign/opal/notebooks/prom60_eda.py:2851`
     - `src/dnadesign/opal/notebooks/prom60_eda.py:2980`
   - Impact: internal refactors break callers; extension requires internal knowledge instead of a contracted API.

2. High: effect payload contracts are not strict on unknown keys (violates no-fallback intent).
   - `src/dnadesign/baserender/src/core/registry.py:63` and `src/dnadesign/baserender/src/core/registry.py:92` validate required fields but do not reject extra keys in `effect.target` / `effect.params`.
   - Drawers consume only subsets:
     - `src/dnadesign/baserender/src/render/effects/span_link.py:73`
     - `src/dnadesign/baserender/src/render/effects/span_link.py:88`
     - `src/dnadesign/baserender/src/render/effects/motif_logo.py:61`
   - Impact: typos in effect params can be silently ignored.

3. High: default IO scope is package-root, not caller/workspace-root for non-workspace runs.
   - `src/dnadesign/baserender/src/config/job_v3.py:152` defaults outputs to `<baserender_root>/results`.
   - `src/dnadesign/baserender/src/workspace.py:38` defaults workspaces to `<baserender_root>/workspaces`.
   - Impact: when called from sibling tools (Cruncher/OPAL), outputs/workspaces are coupled to package location; brittle and can be unwritable in installed environments.

4. Medium: absolute path resolution skips existence checks for required inputs.
   - `src/dnadesign/baserender/src/config/job_v3.py:181` returns absolute paths directly.
   - Existence checks only happen for relative paths (`src/dnadesign/baserender/src/config/job_v3.py:184`).
   - Impact: invalid absolute `input.path` / `selection.path` / adapter path columns fail later at runtime, not at config boundary.

5. Medium: contract bootstrap is hidden import side-effect.
   - Builtin contracts are registered in `src/dnadesign/baserender/src/render/__init__.py:19`.
   - Registries start empty at `src/dnadesign/baserender/src/core/registry.py:107`.
   - Impact: contract behavior depends on import order (`render` imported vs not), reducing predictability for library consumers.

6. Medium: orchestrator is all-in-memory, limiting scalability and composability.
   - Rows are fully collected into a list in `src/dnadesign/baserender/src/api.py:58`.
   - Selection/render happen after collection (`src/dnadesign/baserender/src/api.py:83`).
   - Impact: memory growth on large datasets; harder to support streaming pipelines or alternate sinks.

7. Medium: `sigma70` transform can emit effect references to IDs it did not create.
   - Dedup skips feature creation:
     - `src/dnadesign/baserender/src/pipeline/sigma70.py:119`
     - `src/dnadesign/baserender/src/pipeline/sigma70.py:133`
   - Effect always targets synthetic IDs (`src/dnadesign/baserender/src/pipeline/sigma70.py:158`).
   - Impact: if equivalent features already exist with different IDs, effect target may be unresolved.

8. Low: CLI help text has stale default workspace path.
   - Help strings point to `.../src/workspaces` in:
     - `src/dnadesign/baserender/src/cli.py:50`
     - `src/dnadesign/baserender/src/cli.py:69`
     - `src/dnadesign/baserender/src/cli.py:91`
     - `src/dnadesign/baserender/src/cli.py:219`
     - `src/dnadesign/baserender/src/cli.py:245`
   - Resolver default is `.../baserender/workspaces` (`src/dnadesign/baserender/src/workspace.py:38`).

9. Low: information-architecture drift in file headers after move.
   - Header paths in outputs modules still reference old locations:
     - `src/dnadesign/baserender/src/outputs/images.py:4`
     - `src/dnadesign/baserender/src/outputs/video.py:4`
     - `src/dnadesign/baserender/src/outputs/__init__.py:4`

---

### What is working well

- Job schema strictness and unknown-key rejection are strong (`src/dnadesign/baserender/src/config/job_v3.py:600`).
- Adapter contracts are centralized and explicit (`src/dnadesign/baserender/src/config/adapter_contracts.py:43`).
- Core/render separation is mostly clean; unknown renderer/effect kinds fail fast:
  - `src/dnadesign/baserender/src/render/renderer.py:34`
  - `src/dnadesign/baserender/src/render/effects/registry.py:35`

---

### Open questions

1. Should baserender default IO root be caller/job scoped (for library usage) instead of package scoped?
2. Should effect kinds enforce strict schemas for `target` and `params` (reject unknown keys) globally?
3. Should OPAL/Cruncher consume only a stable `dnadesign.baserender` API surface, with internal modules treated as private?

---

### Recommended hardening sequence

1. Introduce a stable public integration API (record builders + render entrypoints) and remove deep internal imports in OPAL.
2. Add per-effect strict schema validators (`target` + `params`) with unknown-key rejection.
3. Move default output/workspace roots to caller/job context; keep package-local roots only when explicitly requested.
4. Make contract registration explicit at startup (no import side-effect bootstrap).
5. Refactor run pipeline to iterator/stream form for large runs, then layer image/video sinks on top.

## 2026-02-12 - Architecture hardening implementation pass

### Decisions resolved

1. IO default scope is caller/job scoped by default.
   - Non-workspace jobs now default `results_root` to `<caller_root>/results` (caller root defaults to current working directory).
   - Workspace jobs still default to `<workspace>/outputs`.
2. Effect contracts are strict globally.
   - `span_link` and `motif_logo` now reject unknown `target`/`params` keys.
3. Sibling tools should consume only public `dnadesign.baserender` API.
   - OPAL deep imports were replaced with a public helper.
   - A guard test now fails if sibling tools import `dnadesign.baserender.src.*`.

### Hardening changes completed

- Added explicit runtime bootstrap:
  - `src/dnadesign/baserender/src/runtime.py`
  - Removed contract/effect registration side effects from `src/dnadesign/baserender/src/render/__init__.py`
  - Added test reset hooks:
    - `clear_feature_effect_contracts` in `src/dnadesign/baserender/src/core/registry.py`
    - `clear_effect_drawers` in `src/dnadesign/baserender/src/render/effects/registry.py`
- Expanded stable public API surface:
  - `load_record_from_parquet`
  - `render_record_figure`
  - `render_parquet_record_figure`
  - `render_densegen_record_figure`
  - Exported via `src/dnadesign/baserender/src/__init__.py` and `src/dnadesign/baserender/__init__.py`
- Tightened effect contract validation:
  - `span_link.target` and `span_link.params` unknown-key rejection
  - `motif_logo.target` and `motif_logo.params` unknown-key rejection
- Fixed sigma70 link targeting with dedup:
  - `span_link` now references actual emitted or existing feature IDs.
  - Removed unused `strength` effect param from sigma-generated `span_link`.
- Tightened config path contracts:
  - Absolute required paths now validated at load boundary.
  - `load_job_v3(..., caller_root=...)` and `validate_job_v3(..., caller_root=...)` added.
- Updated workspace and CLI defaults:
  - `default_workspaces_root()` now resolves from current working directory.
  - CLI help text updated to reflect `<cwd>/workspaces`.
- Refactored job orchestration to stream:
  - Replaced all-in-memory record collection with iterator pipeline.
  - Added short-circuiting for `input.limit` and `input.sample.mode=first_n` when selection is disabled.
  - Materialization now happens only when required (selection/random sample/video sink).
- Migrated OPAL notebook integration to public API:
  - `src/dnadesign/opal/notebooks/prom60_eda.py` now uses `dnadesign.baserender.render_densegen_record_figure`.
- Corrected output module header path drift:
  - `src/dnadesign/baserender/src/outputs/__init__.py`
  - `src/dnadesign/baserender/src/outputs/images.py`
  - `src/dnadesign/baserender/src/outputs/video.py`

### Verification

- `uv run pytest -q src/dnadesign/baserender/tests` -> passed (`47` tests).
- `uv run ruff check src/dnadesign/baserender src/dnadesign/opal/notebooks/prom60_eda.py` -> passed.
- `uv run ruff format --check src/dnadesign/baserender src/dnadesign/opal/notebooks/prom60_eda.py` -> passed.
- `uv run baserender job validate src/dnadesign/baserender/docs/examples/densegen_job.yml` -> passed.
- `uv run baserender job run src/dnadesign/baserender/docs/examples/densegen_job.yml` -> passed.
- `uv run baserender job validate src/dnadesign/baserender/docs/examples/cruncher_job.yml` -> passed.
- `uv run baserender job run src/dnadesign/baserender/docs/examples/cruncher_job.yml` -> passed.

## 2026-02-12 - Tool-agnostic hardening pass (contract-driven parser + generic API)

### Objectives

- Remove remaining tool-specific coupling from public API and parser logic.
- Keep adapters tool-specific, but keep orchestration/config parsing generic and contract-driven.
- Preserve strict behavior and no-fallback constraints.

### Changes

- Removed tool-specific public helper from API surface:
  - deleted `render_densegen_record_figure` from:
    - `src/dnadesign/baserender/src/api.py`
    - `src/dnadesign/baserender/src/__init__.py`
    - `src/dnadesign/baserender/__init__.py`
- Migrated OPAL integration to generic adapter-based API:
  - `src/dnadesign/opal/notebooks/prom60_eda.py` now uses `render_parquet_record_figure(...)` with explicit adapter kind/columns.
- Refactored adapter policy validation to contract-owned normalizers:
  - Added policy normalizers in `src/dnadesign/baserender/src/config/adapter_contracts.py`.
  - Removed adapter-kind conditional policy parsing branches from `src/dnadesign/baserender/src/config/job_v3.py`.
- Broadened import boundary enforcement:
  - `src/dnadesign/baserender/tests/test_public_import_policy.py` now checks all sibling tool directories under `src/dnadesign/*` (excluding `baserender`) for disallowed `dnadesign.baserender.src.*` imports.
- Added static tool-agnostic architecture tests:
  - `src/dnadesign/baserender/tests/test_tool_agnostic_hardening.py`
  - Enforces no tool names in `src/api.py`.
  - Enforces no adapter-kind-specific parser branching/strings in `config/job_v3.py`.
- Updated runtime/public API test to use generic helper:
  - `src/dnadesign/baserender/tests/test_runtime_and_public_api.py`
- Updated docs to reflect caller-scoped defaults and public API boundary:
  - `src/dnadesign/baserender/README.md`
  - `src/dnadesign/baserender/docs/contracts/job_v3.md`
  - `src/dnadesign/baserender/docs/architecture/overview.md`

### Verification

- `uv run pytest -q src/dnadesign/baserender/tests` -> passed (`50` tests).
- `uv run ruff check src/dnadesign/baserender src/dnadesign/opal/notebooks/prom60_eda.py` -> passed.
- `uv run ruff format --check src/dnadesign/baserender src/dnadesign/opal/notebooks/prom60_eda.py` -> passed.
- CLI smoke:
  - `uv run baserender job validate src/dnadesign/baserender/docs/examples/densegen_job.yml` -> passed.
  - `uv run baserender job run src/dnadesign/baserender/docs/examples/densegen_job.yml` -> passed.
  - `uv run baserender job validate src/dnadesign/baserender/docs/examples/cruncher_job.yml` -> passed.
  - `uv run baserender job run src/dnadesign/baserender/docs/examples/cruncher_job.yml` -> passed.

## 2026-02-12 - Senior maintainer audit pass (workspace demos + import decoupling)

### Findings

1. High: CLI/API import coupling pulled matplotlib into non-render commands.
   - Importing `dnadesign.baserender.src.api` loaded render stack transitively, causing `workspace list`/`job validate` to initialize matplotlib and emit cache warnings in restricted environments.
2. Medium: Curated workspace demos were required but needed to be fully self-contained and runnable in isolated copies.
3. Medium: Public boundary needed stronger enforcement across sibling tools after API hardening.

### Changes

- Added curated independent workspaces under `src/dnadesign/baserender/workspaces/`:
  - `demo_densegen_render`
  - `demo_cruncher_render`
- Added self-contained Cruncher demo artifacts for docs examples:
  - `src/dnadesign/baserender/docs/examples/data/cruncher_demo_elites.parquet`
  - `src/dnadesign/baserender/docs/examples/data/cruncher_demo_elites_hits.parquet`
  - `src/dnadesign/baserender/docs/examples/data/cruncher_demo_config.yaml`
- Updated docs/example configs and workspace docs to point to local, runnable inputs.
- Removed eager render-stack imports from API module load:
  - `src/dnadesign/baserender/src/api.py`
  - lazy render/output imports now happen only in render execution paths.
- Removed eager runtime/bootstrap coupling and made render exports lazy:
  - `src/dnadesign/baserender/src/runtime.py`
  - `src/dnadesign/baserender/src/render/__init__.py`
- Kept explicit runtime contract by requiring `initialize_runtime()` for direct low-level renderer use.

### Tests added/updated

- Added `src/dnadesign/baserender/tests/test_workspace_demos.py`:
  - verifies curated demos are self-contained
  - verifies docs cruncher example uses local data
  - verifies both demos run in isolated copied workspace roots
- Added `src/dnadesign/baserender/tests/test_api_import_coupling.py`:
  - importing `dnadesign.baserender.src.api` must not preload matplotlib.
- Updated direct renderer tests to bootstrap runtime explicitly:
  - `src/dnadesign/baserender/tests/test_densegen_adapter.py`
  - `src/dnadesign/baserender/tests/test_motif_logo_effect.py`

### Verification

- `uv run pytest -q src/dnadesign/baserender/tests` -> passed (`54` tests).
- Non-render CLI commands are now clean:
  - `uv run baserender workspace list --root src/dnadesign/baserender/workspaces` -> no matplotlib warnings.
  - `uv run baserender job validate --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces` -> no matplotlib warnings.
- Render flows still pass:
  - `uv run baserender job run --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces`
  - `uv run baserender job run --workspace demo_cruncher_render --workspace-root src/dnadesign/baserender/workspaces`

## 2026-02-13 - Cruncher elites showcase integration hardening

### Findings

1. High: replacing `overlap_panel` with baserender-backed `elites_showcase` exposed a real contract mismatch between Cruncher analysis PWMs and sampled hit widths.
   - `run_analyze` loaded full PWMs from `config_used.yaml` (`pwms_info.pwm_matrix`), while sampled hits were generated with motif-width capping (`sample.motif_width.maxw`), causing hard failures like:
     - `PWM length mismatch ... matrix rows=22 but hit pwm_width=16`
2. Medium: multi-panel baserender API support was needed for one-figure-per-run elite showcases.
3. Medium: motif logos needed explicit strand-aware placement and matrix orientation for reverse-strand windows.

### Changes

- Added baserender public helper for multi-panel composition:
  - `render_record_grid_figure(...)` in `src/dnadesign/baserender/src/api.py`
  - exported from:
    - `src/dnadesign/baserender/src/__init__.py`
    - `src/dnadesign/baserender/__init__.py`
- Hardened motif logo rendering contract and geometry:
  - added `reverse_complement_matrix(...)`
  - enforced matrix-length == target kmer length
  - forward logos render above windows; reverse logos render below windows
  - files:
    - `src/dnadesign/baserender/src/render/effects/motif_logo.py`
    - `src/dnadesign/baserender/src/render/layout.py`
    - `src/dnadesign/baserender/src/core/registry.py`
- Implemented new Cruncher plot module and wiring:
  - added `src/dnadesign/cruncher/src/analysis/plots/elites_showcase.py`
  - replaced overlap panel invocation in:
    - `src/dnadesign/cruncher/src/app/analyze_workflow.py`
    - `src/dnadesign/cruncher/src/analysis/plot_registry.py`
  - removed old `src/dnadesign/cruncher/src/analysis/plots/overlap.py`
- Fixed root-cause PWM-width mismatch by reconstructing effective sampling PWM windows in analyze metadata when `sample.motif_width` is present:
  - `src/dnadesign/cruncher/src/app/analyze/metadata.py`
  - behavior is strict-but-conditional:
    - if `sample.motif_width` exists, apply deterministic `max_info` windowing to analysis PWMs;
    - if absent, preserve legacy full-width behavior.
- Updated Cruncher config/docs/tests for `elites_showcase` output and `analysis.elites_showcase.max_panels`.

### Verification

- Focused tests:
  - `uv run pytest -q src/dnadesign/baserender/tests/test_runtime_and_public_api.py src/dnadesign/baserender/tests/test_motif_logo_effect.py src/dnadesign/cruncher/tests/analysis/test_overlap_plots.py src/dnadesign/cruncher/tests/analysis/test_analysis_artifacts.py src/dnadesign/cruncher/tests/app/test_end_to_end_demo.py` -> passed
- Lint (touched files):
  - `uv run ruff check <touched files>` -> passed
- Workspace demo CLI runs:
  - `uv run baserender workspace list --root src/dnadesign/baserender/workspaces` -> lists densegen + cruncher demos
  - `uv run baserender job validate|run src/dnadesign/baserender/workspaces/demo_densegen_render/job.yml` -> passed
  - `uv run baserender job validate|run src/dnadesign/baserender/workspaces/demo_cruncher_render/job.yml` -> passed
- Cruncher two-TF runtime behavior:
  - `uv run cruncher analyze --summary -c src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml` -> passed
  - output plot set now includes `plot__elites_showcase.pdf` and no longer includes `plot__overlap_panel.pdf` after rerun.

## 2026-02-13 - Cruncher workspace consolidation to one hotpath demo

### Decision

- Keep one Cruncher-focused baserender workspace only: `demo_cruncher_render`.
- Use Record-shape hotpath input (`generic_features`) as the canonical iteration path.
- Keep Cruncher runtime-adjacent source artifacts in the same workspace `inputs/` as references.

### Implementation

- `demo_cruncher_render/job.yml` now targets:
  - `inputs/elites_showcase_records.parquet`
  - adapter `kind: generic_features`
- Kept source-like reference artifacts in `demo_cruncher_render/inputs/`:
  - `elites.parquet`
  - `elites_hits.parquet`
  - `config_used.yaml`
- Removed separate `demo_elites_showcase_hotpath` workspace to avoid duplicate intent.
- Updated docs/tests to assert one Cruncher demo workspace model.

### Why

- Fast visual iteration should target the exact baserender Record boundary used for design work.
- Reference artifacts from Cruncher remain co-located for accurate runtime-shape grounding and future regeneration.
