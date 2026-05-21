# OPAL Modernization Development Specification

**Status:** Draft for engineering planning and review
**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-21
**Audience:** OPAL maintainers, study integrators, and developer-experience owners
**Date:** 2026-05-21
**Scope:** OPAL campaign runtime, reporting, plot artifacts, CLI JSON surfaces, and generated marimo review notebooks
**Chosen path:** `src/dnadesign/opal/docs/maintainers/dev-spec-modernization.md`

The requested path `docs/opal/dev-spec-modernization.md` is not present in the current repository layout. OPAL's checked-in documentation lives under `src/dnadesign/opal/docs/`, with maintainer planning material under `src/dnadesign/opal/docs/maintainers/`. This spec is placed there so it stays in the tool-local OPAL docs tree instead of creating a new root-level docs island.

This document is a specification only. It intentionally does not implement production code.

## 1. Executive Summary

OPAL should become a small, contract-first active-learning campaign runtime with excellent machine-readable reporting and review artifacts. Its core identity is a campaign loop over one candidate table, one explicit X column, one label source, model/objective/selector plugins, append-only ledgers, run-scoped progress, configured plots, static review bundles, and generated marimo notebooks.

The reason to modernize now is not that the round loop is broken. Repository evidence shows OPAL already has strict config loading, channelized objectives, ledger contracts, run-aware review, plot registries, plot artifact manifests, public reporting exports, generated single-campaign notebooks that use public OPAL helpers, and explicit campaign-set notebook generation. The full-pool 12-round DenseGen-axis dogfood removed the old candidate-cap shortcut and completed scoring with bounded batches, but it exposed a real production risk: label ingest, artifact review, and progress surfaces must make their memory and stale-evidence contracts operator-visible, not merely pass happy-path tests. The remaining gap is product and contract maturity: ingest must stay memory-safe and telemetry-rich, artifacts must be gardened so stale run debris cannot look current, leakage must fail fast, generated notebooks need smaller reusable panel primitives, active dashboard code still carries UMAP/projection residue outside the canonical notebook, progress events still need clearer lock/preflight/run boundaries, and the X contract still has a split between permissive in-memory identity parsing and strict Parquet validation.

Highest-priority changes:

| Priority | Change | Outcome |
| --- | --- | --- |
| P0 | Make `opal ingest-y` column-pruned, memory-guarded, and telemetry-rich | Full-pool campaigns can complete rounds without loading multi-GB X payloads for small label appends |
| P0 | Add artifact gardening contracts for stale files, ignored `.var` run roots, and local-only dogfood evidence | Stale craft buildup is visible, bounded, and never mistaken for current review evidence |
| P0 | Fail fast on leakage, mixed train/eval state, and contaminated label/prediction surfaces | OPAL and study-owned probes fail closed instead of smoothing over invalid evidence |
| P1 | Harden run/progress/review contracts around run_id, attempt_id, preflight events, aborts, locks, raw metrics, derived review metrics, and stale artifacts | Machines and operators can trust status without log archaeology |
| P2 | Unify the X physical schema around fixed-size finite vectors, with explicit import normalization for noncanonical forms | Campaign execution and review fail fast on invalid X |
| P3 | Harden plot artifact manifests and manifest-first review/notebook rendering | Stale files cannot masquerade as current evidence |
| P4 | Extend generic, data-shape-based plot primitives | New campaign diagnostics configure reusable plot kinds instead of bespoke study plots |
| P5 | Refactor generated notebooks into reusable public marimo view-model and component primitives, with campaign and plot dropdowns as generic scoping controls | Notebooks stay navigable and expressive without duplicating template logic or overfitting to a probe |
| P6 | Resolve campaign information architecture with explicit ownership metadata before relocating live study configs | OPAL keeps generic templates while study-owned biological campaigns remain discoverable and future-proof |
| P7 | Expand dogfood coverage beyond local synthetic-oracle evidence | OPAL readiness claims are scoped to durable evidence |

Explicitly out of scope:

- OPAL will not become a LatentDNA geometry browser, UMAP atlas, DenseGen visualizer, or study-specific benchmark harness.
- OPAL will not own DenseGen synthetic-oracle logic, cipro/ethanol/AND biological interpretation, stress-axis aggregate reports, or scratch-only synthetic labels.
- OPAL will not silently migrate invalid data, guess run scope, or fall back from configured contracts.
- OPAL will not rely on hidden candidate caps, implicit subsets, or memory-unsafe full-table materialization as routine campaign semantics.
- This spec does not implement production code.

## 2. Current-State Assessment

### Verified Facts

OPAL's stated package intent is active-learning campaigns over labeled sequence datasets with explicit feature, objective, selection, and ledger contracts. The README states that directly, and the architecture doc maps the round lifecycle from config load through labels, X matrices, model fit, objective channels, selection, and persisted outputs.

The current runtime already has good seams:

- `campaign.yaml` declares campaign, data, label source, transforms, model, objectives, selection, training, scoring, writeback, safety, and plot config.
- Config loading forbids unknown Pydantic fields and unknown plugins; shared `usr_sidecar` labels require matching USR dataset, explicit writeback, `y_space`, and relative sidecar path.
- Objectives emit explicit channels, and selection consumes configured refs such as `sfxi_v1/sfxi`.
- Ledgers use allow-listed schemas for `run_pred`, `run_meta`, and `label`, and reject unknown columns unless an explicit environment escape hatch is enabled.
- Public package exports now include `load_config`, `read_campaign_predictions`, `build_campaign_progress`, `render_campaign_progress_text`, `build_campaign_review`, and `validate_x_parquet_column`.
- `opal progress --json` and `opal review --json` exist, and the study-owned DenseGen probe uses the public OPAL progress/review helpers rather than importing OPAL internals.

Current CLI behavior observed during this pass:

| Command | Result |
| --- | --- |
| `git status --short` | Dirty worktree included pre-existing study-owned DenseGen probe hardening plus OPAL modernization edits; generated/run artifacts must be reviewed separately before any commit. |
| `uv run opal --help` | Passed; CLI includes `progress`, `review`, `plot`, `notebook`, `status`, and `runs`. |
| `uv run opal progress --help` | Passed; accepts `--config`, `--round`, and `--json/--text`. |
| `uv run opal review --help` | Passed; accepts `--run-id`, `--plots/--no-plots`, and JSON output. |
| `uv run opal plot --help` | Passed; supports `--list`, `--list-config`, `--describe`, `--round`, `--run-id`, `--name`, and tags. |
| `uv run opal plot --list` | Passed; listed 13 registered plot kinds, including `metric_over_rounds`, `feature_importance_heatmap`, and `vector_summary_heatmap`. |
| `uv run opal plot --list --json` | Passed; emitted schema `opal.plot_registry.v1` with `PlotMeta` data shapes, tidy schemas, and failure modes where declared. |
| `uv run opal notebook --help` | Passed; has `generate` and `run`. |
| `uv run opal status --help` | Passed; supports `--with-ledger` and JSON. |
| `uv run opal runs --help` | Passed; has `list` and `show`. |
| `uv run opal notebook generate --help` | Passed; supports directory-capable config, `--round`, `--out`, `--name`, `--force`, and validation. |
| `uv run marimo check <generated OPAL notebook>` | Passed for the representative cipro-positive dogfood notebook. |

### Campaign Lifecycle

The documented OPAL round lifecycle is:

1. Load and validate config and plugin names.
2. Resolve labels through the configured source.
3. Build feature matrices through `transforms_x`.
4. Fit a model and predict.
5. Apply Y-op inversion.
6. Evaluate objectives into named channels.
7. Run selection using explicit channel refs and persist outputs.

The implementation follows the same shape: `run_round()` checks state, round overwrite safety, writes a start event, runs training, creates a run context/run_id, builds X matrices, scores/predicts, writes artifacts, appends ledgers, updates state, and writes terminal log events.

Important current issue: `cmd_run()` writes `command_start`, `records_load_start`, and `records_load_done` before acquiring `CampaignLock`, while `run_round()` only creates `run_id` after training starts. That makes early progress useful, but also means preflight/aborted events and actual run events are not yet cleanly typed.

### Ingest Lifecycle And Runtime Cost

The full-pool 12-round dogfood changed the runtime risk profile. OPAL scoring can be made memory-bounded with `score_batch_size=256`, and fixed-universe `usr_sidecar` label ingest now uses a narrow identity frame rather than materializing the full X-heavy records table. The remaining ingest work is to keep that behavior explicit and test-pinned: the CLI must expose `opal.ingest_runtime.v1`, loaded columns, candidate index size, estimated frame memory, optional peak RSS, unknown-label policy, and write scope in both JSON and operator text.

The desired production invariant remains memory proportionality: ingest memory should scale with the incoming label batch plus a narrow identity index, not with the full feature payload. Full candidate scoring may need batched X reads; label ingestion should not.

### Public APIs

The public package exports are intentionally small and useful for study code. This is the correct direction. Generated notebooks now import public OPAL helpers instead of `dnadesign.opal.src.*`. The remaining boundary issue is size and granularity: durable generated artifacts should depend on a small public view-model/component API rather than a broad helper set and a single large template.

### CLI Progress And Review

`build_campaign_progress()` returns `opal.campaign_progress.v1` with generated time, campaign identity, state path, selector, status, round count, and per-round summaries. `build_campaign_review()` returns and writes `opal.campaign_review.v1`, including campaign metadata, review scope, run summary, progress summary, selection preview, plots, and artifact paths.

The review manifest is a good start, and the generated notebook now uses a manifest-backed `NotebookViewModel` for configured plot choices. The remaining manifest-authority gap is stale evidence handling: review can still emit `plots: []` while old plot PNGs exist on disk, so review/notebook JSON should report those extras as stale instead of allowing filesystem discovery to imply current evidence.

### Plot Registry And Configuration Surface

The plot system is already plugin-oriented:

- `PlotMeta` records `summary`, `params`, `requires`, and `notes`.
- `plots.yaml` is preferred via `plot_config`.
- Unknown plot config keys fail.
- Built-in data paths include records, outputs, ledger predictions, ledger runs, and ledger labels.
- `PlotContext` carries round selector, run_id, output path, format, DPI, and `save_data`.

Current registered plot kinds are:

| Kind | Current role |
| --- | --- |
| `feature_importance_bars` | Overlaid feature importance bars across rounds |
| `feature_importance_heatmap` | Attribution matrix with stable feature rows and round columns |
| `metric_over_rounds` | Generic scalar metric summaries over rounds by cohort |
| `scatter_score_vs_rank` | Score versus rank with selected candidates highlighted |
| `percent_high_activity_over_rounds` | Thresholded score progress over rounds |
| `fold_change_vs_logic_fidelity` | SFXI score/effect versus logic fidelity diagnostics |
| `sfxi_factorial_effects` | Factorial-effects map from predicted logic vectors |
| `sfxi_setpoint_sweep` | Setpoint sweep over labels |
| `sfxi_support_diagnostics` | Distance to labeled logic support versus score |
| `sfxi_uncertainty` | Uncertainty versus score/effect diagnostics |
| `sfxi_intensity_scaling` | Denominator, clipping, and raw-effect scaling diagnostics |
| `sfxi_logic_fidelity_closeness` | Observed label closeness to setpoint |
| `vector_summary_heatmap` | Vector-channel summary over setpoint, cohorts, and rounds |

The main plot gap has moved from artifactization to semantic completeness. Configured plots now write media, optional tidy CSVs, per-plot `opal.plot_artifact.v1` manifests, and aggregate `opal.plot_manifest_index.v1` indexes. The remaining work is to make every plot manifest consistently useful for review: declared data shape, tidy schema, failure modes, source freshness, stale state, captions, and decision relevance must be complete rather than optional metadata.

### Notebook Generation

The generated marimo notebook is campaign-specific and covers records, ledgers, selected records, labels, predictions, configured plot deliverables, and CLI handoff. It now imports public OPAL helpers, reads a manifest-backed `NotebookViewModel`, exposes round/run/record/data-source/plot dropdowns, and uses lazy accordions. Campaign-set notebook generation now exists as an explicit repeated `--campaign` mode backed by `NotebookCampaignSetViewModel`; it provides campaign and plot dropdowns without changing the single-campaign contract. The remaining UX work is deeper componentization and richer campaign-set drill-downs, not first support for campaign navigation.

### Study/Probe Separation

The stress study route correctly describes OPAL as consuming the `usr_prom_eth_cip_opal_candidates` candidate feature table with explicit X column, while LatentDNA owns prior X selection and study code owns pre-assay batch-0/probe logic. The DenseGen axis probe is study-owned, exports no package-root compatibility API, uses scratch-only synthetic labels, and calls public OPAL progress/review APIs.

The available DenseGen-axis dogfood evidence remains scoped. The current all-round run covers 12 scratch campaigns and correctly reports `STOP`/`attention` with structured decision reasons and weak count-aware tests; this is evidence that OPAL can run and review the probe, not evidence of real assay behavior. The earlier cipro/random pass remains one narrow positive scenario and must not be generalized to ethanol, dual/AND, leave-sigma35, or real biological readiness.

Campaign configuration ownership needs a deliberate information-architecture decision, not an accidental file move. OPAL may ship generic examples, reusable campaign templates, and fixture campaigns that demonstrate plugin contracts. Study-specific live campaigns that encode dataset IDs, biology, or stress-axis choices should either live under the owning study or carry explicit ownership metadata that marks them as external study fixtures. The migration should not use hidden compatibility shims; path changes should be explicit in docs, tests, and CLI examples.

### Available Artifact And Plot Quality Baseline

Local-only read-only audit of the current full-pool dogfood run root found a completed 12-campaign, 12-round run without a candidate cap: 144 OPAL rounds, `selection_k=6`, final `train_count=72`, final random-split eval count of 157,088, final leave-sigma35 eval count of 31,637, and 144 `selection_top_k.csv` files with exactly 6 rows each. The run tree is about 8.1 GB with 2,240 files. The aggregate review manifest reports 36 configured OPAL plots across 12 campaigns, all loaded with `plot_quality.status=ok`; the report bundle has 13 HTML files, 256 local refs with 0 missing refs, and 104 readable/nonblank/non-undersized PNGs.

That `.var` run is ignored local evidence, not durable CI evidence. It is a mechanical-health pass and a runtime-cost signal, not an interpretability pass or a real biological validation. Current checks prove files exist, are nonblank, are linked, and are manifest-backed. They do not yet prove that a plot explains what decision it supports, has a useful caption, includes threshold/reference context, exposes source freshness, or makes positive/null comparisons easy to compare across campaigns. Future dogfood claims should be backed by a tracked summary fixture or a reproducible command recorded in CI/nightly logs.

### Known Issues And Risks

| Issue | Risk |
| --- | --- |
| `opal ingest-y` fixed-sidecar appends now avoid the full records table, but the contract must stay visible and regression-tested | Operators need to see `mode=identity_index`, loaded columns, memory estimates, and write scope during normal dogfood |
| `opal run` has scoring memory guards; `ingest-y` now reports estimated frame memory and optional peak RSS but still needs configurable hard thresholds | Operators can complete scoring safely and should get explicit ingest memory posture before label writes |
| Local dogfood roots under ignored `.var` can grow to multi-GB bundles and accumulate stale plot siblings | Local evidence, stale artifacts, and workspace storage pressure can drift unless artifact gardening is explicit |
| Probe review currently enriches report metrics and status from a review pass | Raw execution evidence and derived review evidence can become hard to distinguish unless artifacts are split |
| Leakage guards are partly study-owned and partly OPAL-owned | Invalid synthetic-oracle evidence, train/eval overlap, or prediction/label contamination must fail fast at the correct layer |
| Stress-specific campaign configs currently live under the OPAL package tree | OPAL information architecture can imply ownership of study-specific biology unless configs are moved or explicitly marked as study-owned fixtures |
| Active OPAL dashboard helpers still mention UMAP/projection/cluster views outside the canonical generated notebook | Boundary drift: noncanonical dashboard code can be mistaken for the OPAL campaign-review surface |
| Generated notebook implementation now has the first plot-gallery component split into `notebook_components.py`, but the template module remains a 648-line generated string and a representative generated notebook is about 720 lines | Notebook UX and maintenance can regress until the remaining panels are independently testable marimo primitives |
| Campaign-set notebook UX now exists, but it is overview-first: campaign/plot dropdowns, status, provenance, warnings, stale artifacts, and plot cards | A 12-campaign probe can be triaged in one OPAL notebook, but record-level drill-down still belongs to single-campaign notebooks |
| Plot manifests exist, but not every plot kind declares data shape, tidy schema, and failure modes | Plot cards can be mechanically valid while still under-explaining what the visual decides |
| Plot manifest schema is evolving from mechanical artifact presence to semantic quality/freshness metadata | Docs and tests must pin status enums, tidy schema validation, captions, and freshness fields so agents can trust manifests literally |
| Review manifests and plot manifests detect stale artifacts, but stale/fresh semantics are still shallow | Review evidence can be mechanically present while source freshness or decision relevance remains unclear |
| `cmd_run()` logs early command/records events before lock and before run_id | Progress can mix preflight, abort, and run events |
| X docs now declare canonical Parquet fixed-size lists, but `identity` still accepts scalar/list/JSON cells as an in-memory transform | Runtime parsing remains more permissive than the campaign-storage contract and needs a deliberate compatibility boundary |
| `CampaignLock` and `PathLock` are local-host locks | Shared/network mutation needs a stronger lease or documented non-support |
| Plot primitives include several SFXI-specific single-round diagnostics | Useful diagnostics, but future plots should be data-shape primitives first |

## Evidence Ledger

| Observation | Evidence |
| --- | --- |
| OPAL intent is explicit active-learning over feature/objective/selection/ledger contracts | `src/dnadesign/opal/README.md:3-4` |
| Documented round lifecycle and runtime surfaces | `src/dnadesign/opal/docs/concepts/architecture.md:9-32` |
| OPAL's documented fail-fast model | `src/dnadesign/opal/docs/concepts/architecture.md:54-63` |
| Config top-level blocks, defaults, shared sidecar policy, and plot config wiring | `src/dnadesign/opal/docs/reference/configuration.md:12-30`, `src/dnadesign/opal/docs/reference/configuration.md:44-62`, `src/dnadesign/opal/docs/reference/configuration.md:94-101`, `src/dnadesign/opal/docs/reference/configuration.md:180-193` |
| Loader forbids unknown fields and validates plugin names and shared sidecar constraints | `src/dnadesign/opal/src/config/loader.py:179-205`, `src/dnadesign/opal/src/config/loader.py:253-291`, `src/dnadesign/opal/src/config/loader.py:315-333` |
| Public OPAL API currently exports config, predictions, progress, review, plot, notebook view-model, and X validation helpers | `src/dnadesign/opal/__init__.py:14-43` |
| Data-contract docs declare runtime X as canonical Parquet fixed-size vectors, while identity transform remains permissive for in-memory parsing | `src/dnadesign/opal/docs/reference/data-contracts.md:41-59`, `src/dnadesign/opal/src/storage/x_contracts.py:70-97`, `src/dnadesign/opal/src/transforms_x/identity.py:29-78` |
| Strict X validator tests reject variable-list and scalar physical schemas | `src/dnadesign/opal/tests/storage/test_x_contracts.py:61-90` |
| Runtime identity transform accepts scalar/list/JSON-string inputs | `src/dnadesign/opal/src/transforms_x/identity.py:29-78` |
| CLI run logs command/X-validation/records events before acquiring the campaign lock later | `src/dnadesign/opal/src/cli/commands/run.py:97-152` |
| Runtime creates run_id after training and logs later run-scoped stages | `src/dnadesign/opal/src/runtime/run_round.py:119-177`, `src/dnadesign/opal/src/runtime/run_round.py:182-268`, `src/dnadesign/opal/src/runtime/run_round.py:330-380` |
| Progress JSON builder summarizes campaign state, round logs, warnings, stale artifacts, and local lock state | `src/dnadesign/opal/src/reporting/progress.py:30-82`, `src/dnadesign/opal/src/reporting/progress.py:143-182` |
| Round summary filters by run_id when supplied, slices to latest start when multiple starts exist, and reports run scope plus phase counts | `src/dnadesign/opal/src/reporting/summary.py:114-202` |
| Review writer creates schema `opal.campaign_review.v1`, manifest, Markdown, HTML, selection preview, and plot status list | `src/dnadesign/opal/src/reporting/review.py:67-166`, `src/dnadesign/opal/src/reporting/review.py:188-253`, `src/dnadesign/opal/src/reporting/review.py:256-327` |
| Review tests cover manifest path, schema, run_id scope, and run-log mismatch failure | `src/dnadesign/opal/tests/reporting/test_review.py:50-117` |
| Plot docs define PlotContext, plots.yaml, strict params placement, built-in paths, and save_data | `src/dnadesign/opal/docs/reference/plots.md:7-28`, `src/dnadesign/opal/docs/reference/plots.md:83-101`, `src/dnadesign/opal/docs/reference/plots.md:163-178` |
| Plot docs and implementation use `written`/`failed` manifest statuses and include quality/freshness metadata | `src/dnadesign/opal/docs/reference/plots.md:117-132`, `src/dnadesign/opal/src/plots/manifests.py:22-170`, `src/dnadesign/opal/src/plots/runner.py:278-309` |
| PlotMeta shape and registry/entry point loading | `src/dnadesign/opal/src/registries/plots.py:27-32`, `src/dnadesign/opal/src/registries/plots.py:40-55`, `src/dnadesign/opal/src/registries/plots.py:63-82` |
| Plot config rejects unknown keys and conflicting inline/external plot config | `src/dnadesign/opal/src/plots/config.py:23-42`, `src/dnadesign/opal/src/plots/config.py:75-78`, `src/dnadesign/opal/src/plots/config.py:161-216` |
| Plot runner injects built-in data paths, builds PlotContext, calls plugins, writes per-plot manifests, and writes aggregate plot manifest indexes | `src/dnadesign/opal/src/plots/runner.py:93-107`, `src/dnadesign/opal/src/plots/runner.py:243-315`, `src/dnadesign/opal/src/plots/manifests.py:22-130` |
| Feature importance plot currently discovers per-round files and writes optional tidy CSV | `src/dnadesign/opal/src/plots/feature_importance_bars.py:34-53`, `src/dnadesign/opal/src/plots/feature_importance_bars.py:163-176`, `src/dnadesign/opal/src/plots/feature_importance_bars.py:213-317` |
| Fixed-sidecar `opal ingest-y` loads only the identity frame and emits `opal.ingest_runtime.v1` runtime telemetry | `src/dnadesign/opal/src/cli/commands/ingest_y.py`, `src/dnadesign/opal/src/storage/data_access.py`, `src/dnadesign/opal/src/runtime/ingest_runtime.py` |
| `opal ingest-y` needs ID/sequence membership and required-column checks for fixed-universe sidecars, not the configured X payload | `src/dnadesign/opal/src/cli/commands/ingest_y.py`, `src/dnadesign/opal/src/runtime/ingest.py` |
| Generated notebook imports public OPAL helpers rather than `dnadesign.opal.src.*` | `src/dnadesign/opal/src/analysis/notebook_template.py:41-88`, `src/dnadesign/opal/tests/notebooks/test_notebook_template.py:59-70` |
| Generated notebook builds a manifest-backed view model and uses manifest-backed plot choices | `src/dnadesign/opal/src/reporting/notebook.py:34-113`, `src/dnadesign/opal/src/analysis/notebook_template.py:99-102`, `src/dnadesign/opal/src/analysis/notebook_template.py:401-518` |
| Generated notebook exposes round, run, record, data-source, and plot dropdowns plus lazy accordions | `src/dnadesign/opal/src/analysis/notebook_template.py:159-203`, `src/dnadesign/opal/src/analysis/notebook_template.py:306-340`, `src/dnadesign/opal/src/analysis/notebook_template.py:439-518`, `src/dnadesign/opal/src/analysis/notebook_template.py:700-746` |
| Notebook template is still mostly a large generated string and generated artifact, despite the extracted plot-gallery cell helper | `src/dnadesign/opal/src/analysis/notebook_template.py`, `src/dnadesign/opal/src/analysis/notebook_components.py`; `wc -l` measured 648 lines in the template, 141 lines in the first component module, and about 720 lines in a representative generated notebook |
| Active dashboard analysis includes UMAP chart/view code | `src/dnadesign/opal/src/analysis/dashboard/views/plots.py:6-7`, `src/dnadesign/opal/src/analysis/dashboard/views/plots.py:101-139` |
| Notebook tests already assert lateral tools and default UMAP column strings stay out of generated template | `src/dnadesign/opal/tests/notebooks/test_notebook_template.py:70-86` |
| Locks are local-host file locks | `src/dnadesign/opal/src/storage/locks.py:86-108` |
| Ledgers are strict and have an explicit escape hatch | `src/dnadesign/opal/src/storage/ledger.py:37-168` |
| Ledger reader enforces run_id disambiguation when multiple runs exist | `src/dnadesign/opal/src/storage/ledger.py:340-401` |
| SFXI is an objective plugin over vec8 predictions and setpoints, not a study core concept | `src/dnadesign/opal/src/objectives/sfxi_v1.py:328-345`, `src/dnadesign/opal/src/objectives/sfxi_v1.py:363-427`, `src/dnadesign/opal/src/objectives/sfxi_math.py:18-34`, `src/dnadesign/opal/src/objectives/sfxi_math.py:45-60` |
| Stress OPAL route defines OPAL candidate table, explicit X column, and OPAL/study boundary | `docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md:23-35`, `docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md:45-74` |
| Candidate-table context keeps shared labels and campaign ledgers separate | `docs/studies/stress_ethanol_cipro_growth/contexts/opal/candidate-table.md:19-38`, `docs/studies/stress_ethanol_cipro_growth/contexts/opal/candidate-table.md:40-54` |
| DenseGen probe is study-owned, scratch-only, and forbids label leakage | `docs/studies/stress_ethanol_cipro_growth/contexts/opal/densegen-axis-probe-v0.md:1-36`, `docs/studies/stress_ethanol_cipro_growth/contexts/opal/densegen-axis-probe-v0.md:143-179` |
| DenseGen probe package root exports no flat compatibility API | `src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/__init__.py:1-9` |
| DenseGen probe consumes public OPAL progress and review APIs | `src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/progress.py:9-27`, `src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/review.py:14-39` |
| Local-only full-pool DenseGen-axis dogfood report is STOP/attention with 12 campaigns, 144 OPAL rounds, no candidate cap, 36 configured OPAL plots, 8 aggregate probe plots, 3 null-control decision reasons, and metric definitions for lift, null lift, p, and round semantics; this ignored `.var` path is not durable CI evidence | `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/dogfood_all_rounds12_full_streaming_conservative_20260521T0231Z/reports/review_manifest.json`, `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/dogfood_all_rounds12_full_streaming_conservative_20260521T0231Z/reports/metrics.json` |
| Local-only full-pool dogfood artifact audit found an 8.1 GB run tree with 2,240 files, 13 HTML files, 256 local refs with 0 missing refs, and 104 readable/nonblank/non-undersized PNGs; future claims need a tracked summary or reproducible CI/nightly log | Read-only audit command recorded in this spec pass; root `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/dogfood_all_rounds12_full_streaming_conservative_20260521T0231Z` |
| Local-only full-pool dogfood showed scoring stayed bounded with `score_batch_size=256`; future ingest regressions should be caught by the identity-frame contract and memory telemetry | Operator polling during full-pool run; follow-up dogfood should record `opal.ingest_runtime.v1` evidence |

## 3. Goals

| Goal | Target outcome |
| --- | --- |
| Maintainability | OPAL remains a small runtime kernel plus narrow extension points; internal modules can be refactored without breaking study code or generated notebooks. |
| Runtime memory safety | Ingest, run, review, plot, and notebook generation have explicit memory posture; normal full-pool campaigns do not require loading full X payloads outside bounded scoring. |
| Artifact gardening | Generated artifacts are inventoried, stale files are visible, local-only evidence is labeled, and cleanup/prune actions are explicit operator commands. |
| Fail-fast contracts | Unknown config keys, ambiguous run selection, missing columns, invalid X, stale artifacts, and mixed-run review produce explicit errors or warnings with stable machine-readable codes. |
| Leakage prevention | OPAL fails fast on generic train/eval, label-source, and prediction-contamination invalid states; study/probe code fails fast on study-specific forbidden inputs. |
| Public/private API hygiene | Study/probe code and generated notebooks use `dnadesign.opal` public APIs or documented subpackages; `dnadesign.opal.src.*` remains internal. |
| Robust progress and review | Progress and review are run_id-scoped, lock-aware, abort-aware, manifest-authoritative, and clear about stale/missing data. |
| Artifactized plots | Every plot run emits a per-plot manifest plus media and optional tidy CSV so review surfaces can render evidence without directory scraping. |
| Marimo notebook UX | Generated notebooks become manifest-backed campaign review viewers with at-a-glance state, validity/change/evidence/limitations sections, lazy accordions, and no representation-browser content. |
| Extensible plot ontology | New plots are based on reusable data shapes: scalar, vector, matrix, categorical, overlap, attribution, uncertainty/support, objective decomposition, audit table, and calibration. |
| Campaign-agnostic review | OPAL review describes campaign contracts, ledgers, progress, selection behavior, configured plots, and limitations without encoding study semantics. |
| CI/testability | Contracts are tested through unit tests, CLI JSON tests, schema fixtures, snapshot tests, and dogfood gates with explicit coverage labels. |

## 4. Non-Goals

- No LatentDNA geometry browser inside OPAL.
- No UMAP atlas, cluster browser, projection-review gate, or representation-browser content in canonical OPAL review/notebook surfaces.
- No DenseGen, cipro, ethanol, AND, or stress-specific logic in OPAL core.
- No study-owned synthetic-oracle benchmarking inside OPAL.
- No silent migration, hidden fallback behavior, or guessed schema coercion during campaign execution.
- No overfitted plot kinds that only work for one probe when the underlying data shape is generic.
- No filesystem-discovery-first review or notebook rendering.
- No probe-specific campaign-set notebook implementation; campaign selection and plot selection are generic view-model controls.
- No automatic deletion of generated artifacts during inspection commands.
- No production implementation in this spec pass.

## 5. Architecture Principles And Invariants

1. **One explicit candidate table.** A campaign consumes exactly one configured candidate table. The candidate universe is a contract, not an implicit subset of upstream representation artifacts.
2. **One explicit X column.** A campaign names one X column. OPAL may report its provenance as a string, but must not depend on how that X was produced.
3. **Explicit label source.** Label source kind, dataset/path, y space, ID column, round column, batch column, and dedup policy are explicit. A configured shared sidecar never falls back to campaign-local history.
4. **Explicit channel refs.** Selection reads score and uncertainty by channel ref, not by implicit "the score" columns.
5. **Append-only ledgers.** Ledgers are the durable audit layer. Compaction and repair are explicit operator actions.
6. **Run_id-scoped progress.** Progress and review must disclose run_id scope and ambiguity. Multiple runs per round are not silently mixed.
7. **Manifest-authoritative artifacts.** Review and notebook surfaces read manifests first. Directory contents are advisory and can only produce stale-file warnings.
8. **Strict config loading.** Unknown keys, duplicate YAML keys, unknown plugins, invalid plugin params, and incompatible shared-label config fail fast.
9. **Public API boundaries.** Cross-package consumers use public APIs. Internal module imports are allowed inside OPAL implementation but not as durable external contracts.
10. **Campaign agnosticism.** OPAL primitives operate on data shapes and declared channels. Study-specific semantics configure those primitives.
11. **Machine readability.** CLI JSON, ledgers, manifests, plot data, and notebook view models carry schema versions and stable error/warning structures.
12. **Progressive disclosure.** CLI/JSON are the control plane. Static review and marimo are inspection layers with heavy sections hidden behind lazy accordions.
13. **Memory proportionality.** Ingest and inspection memory should scale with the requested operation, not with the full candidate feature payload unless the operation explicitly scores or validates that payload.
14. **Leakage fails closed.** Generic OPAL contamination states and study-owned forbidden-input states are errors by default; warnings are only acceptable for read-only inspections that cannot prove contamination.
15. **Generated artifact lifecycle.** Artifact creation, stale detection, retention, pruning, and local-only evidence summaries are explicit lifecycle states, not incidental filesystem residue.

## 6. Proposed Target Architecture

The target architecture keeps abstractions small. Use dataclasses, protocols, JSON schemas, and registry metadata before introducing heavier frameworks.

### Contract Components

| Component | Responsibility | Minimal shape |
| --- | --- | --- |
| `CampaignConfigContract` | Validated campaign config, config source path, schema version, and plugin refs | Dataclass wrapping `RootConfig` plus `schema_version`, `config_path`, `strict_mode` |
| `XMatrixContract` | Physical and logical X schema for candidate table | `records_path`, `x_column`, `id_column`, `physical_type`, `x_dim`, `row_count`, `normalization_status` |
| `LabelSourceContract` | Label source identity, y space, dedup policy, and write lock semantics | `kind`, `dataset`, `path`, `y_space`, columns, `requires_existing_for_run`, `lock_scope` |
| `IngestRuntimeContract` | Memory-safe label ingest plan and result | `mode`, `identity_columns`, `input_rows`, `candidate_index_rows`, `estimated_memory_bytes`, `peak_rss_bytes`, `unknown_policy`, `write_scope` |
| `LeakageGuardContract` | Generic contamination and study-owned forbidden-input checks | `checks`, `scope`, `status`, `violations`, `severity`, `owner` |
| `RoundRunContract` | One actual run attempt, separated from preflight | `run_id`, `round_index`, `phase`, `started_at`, `completed_at`, `aborted_at`, `status`, `lock_token` |
| `ProgressEventContract` | Structured event stream for preflight and run phases | `schema_version`, `event_id`, `phase`, `run_id`, `stage`, `severity`, `ts`, `payload` |
| `ReviewManifestContract` | Authoritative campaign review bundle | `schema_version`, `review_scope`, `campaign`, `run`, `progress`, `selection`, `plots`, `stale_artifacts`, `warnings` |
| `ArtifactGardenContract` | Run-root and artifact-dir inventory, stale detection, retention, and prune plan | `artifact_roots`, `active_manifests`, `stale_artifacts`, `local_only`, `bytes`, `retention_policy`, `prune_plan` |
| `PlotDataContract` | Plot plugin input declaration and tidy data schema | `kind`, `required_sources`, `required_columns`, `optional_columns`, `tidy_schema`, `failure_modes` |
| `PlotArtifactManifest` | Per-plot output authority | `schema_version`, `plot_id`, `kind`, `params`, `run_id`, `rounds`, `inputs`, `outputs`, `status`, `generated_at`, `stale_state` |
| `NotebookViewModel` | Manifest-backed marimo input surface | `campaign_state`, `review_manifest`, `plot_manifests`, `warnings`, `links`, `tables` |
| `NotebookCampaignSetViewModel` | Optional multi-campaign notebook input surface | `campaigns`, `active_campaign_id`, `campaign_states`, `plot_catalog`, `warnings`, `comparison_tables` |
| `MarimoComponentPrimitives` | Public render helpers used by single-campaign and campaign-set notebooks | `render_campaign_selector`, `render_at_a_glance`, `render_validity_panel`, `render_plot_card`, `render_plot_gallery`, `render_distrust_panel` |
| `Public Reporting API` | Stable functions for progress, review, predictions, status, manifests | `build_campaign_progress`, `build_campaign_review`, `read_campaign_predictions`, `load_review_manifest`, `inspect_campaign_status` |
| `Public Plot API` | Stable functions for plot metadata and manifests | `list_plot_kinds`, `describe_plot_kind`, `load_plot_artifact_manifest`, `run_configured_plots` |
| `Public Notebook API` | Stable generated-notebook helper surface | `build_notebook_view_model`, `build_campaign_set_view_model`, `render_campaign_notebook`, `render_campaign_set_notebook`, `smoke_check_notebook` |

### Illustrative Dataclass Shapes

```python
@dataclass(frozen=True)
class XMatrixContract:
    schema_version: str
    records_path: Path
    x_column: str
    id_column: str = "id"
    physical_type: str = "fixed_size_list<float32>"
    x_dim: int
    row_count: int
    null_count: int = 0
    nonfinite_count: int = 0
    canonical: bool = True
```

```python
@dataclass(frozen=True)
class PlotArtifactManifest:
    schema_version: str
    plot_id: str
    name: str
    kind: str
    status: Literal["written", "skipped", "failed", "stale"]
    generated_at: str
    run_id: str | None
    rounds: list[int] | Literal["all", "latest", "unspecified"]
    params: dict[str, object]
    inputs: list[dict[str, object]]
    outputs: list[dict[str, object]]
    tidy_csv: str | None
    warnings: list[dict[str, object]]
    error: dict[str, object] | None
```

```python
@dataclass(frozen=True)
class IngestRuntimeContract:
    schema_version: str
    mode: Literal["identity_index", "record_create", "full_record_update"]
    input_rows: int
    identity_columns: list[str]
    candidate_index_rows: int
    estimated_memory_bytes: int
    peak_rss_bytes: int | None
    unknown_policy: Literal["error", "drop", "create"]
    write_scope: Literal["label_sidecar", "records", "ledger"]
    warnings: list[dict[str, object]]
```

```python
@dataclass(frozen=True)
class NotebookViewModel:
    schema_version: str
    campaign: dict[str, object]
    status: dict[str, object]
    review_manifest: dict[str, object] | None
    plot_manifests: list[dict[str, object]]
    stale_artifacts: list[dict[str, object]]
    warnings: list[dict[str, object]]
```

```python
@dataclass(frozen=True)
class NotebookCampaignSetViewModel:
    schema_version: str
    campaigns: list[dict[str, object]]
    active_campaign_id: str | None
    campaign_states: dict[str, NotebookViewModel]
    plot_catalog: list[dict[str, object]]
    comparison_tables: dict[str, object]
    warnings: list[dict[str, object]]
```

## 7. Concrete Improvement Areas

### A. OPAL/Study Boundary Cleanup

| Field | Specification |
| --- | --- |
| Problem | Canonical generated notebooks have removed lateral UMAP defaults, but active OPAL dashboard helpers still include UMAP/projection/cluster code. That noncanonical code can be mistaken for OPAL campaign review. |
| Proposed change | Quarantine active UMAP/projection dashboard helpers outside the canonical OPAL review and generated notebook path. Preserve only generic X-column provenance: `x_column`, `x_dim`, `records_path`, and optional `x_contract`. |
| Contract shape | `campaign.x_provenance = {"x_column": str, "x_dim": int | None, "source_note": str | None}`. No UMAP, cluster, LatentDNA atlas, or producer-specific readiness gates. |
| Affected modules | `src/dnadesign/opal/src/analysis/dashboard/*`, `src/dnadesign/opal/src/analysis/notebook_template.py`, notebook tests, reference notebook docs. |
| Migration notes | If UMAP dashboard code is still needed, move it to an archived/noncanonical namespace or a producer/study-owned package. Add tests that canonical OPAL notebook/review surfaces contain no `LatentDNA`, `UMAP`, `projection`, `cluster__ldn`, or `DenseGen visual` strings except in boundary docs. |
| Acceptance criteria | Generated notebooks and `opal review` mention only campaign contracts, X column provenance, ledgers, progress, selection, labels, predictions, plots, and limitations. Study/probe code remains free to link OPAL review artifacts from study-owned reports. |
| Tests | Snapshot tests for generated notebook, review Markdown, review HTML, and public JSON. Static grep/lint guard over canonical OPAL surfaces. |

Campaign configuration ownership rule:

- OPAL package docs may include generic examples, schema fixtures, and reusable templates.
- Study-specific live campaigns that encode real dataset IDs, biological setpoints, stress axes, or study route decisions should live with the owning study unless they are explicitly marked as external study fixtures.
- If existing stress campaign configs are moved, move them through a visible migration: update docs, tests, CLI examples, and campaign discovery. Do not leave hidden compatibility shims or duplicate canonical paths.
- If they temporarily remain under `src/dnadesign/opal/campaigns/`, add ownership metadata such as `owner_scope: study_fixture`, `study_id`, `dataset_id`, and `portable: false` so future agents do not mistake them for OPAL-core semantics.

Initial implementation status: the checked-in stress ethanol/ciprofloxacin OPAL
configs now carry a strict `ownership:` block with `owner_scope: study_fixture`,
`study_id: stress_ethanol_cipro_growth`,
`dataset_id: usr_prom_eth_cip_opal_candidates`, and `portable: false`. The
visible path migration remains future work.

### B. Public/Private API Boundary

| Field | Specification |
| --- | --- |
| Problem | The first public API cleanup is in place, but the generated notebook still pulls many low-level helper functions directly. Durable notebooks should depend on a smaller public view-model/component API, not a broad helper grab-bag. |
| Proposed change | Keep `dnadesign.opal` public APIs narrow and add explicit notebook/reporting adapter functions for view-model construction and rendering. Generated notebooks should import only public APIs and general third-party packages. |
| Contract shape | Public exports: `build_campaign_progress`, `build_campaign_review`, `read_campaign_predictions`, `load_review_manifest`, `list_plot_kinds`, `describe_plot_kind`, `load_plot_artifact_manifest`, `load_plot_manifest_index`, `build_notebook_view_model`, `render_campaign_notebook`, `smoke_check_notebook`, `render_campaign_progress_text`. |
| Affected modules | `src/dnadesign/opal/__init__.py`, `src/dnadesign/opal/src/analysis/notebook_template.py`, `src/dnadesign/opal/src/reporting/*`, `src/dnadesign/opal/src/plots/*`. |
| Migration notes | Keep internal modules intact. Add thin public adapters rather than relocating large internals. Generated notebooks should pin the public schema versions they expect. |
| Acceptance criteria | No generated notebook imports from `dnadesign.opal.src.*`; notebook render helpers are documented public contracts; study packages use only `dnadesign.opal` public helpers. |
| Tests | Public import tests, generated notebook text tests, architecture boundary check, study probe import tests, and contract tests for the notebook view-model schema. |

### C. Ingest Runtime Memory Safety

| Field | Specification |
| --- | --- |
| Problem | `opal ingest-y` has the correct fixed-sidecar identity-frame path, but that memory-safety posture needs to remain a visible runtime contract rather than an implementation detail buried in tests. |
| Proposed change | Keep the memory-safe ingest plan first-class. For fixed-universe sidecars and `--unknown-sequences error/drop`, load only an identity frame (`id`, `sequence`, and minimal required metadata). Reserve full-record materialization for explicit record-creation or record-update modes that truly need it. Emit ingest memory estimates, phase telemetry, and peak RSS where the platform exposes it. |
| Contract shape | `IngestRuntimeContract = {schema_version, mode, input_rows, identity_columns, candidate_index_rows, estimated_memory_bytes, peak_rss_bytes, unknown_policy, write_scope, warnings}`. `mode=identity_index` is the default for shared sidecar label appends. |
| Affected modules | `src/dnadesign/opal/src/cli/commands/ingest_y.py`, `src/dnadesign/opal/src/storage/records_io.py`, `src/dnadesign/opal/src/storage/data_access.py`, `src/dnadesign/opal/src/storage/label_sources.py`, `src/dnadesign/opal/src/runtime/memory_guard.py`, ingest CLI tests. |
| Migration notes | Do not add a compatibility shim that silently falls back to `store.load()`. If the narrow identity frame cannot be built, fail with an `IngestContractError` explaining the missing columns or unsupported mode. If `unknown_sequences=create` is requested for a fixed `usr_sidecar`, fail before any full-record load. |
| Acceptance criteria | A full-pool six-row label append for a fixed sidecar succeeds without calling `RecordsStore.load()`, reports `mode=identity_index`, emits estimated memory and optional peak RSS, and writes the same label-sidecar result as the old path. Repeated 12-round dogfood can complete rounds without application-memory pressure from ingest. |
| Tests | Monkeypatch `RecordsStore.load()` to fail and assert `opal ingest-y` succeeds for `usr_sidecar + unknown_sequences=error/drop`; assert `unknown_sequences=create` fails before full load for shared sidecars; fixture benchmark verifies peak RSS or estimated memory stays bounded; JSON snapshot covers `IngestRuntimeContract`. |

Implementation guidance:

- Add a small storage helper such as `RecordsStore.load_ingest_identity_frame(required_columns: Iterable[str])`.
- Build `known_ids` and `sequence -> id` from the identity frame, not the full record table.
- Include `bio_type` and `alphabet` only when new-row creation or required-column validation needs them.
- Include the X column only in explicit record-creation/update modes that must write records, never for fixed-universe sidecar appends.
- Keep sidecar append rewrite optimization lower priority until sidecars are large enough to dominate runtime.

Initial implementation status: fixed-sidecar ingest now uses
`RecordsStore.load_ingest_identity_frame()`, emits `opal.ingest_runtime.v1` in
JSON, and prints a `[Runtime] ingest-y` text block with mode, write scope,
loaded columns, candidate index rows, estimated frame memory, optional peak RSS,
unknown-label policy, and full-record/X-column load status. Remaining work in
this section is configurable hard thresholds, broader memory regression fixtures,
and any future explicit record-create/update ingest modes.

### D. Leakage And Contamination Fail-Fast

| Field | Specification |
| --- | --- |
| Problem | Leakage checks currently live partly in study-owned probe contracts and partly in generic OPAL validation. Invalid evidence can come from train/eval overlap, selected IDs outside the eval universe, duplicate prediction IDs, malformed labels, label-source drift, or study-specific forbidden inputs. |
| Proposed change | Define a generic OPAL leakage/contamination guard and keep study-specific source policies in study code. OPAL validates generic AL invariants; probes validate biological/synthetic-oracle constraints. Both fail closed for execution and report structured violations for read-only review. |
| Contract shape | `LeakageGuardContract = {schema_version, owner, scope, checks, status, violations, severity}`. Generic checks include train/eval disjointness, selected-in-eval, unique prediction IDs, label-source/y-space match, run_id scope, and no prediction-ledger reuse as labels unless explicitly configured. Study checks include forbidden input columns, synthetic-label scratch isolation, null-label provenance, and split manifest integrity. |
| Affected modules | OPAL runtime/evaluation/review guards, study-owned probe source-contract and decision modules, review/status JSON, tests. |
| Migration notes | Do not teach OPAL DenseGen, cipro, ethanol, or SFXI biological rules. OPAL owns generic contamination states; the study probe owns source-policy names and biological forbidden-input lists. |
| Acceptance criteria | Any duplicate prediction IDs, selected IDs outside eval, train/eval overlap, malformed y-space, or forbidden study input produces an error before a PASS-like decision or review success. Read-only status surfaces show `leakage.status` and `violations` instead of burying these checks in prose. |
| Tests | Generic OPAL fixtures for train/eval overlap and duplicate predictions; study probe fixtures for forbidden input columns and synthetic-label leakage; review/status JSON snapshots; no smoothing/truncation behavior in metrics. |

Initial implementation status: OPAL now has a generic `opal.leakage_guard.v1`
contract for shared-sidecar record contamination and train/eval overlap after
configured labeled-candidate exclusion. `verify-outputs` and campaign review
now reject duplicate run-scoped prediction IDs, and `verify-outputs` rejects
selected IDs outside the run-scoped prediction/eval evidence through the same
leakage-guard contract. Remaining checks in this section are broader review/status
JSON snapshots, raw-vs-derived metrics separation, and study-owned forbidden-input
policies.

### E. Run/Progress Semantics

| Field | Specification |
| --- | --- |
| Problem | Event phases and lock inspection exist, but `cmd_run()` still writes command/X-validation/records events before the campaign lock and before a `run_id` exists. Progress can distinguish phase counts, but the lock/preflight/run ownership contract is still incomplete. |
| Proposed change | Keep explicit event phases (`command`, `preflight`, `run`, `abort`, `finalize`) and add a preflight attempt ID or lock token before writing events that imply mutation. Emit terminal abort events on operator cancellation and contract failures where possible. |
| Contract shape | `ProgressEventContract = {schema_version, event_id, phase, run_id, round, stage, severity, status, ts, lock_token, message, payload}`. `run_id` is required for `phase=run` after `run_context`. |
| Affected modules | `src/dnadesign/opal/src/cli/commands/run.py`, `src/dnadesign/opal/src/runtime/run_round.py`, `src/dnadesign/opal/src/reporting/progress.py`, `src/dnadesign/opal/src/reporting/summary.py`, `src/dnadesign/opal/src/storage/locks.py`. |
| Migration notes | Preserve old fields while adding `schema_version` and `phase`; readers should accept old logs but label them `legacy_event_contract`. Do not silently discard old preflight events. |
| Acceptance criteria | `opal progress --json` reports top-level `event_contract.*`, `locks.campaign.*`, and per-round `rounds[].summary.run_scope.*`, including `ambiguous_run_scope`, `aborted`, `legacy_events`, `command_events`, `preflight_events`, `run_events`, `abort_events`, and `finalize_events`. `opal review --json` refuses mixed-run review unless run_id is explicit. |
| Tests | Unit tests for aborted prompt, lock conflict, stale lock, multiple starts, multiple run_ids per round, missing done event, and legacy log parsing. CLI JSON snapshots. |

### F. X Contract Unification

| Field | Specification |
| --- | --- |
| Problem | Runtime `identity` accepts scalar/list/JSON string vectors, docs describe Arrow list or JSON string, and public validation requires Parquet `fixed_size_list`. This split weakens fail-fast behavior. |
| Proposed change | Define canonical physical schema as Parquet Arrow `fixed_size_list<float32 or float64>` with finite, non-null values and stable row count. Noncanonical forms are allowed only through explicit import/normalization commands, never inside campaign execution. |
| Contract shape | `XMatrixContract` with `physical_type`, `x_dim`, `canonical=true`, `validation_level=parquet_schema_and_values`, and `normalization_source` if converted. |
| Affected modules | `src/dnadesign/opal/src/storage/x_contracts.py`, `src/dnadesign/opal/src/transforms_x/identity.py`, `src/dnadesign/opal/src/runtime/round/stages.py`, `src/dnadesign/opal/docs/reference/data-contracts.md`, validate/init/run/explain paths. |
| Migration notes | Keep `identity` as a model-matrix transform, but require candidate records to validate before run/explain/review. Add `opal x normalize` or `opal import-records` if scalar/list/JSON compatibility is needed for legacy inputs. |
| Acceptance criteria | `opal run`, `opal explain`, `opal review`, and notebook view model all fail or warn on invalid/noncanonical X according to severity. Docs no longer describe JSON-string X as runtime-canonical. |
| Tests | Fixed-size list accept; variable list/scalar/JSON reject at campaign execution; normalization command accepts legacy forms and writes canonical records; review exposes x contract. |

### G. Review Manifest Authority And Artifact Gardening

| Field | Specification |
| --- | --- |
| Problem | Existing review manifests can omit plots while stale PNGs remain on disk. Full dogfood roots can also accumulate multi-GB scratch state and stale configured-plot siblings. Notebook/review surfaces that trust directory contents can mislead users, and ignored `.var` evidence can be over-treated as durable. |
| Proposed change | Make `manifest.json`, `review_manifest.json`, and plot manifest indexes authoritative snapshots. Review/notebook readers render only artifacts referenced by active manifests. Extra files under known artifact directories become `StaleArtifactWarning` entries. Add an artifact-gardening surface that inventories run roots, stale siblings, local-only status, byte size, retention policy, and explicit prune plans. |
| Contract shape | `ReviewManifestContract = {schema_version, generated_at, review_scope, campaign, run, progress, selection, plots, stale_artifacts, warnings, artifacts}`. `ArtifactGardenContract = {schema_version, root, local_only, active_manifests, stale_artifacts, bytes_total, retention_policy, prune_plan, warnings}`. |
| Affected modules | `src/dnadesign/opal/src/reporting/review.py`, generated notebook, study probe review wrappers, plot manifest readers, future artifact-audit/prune CLI, tests. |
| Migration notes | Existing manifests without `stale_artifacts` are accepted as v1 and upgraded in memory. Inspection commands never delete. Add explicit `opal artifacts audit` and `opal artifacts prune --dry-run/--apply` or equivalent later; prune must operate from manifests and retention policy, not filename guesses. |
| Acceptance criteria | If `outputs/review/plots/*.png` exists but manifest `plots` is empty, review JSON and notebook show a stale-artifact warning and do not render those PNGs as current evidence. If `_r11` configured-plot siblings exist after `_rall` is active, the top-level review reports them as ignored stale artifacts. Dogfood run summaries label `.var` roots as local-only and record run-root size and file count. |
| Tests | Fixture with manifest empty plus stale files; fixture with missing manifest-referenced file; fixture with status `failed`; fixture with stale configured-plot siblings; artifact-audit dry-run fixture; JSON schema validation. |

Raw run artifacts and derived review artifacts must be separated. A run-completion `metrics.json` or status payload should be immutable execution evidence. Review/report generation may enrich, explain, render, and index those metrics, but it should write derived payloads such as `review_metrics.json`, `review_manifest.json`, and HTML/Markdown. Rerunning review should not silently rewrite raw run metrics unless an explicit migration command is invoked.

### H. Plot Artifactization

| Field | Specification |
| --- | --- |
| Problem | Configured plots now write per-plot manifests, but the manifest contract is not yet strong enough to guarantee interpretability. Some plot metadata lacks data shape, tidy schema, declared failure modes, source freshness, and decision-purpose captions. |
| Proposed change | Treat `PlotArtifactManifest` as the authority for plot cards and downstream review, and require every built-in plot to populate useful metadata. Add stale/fresh state, plot-purpose captions, and manifest schema tests for every registered plot. |
| Contract shape | `PlotArtifactManifest` as defined above. Inputs include path, role, exists, size, mtime, and optional content hash where practical. Outputs include media path, tidy CSV path, format, bytes, mtime, and status. Metadata includes data shape, tidy schema, failure modes, and review purpose. |
| Affected modules | `src/dnadesign/opal/src/plots/runner.py`, `src/dnadesign/opal/src/plots/_context.py`, plot plugins, `src/dnadesign/opal/src/registries/plots.py`, docs/tests. |
| Migration notes | Keep the runner-level manifest wrapper. Harden metadata incrementally by plot family, starting with generic primitives and then SFXI diagnostics. |
| Acceptance criteria | `opal plot` emits one manifest per plot entry, an aggregate `outputs/plots/plot_manifest.json`, and every built-in plot has nonempty `data_shape`, `tidy_schema` when `save_data` is supported, `failure_modes`, `caption` or `review_purpose`, `quality`, and `freshness`. Failed plots write failed manifests with error taxonomy and no ambiguous success. |
| Tests | Success manifest, failed manifest, save_data CSV manifest, missing input failure, stale source detection, aggregate manifest schema, and registry metadata completeness. |

### I. Plot Ontology And New Primitives

OPAL plots should communicate campaign state, not just produce images. The reusable information classes are:

- selection behavior
- round-over-round progress
- model behavior
- feature attribution stability
- uncertainty
- support/OOD distance
- objective geometry
- vector alignment
- diversity and collapse
- label acquisition
- metadata composition
- calibration
- auditability

Current OPAL already has three of the requested generic primitives: `metric_over_rounds`, `feature_importance_heatmap`, and `vector_summary_heatmap`. Treat those as the baseline tracer bullets. The remaining primitives should follow their shape-first pattern rather than cloning study/probe-specific plot kinds.

The future plot ontology should be data-shape based:

| Data shape | Plot family |
| --- | --- |
| scalar over rounds | Metrics, thresholds, score summaries, uncertainty summaries |
| vector over rounds | Mean predicted/observed vectors, setpoint alignment, channel drift |
| matrix heatmap | Feature importance, audit matrix, objective component grids |
| categorical composition | Metadata composition, label composition |
| selected overlap | Jaccard or set overlap across rounds/runs |
| attribution matrix | Feature importances by round/model |
| uncertainty/support distribution | OOD distance, predictive uncertainty, selected percentile |
| objective decomposition | Score component summaries and tradeoffs |
| candidate audit table | Selected rows with rank, score, facets, vector channels, status |
| calibration/agreement | Predicted versus observed agreement after labels arrive |

Campaign-specific semantics such as SFXI setpoints, stress axes, cipro, ethanol, AND, DenseGen, or future studies should configure these primitives. They should not create bespoke plot kinds unless the data shape is genuinely new.

#### Recommended Generic Primitives

| Primitive | Purpose | Required inputs | Optional params | Output artifacts | Tidy CSV shape | Failure modes | Generic rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `metric_over_rounds` | Track a declared numeric metric over rounds | `run_pred` or `run_meta`, `as_of_round`, metric field | `cohort`, `summary`, `quantiles`, `top_k`, `thresholds`, `reference_lines` | line/interval plot, CSV, manifest | `round, cohort, metric, summary, value` | missing metric, nonnumeric values, no rows for cohort, ambiguous run_id | Any campaign can summarize scalar metrics |
| `feature_importance_heatmap` | Show feature attribution changes by round | per-round feature importance artifacts with stable feature IDs | `sort`, `top_n`, `cluster=false`, `side_summaries` | heatmap, optional rank-change panel, CSV, manifest | `round, feature_id, importance, rank` | missing artifacts, unstable feature IDs, duplicate feature IDs, nonfinite importance | Attribution matrix is model/campaign generic |
| `feature_stability_over_rounds` | Summarize attribution stability | feature importance per round | `top_n`, `rank_metric`, `adjacent_only` | line plot or table, CSV, manifest | `round_a, round_b, metric, value` | fewer than two rounds, inconsistent feature universe | Stability is generic model behavior |
| `vector_summary_heatmap` | Compare vector channels across setpoint and rounds | vector field from predictions/labels plus round/run scope | `cohort`, `vector_field`, `channel_labels`, `include_setpoint`, `aggregation` | heatmap, CSV, manifest | `row_type, round, cohort, channel, value` | vector length mismatch, missing setpoint when requested, nonfinite values | Vector-shaped, not SFXI-specific |
| `objective_decomposition_over_rounds` | Track objective components | declared objective component fields in `run_pred` or `run_meta` | `components`, `summary`, `cohort`, `quantiles` | multi-line/facet plot, CSV, manifest | `round, cohort, component, summary, value` | missing component, mixed objective schema, ambiguous run_id | Objective components are declared channels/diagnostics |
| `selection_overlap_heatmap` | Detect churn, lock-in, or collapse | selected candidate IDs by round/run | `metric=jaccard|count`, `cohort=selected|top_k` | heatmap, CSV, manifest | `round_a, round_b, cohort, metric, value` | missing selected flags, fewer than two rounds | Set overlap is campaign agnostic |
| `selection_composition_over_rounds` | Show metadata composition of selected/top-k rows | predictions joined to records metadata | `metadata_field`, `cohort`, `normalize`, `top_categories` | stacked bars or heatmap, CSV, manifest | `round, cohort, metadata_field, category, count, fraction` | missing metadata, high cardinality without top_categories | Metadata facets configure the primitive |
| `score_distribution_over_rounds` | Show score distribution and selected position | prediction score field, selected flag | `cohort`, `summary`, `plot=box|violin|ridge` | distribution plot, CSV, manifest | `round, cohort, metric, value` or summary table | missing score, too few rows for distribution | Score distributions are universal AL diagnostics |
| `uncertainty_over_rounds` | Track uncertainty behavior | uncertainty channel or field | `cohort`, `percentile`, `summary`, `selected_percentile` | distribution/line plot, CSV, manifest | `round, cohort, uncertainty_field, summary, value` | missing uncertainty channel, invalid negative uncertainty | Useful for any uncertainty-aware model/objective |
| `support_distance_over_rounds` | Track selected distance to labeled support | support/OOD distance field or plugin output | `cohort`, `distance_field`, `summary` | distribution/line plot, CSV, manifest | `round, cohort, distance_field, summary, value` | missing support field, inconsistent basis | OOD support is generic model trust evidence |
| `label_acquisition_over_rounds` | Show label counts and coverage | labels ledger, records metadata | `metadata_field`, `y_space`, `cumulative` | count/composition plot, CSV, manifest | `round, metadata_field, category, count, cumulative_count` | missing labels ledger, missing round column | Label acquisition is core AL state |
| `tradeoff_frontier` | Show objective component tradeoffs | two or more numeric fields | `x`, `y`, `color`, `selected_only`, `pareto` | scatter/frontier plot, CSV, manifest | `id, round, x_metric, y_metric, x_value, y_value, selected` | missing fields, nonnumeric fields, no selected rows | Any campaign can trade off declared metrics |
| `calibration_over_rounds` | Compare prediction and later observations | predictions joined to observed labels by ID and round | `metric`, `bins`, `lag_policy`, `cohort` | calibration curve, agreement plot, CSV, manifest | `round, bin, predicted_mean, observed_mean, count, metric` | no later labels, ID mismatch, incompatible y space | Calibration is generic once labels arrive |
| `candidate_audit_matrix` | Compact review table for selected candidates | predictions, records, labels optional | `columns`, `include_vectors`, `max_rows`, `metadata_fields` | HTML/CSV/table artifact, manifest | one row per candidate with declared columns | missing requested columns, too many vector channels without explicit max | Auditable selected candidates are universal |

#### Required Example: `feature_importance_heatmap`

Rows are stable feature IDs. Columns are rounds. Values are importances. The default preserves feature identity and input order; clustering is optional and off by default because clustering can obscure which feature changed. Optional side summaries:

- rank change by feature
- top-N turnover between adjacent rounds
- adjacent-round Spearman correlation

Acceptance criteria:

- Fails if feature IDs are missing, duplicated, or inconsistent unless `allow_missing_features` is explicit.
- Emits tidy CSV: `round, feature_id, importance, rank, source_path`.
- Manifest records model kind, feature importance source files, run/round scope, and ordering policy.

#### Required Example: `metric_over_rounds`

X is round. Y is a declared numeric field such as `pred__score_selected`, `obj__logic_fidelity`, `obj__effect_scaled`, or any future declared metric. Supported cohorts:

- selected
- top-k
- all-pool
- labels

Supported summaries:

- mean
- median
- quantiles
- count
- threshold/reference lines

Tidy CSV fields: `round, cohort, metric, summary, value`.

#### Required Example: `vector_summary_heatmap`

Rows include setpoint first when configured, then chronological rounds. Columns are vector channels. Values can be:

- mean selected predicted vector
- observed vector summary
- top-k predicted vector
- all-pool predicted vector
- labeled vector summary

SFXI may configure semantic labels for channels, but the primitive remains vector-shaped. The manifest should record vector length, channel labels, cohort, aggregation, and source field.

### J. Marimo Notebook UX

| Field | Specification |
| --- | --- |
| Problem | The generated single-campaign notebook is useful and manifest-backed, but it is still mostly a large campaign-local artifact. The first plot-gallery helper now lives in `notebook_components.py`; campaign-set generation now exists as a separate overview notebook. Remaining monolith risk is in the single-campaign template and missing reusable panel primitives. |
| Proposed change | Continue refactoring generated marimo notebooks into thin composition files built from reusable public OPAL primitives. Single-campaign and campaign-set notebooks should use the same component vocabulary; campaign-set behavior is a generic campaign selector plus the same plot gallery, status panels, and raw artifact panels, not a probe-specific notebook. Extend campaign-set review from repeated config paths toward an optional manifest/index input when the contract stabilizes. |
| Contract shape | `NotebookViewModel` plus `NotebookCampaignSetViewModel`. Public render primitives: `campaign_selector`, `plot_selector`, `at_a_glance_panel`, `validity_panel`, `changes_panel`, `metric_definitions_panel`, `plot_card`, `plot_gallery`, `records_panel`, `labels_predictions_panel`, `artifact_garden_panel`, `distrust_panel`, and `raw_artifacts_panel`. Sections: `At a glance`, `Validity`, `Changes`, `Evidence`, `Metric definitions`, `Distrust and limitations`, `Artifacts`, `Raw tables`. |
| Affected modules | `src/dnadesign/opal/src/analysis/notebook_template.py`, `src/dnadesign/opal/src/reporting/notebook.py`, public notebook API, tests, CLI notebook command. |
| Migration notes | Preserve current ability to generate before first run. In that state, the view model reports `not_started` and missing manifest states explicitly. Keep the single-campaign path as the default; add campaign-set mode only when the user supplies a campaign index, run-root manifest, or explicit repeated `--campaign` values. |
| Acceptance criteria | First viewport shows campaign selector when multiple campaigns exist, campaign status, run scope, X column, label source, latest run_id, selected count, stale warnings, and missing artifacts. Heavy tables and plots are inside `mo.accordion(..., multiple=True, lazy=True)`. The generated notebook should be mostly wiring; reusable component/view-model code should be importable and unit-tested outside marimo. |
| Tests | `marimo check` smoke, generated Python parse, no private imports, no UMAP/LatentDNA strings, manifest-only plot rendering, missing/stale plot states, component-level unit tests, campaign-set fixture with two campaigns, and a text guard against re-growing a large all-in-one template. |

Notebook design rule: the user path is "Is it valid?", "What changed?", "What visual evidence supports that?", and "What should I distrust?" Plot cards must show source data, params, status, stale/fresh state, generated time, media links, and tidy CSV links. This is a product UX principle, not a static art direction: the visual language should be quiet evidence cartography, where status, scope, and distrust are visible before detailed tables.

Campaign and plot dropdowns are both scoping controls. A campaign dropdown should select the active `NotebookViewModel`; a plot dropdown should select a manifest-backed plot card or gallery subset inside that active campaign. The same primitives should also support static review pages and JSON-backed dashboards where practical. Avoid duplicating notebook logic for "campaign-set" and "single-campaign" beyond the thin composition shell.

Initial implementation status: generated single-campaign notebooks now use a
schema-pruned records preview instead of loading the full X payload, and
single-campaign plus campaign-set templates share public notebook primitives for
campaign summary rows, at-a-glance panels, evidence rows, plot galleries, and
plot card detail lines. The next slice added shared metric-definition tables and
artifact-garden panels backed by manifest metadata and dry-run prune plans.
This slice adds shared validity panels, progress-derived change tables, and a
generated-template size guard so future notebook UX work cannot quietly rebuild
a large all-in-one template. Remaining work in this section is manifest/index
inputs for campaign sets and richer visual treatment of validity/change state.

**Review Surface Philosophy: Evidence Cartography.** OPAL review surfaces should feel like maps of campaign evidence, not galleries of disconnected artifacts. Space should be allocated by decision value: state, scope, validity, warnings, and current selection evidence occupy the first viewport; raw ledgers, method details, and large tables sit behind lazy sections.

Color and material should encode contract state, not campaign branding. Use restrained status colors for ok/attention/error/stale, neutral plot framing, and consistent manifest badges. A plot card should make its authority obvious through manifest status and source links before visual ornament.

Scale and rhythm should support repeated operator use. Dense tables are acceptable when they are sortable, paged, and scoped; oversized hero-like visual blocks are not. Round, run, campaign, and plot controls should stay near the evidence they scope so the notebook reads as an inspection tool rather than a long report.

Hierarchy should make distrust explicit. Limitations, stale artifacts, missing labels, weak statistics, and ambiguous run scope should be visible as first-class panels, not footnotes. This keeps OPAL honest when synthetic dogfood is mechanically healthy but scientifically underpowered.

### K. CLI UX And Machine Surfaces

| Surface | Proposed contract |
| --- | --- |
| `opal progress --json` | Expose `schema_version`, top-level `event_contract.*`, per-round `rounds[].summary.run_scope.*`, `warnings`, `locks.campaign.*`, manifest-authoritative `artifact_garden` and `stale_artifacts`, legacy-event accounting, phase counts, and stable JSON error categories. |
| `opal review --json` | Return `ReviewManifestContract` plus write paths. Refuse ambiguous run scope unless `--run-id` is explicit. Include stale-file warnings. |
| `opal status` | Become the operator summary over state, latest run, lock state, manifest freshness, and next safe commands. |
| `opal runs` | Add JSON schemas for `runs list` and `runs show`; expose run status, done/aborted state, artifact manifest refs, and duplicate/compaction warnings. |
| `opal ingest-y --json` | Emit `IngestRuntimeContract`, including ingest mode, identity columns loaded, input rows, estimated memory, optional peak RSS, unknown policy, write scope, and fail-fast leakage/contract violations. |
| `opal plot --list` | Text and JSON already exist; keep JSON schema `opal.plot_registry.v1` stable and fill metadata gaps for every built-in plot. |
| `opal plot --list-config --json` | Return structured configured-plot objects (`name`, `kind`, `enabled`, `tags`, optional `preset`) instead of display strings. |
| `opal plot --describe` | Include required data shape, tidy CSV schema, output manifest schema, and failure modes. |
| `opal notebook generate` | Supports repeated `--campaign` options for campaign-set notebooks. Future work: accept an explicit campaign-set manifest/index, review manifest overrides, plot manifest overrides, smoke-check-by-default, and JSON summaries. |
| `opal artifacts audit/prune` | Existing explicit gardening surface for stale files, ignored run roots, manifest authority, byte counts, retention policy, and dry-run/apply pruning. Inspection is read-only; pruning requires `--apply`. |

All JSON errors should have:

```json
{
  "ok": false,
  "error": {
    "category": "RunScopeAmbiguityError",
    "message": "...",
    "exit_code": 2,
    "hints": ["pass --run-id ..."]
  }
}
```

### L. Error Taxonomy And Fail-Fast Behavior

| Category | When raised | Severity |
| --- | --- | --- |
| `ConfigContractError` | Unknown config keys, duplicate YAML keys, unknown plugins, invalid params, incompatible shared-label config | Error |
| `XContractError` | Missing X column, noncanonical physical schema, null/nonfinite/ragged X, mismatched X dim | Error for run/review; warning only for pre-run status if records missing |
| `LabelSourceError` | Missing configured sidecar for run, malformed labels, unknown candidate IDs, y_space mismatch | Error |
| `IngestContractError` | Ingest path cannot build a narrow identity frame, unknown/create policy is incompatible with fixed sidecar, or memory estimate exceeds configured threshold | Error |
| `LeakageContractError` | Train/eval overlap, selected IDs outside eval, duplicate prediction IDs, contaminated label/prediction surfaces, or study-owned forbidden inputs | Error for execution and PASS-like decisions |
| `RunScopeAmbiguityError` | Multiple run_id values for a selected round without explicit run_id | Error for review/plots/predictions; warning for listing |
| `ProgressContractError` | Malformed round log, missing terminal state, run_id mismatch | Error or attention depending surface |
| `PlotDataContractError` | Missing required plot input, missing required columns, nonnumeric data, vector length mismatch | Error per plot, aggregate command exit nonzero |
| `ReviewManifestError` | Missing/invalid review manifest, schema mismatch, manifest references missing files | Error for manifest read; warning for stale extras |
| `StaleArtifactWarning` | Files exist in artifact dirs but are absent from active manifest or older than inputs | Warning |
| `ArtifactGardenWarning` | Ignored local run roots, stale siblings, retention drift, or large generated bundles need explicit operator cleanup | Warning unless an apply prune command fails |
| `NotebookViewModelError` | Notebook generation cannot build a valid view model | Error |
| `PublicApiBoundaryError` | Tests detect generated/study imports from private OPAL internals | Test/lint failure |

Fail-fast policy:

- Execution commands fail on invalid contracts.
- Inspection commands may return `attention` with structured warnings when a campaign has not started.
- Stale artifacts never silently render as current evidence.
- Explicit repair/prune commands may mutate artifacts; inspection commands do not.

### M. Testing And Quality Gates

Required test coverage:

| Area | Tests |
| --- | --- |
| Strict config behavior | Unknown top-level keys, duplicate YAML keys, unknown plugins, invalid plugin params, shared sidecar incompatibilities |
| Ingest memory safety | Fixed-sidecar ingest uses identity-frame column pruning, never calls full `RecordsStore.load()` in identity mode, reports memory estimate/peak RSS, and fails before full load for unsupported create/update modes |
| Leakage and contamination | Train/eval overlap, duplicate prediction IDs, selected IDs outside eval, y_space mismatch, forbidden study input columns, and scratch-label leakage all fail fast |
| X schema validation | Accept fixed-size list; reject variable list, scalar, JSON string, nulls, nonfinite values; normalization command writes canonical schema |
| Run_id-scoped progress | Multiple runs per round, explicit run_id filter, missing run_id, legacy events, latest selector |
| Interrupted/aborted runs | Prompt abort, exception during records load, exception after run_context, lock conflict, stale lock |
| Review manifest schema | Valid schema, missing referenced files, stale extras, no-plots with old PNGs, JSON schema version |
| Artifact gardening | Stale sibling detection, ignored `.var` local-only reporting, byte-count inventory, prune dry-run output, and refusal to delete without explicit apply |
| Plot plugin metadata | Every registered plot has `PlotMeta`, required inputs, output data shape, failure modes, and `--describe` coverage |
| Plot tidy CSV shape | Each plot with `save_data` writes declared tidy schema and manifest references it |
| Marimo notebook | Python parse, `marimo check`, no private imports, no UMAP/LatentDNA residue, manifest-backed plot cards, lazy accordions, component-level helpers, campaign-set dropdown fixture, and template-size regression guard |
| Public API import paths | Study code imports OPAL public API only; generated notebooks use public API only |
| Module decomposition | Size/regrowth guards for notebook templates and study probe review modules; component-level tests cover extracted primitives before deleting old code paths |
| Dogfood | cipro random positive/null, ethanol random, dual/AND random, leave-sigma35 variants, and real-assay workflows only when evidence exists |

Recommended targeted commands during implementation:

```bash
uv run pytest -q src/dnadesign/opal/tests/storage/test_x_contracts.py
uv run pytest -q src/dnadesign/opal/tests/cli -k "ingest"
uv run pytest -q src/dnadesign/opal/tests/reporting
uv run pytest -q src/dnadesign/opal/tests/cli/test_cli_runs_status_log.py
uv run pytest -q src/dnadesign/opal/tests/cli/test_cli_plot.py
uv run pytest -q src/dnadesign/opal/tests/notebooks
uv run ruff check src/dnadesign/opal
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
```

Full repo gates remain the project-local definition of done:

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
uv run pytest -q
uv run python -m dnadesign.devtools.docs.checks
```

## N. Documentation

Update these docs as implementation lands:

| Document | Required updates |
| --- | --- |
| `src/dnadesign/opal/docs/reference/cli.md` | Keep live help and docs aligned for progress/review/runs/status/plot/notebook; document ambiguity errors, stale warnings, campaign-set notebook generation, and JSON error categories. |
| `src/dnadesign/opal/docs/reference/ingest.md` or CLI ingest section | Document column-pruned ingest modes, memory telemetry, unknown-sequence policy, sidecar fixed-universe behavior, and `IngestRuntimeContract`. |
| `src/dnadesign/opal/docs/reference/data-contracts.md` | Keep canonical X physical schema explicit; move any remaining legacy/noncanonical behavior to import/normalization docs. |
| `src/dnadesign/opal/docs/reference/plots.md` | Tighten plot artifact manifests, align documented status enum with implementation, tidy CSV schema expectations, `PlotMeta` requirements, generic plot primitive guide, and per-plot interpretability captions. |
| `src/dnadesign/opal/docs/reference/configuration.md` | Clarify that `plot_config` configures review artifacts but runtime round execution does not depend on plots. |
| `src/dnadesign/opal/docs/reference/review-manifests.md` | Keep campaign review manifest and stale artifact behavior aligned with implementation. |
| `src/dnadesign/opal/docs/reference/artifacts.md` | Document artifact gardening, stale artifact semantics, local-only `.var` evidence, retention policy, and prune dry-run/apply behavior if that surface is added. |
| `src/dnadesign/opal/docs/reference/notebooks.md` | Extend generated notebook contract for reusable marimo primitives, campaign-set view models, public imports, smoke checks, and no representation-browser boundary. |
| `docs/studies/stress_ethanol_cipro_growth/contexts/opal/*` | Keep study/probe benchmark docs outside OPAL core and remove wording that generalizes cipro/random dogfood. |

## O. Migration Plan

| Phase | Scope | Dependencies | Risk | Acceptance criteria | Rollback/migration concerns |
| --- | --- | --- | --- | --- | --- |
| P0a: Ingest runtime safety | Column-pruned sidecar ingest, ingest memory estimates/telemetry, fail-fast unsupported create/update modes | None | Medium; ingest touches operator workflows | Full-pool fixed-sidecar label append avoids full records load and emits `IngestRuntimeContract` | Keep old implementation only as test fixture reference, not a hidden fallback |
| P0b: Leakage and raw/derived evidence guard | Add generic leakage checks, keep study-specific forbidden-input checks study-owned, split raw run metrics from derived review metrics | P0a independent | Medium; failures may expose previously smoothed invalid states | PASS-like decisions impossible with leakage violations; review reruns do not mutate raw metrics | Provide explicit migration for old report payloads; do not rewrite silently |
| P0c: Artifact gardening | Stale sibling inventory, local-only run-root labeling, byte counts, prune dry-run/apply contract | None | Low/medium; deletion commands require care | Review/notebook surfaces show stale ignored artifacts; prune requires explicit apply | Inspection commands remain read-only |
| P1: Run/progress/review contract hardening | Complete attempt_id/run_id, lock/preflight/run contract, abort events, run ambiguity JSON, stale artifact warnings in review | P0b helpful | Medium; old logs exist | Progress/review JSON schemas include warnings and refuse mixed-run review | Readers accept old logs with `legacy_event_contract` warnings |
| P2: X contract unification | Enforce canonical fixed-size list at run/review; add normalization/import path for legacy X | P1 helpful | Medium/high; existing fixtures may use legacy X | Runtime, validator, docs, and tests agree on canonical schema | Provide explicit migration command; do not silently coerce during run |
| P3: Plot artifact manifest hardening | Complete metadata, stale/fresh state, interpretability captions, registry completeness, and all-round/default semantics for multi-round plots | P0c/P1 | Medium; all plot plugins affected indirectly | Every configured plot produces written/failed/skipped manifest with meaningful metadata | Runner-level manifest wrapper minimizes plugin edits |
| P4: Generic plot primitives | Extend beyond existing `metric_over_rounds`, `feature_importance_heatmap`, and `vector_summary_heatmap` into overlap/composition/uncertainty/support/calibration primitives | P3 | Medium; temptation to over-abstract | New primitives pass data-shape contracts and avoid study names | Keep SFXI-specific plots during transition as configured diagnostics |
| P5: Marimo review notebook modernization | Public component primitives shared by single-campaign and campaign-set notebooks, manifest-backed plot cards, campaign/plot dropdowns, lazy accordions, smoke checks | P3 | Medium; generated artifacts are durable | Notebook uses public imports and manifest authority; related campaigns are navigable from one generated notebook without duplicating implementation | Keep behavior stable through explicit view-model versioning; avoid legacy template shims |
| P6: Campaign information architecture | Decide and execute ownership model for study-specific campaign configs: move to studies or mark as study fixtures with metadata | P1-P5 not required but helpful | Medium; path churn affects docs/tests | OPAL package contains generic configs/templates; study configs are study-owned or explicitly marked | Use explicit docs/tests migration, not hidden compatibility paths |
| P7: Module decomposition | Split `ingest_y.py`, probe `review.py`, probe `decision.py`, and notebook templates by contract | Can proceed slice-by-slice | Medium; review churn | Smaller modules have focused tests and no behavior drift | Extract helpers under tests before deleting old paths |
| P8: Dogfood expansion and CI gates | Add broader synthetic and real-data smoke coverage with explicit evidence labels, including multi-round synthetic probe loops owned by study code | P0-P7 depending gate | High; runtime cost and study availability | Reports distinguish cipro/random, random-all, leave-sigma35, multi-round synthetic pressure tests, and real assay evidence | Keep expensive gates optional/nightly; never block core OPAL unit tests on study-specific benchmarks |

## P. Risks And Tradeoffs

| Risk | Tradeoff and mitigation |
| --- | --- |
| Ingest optimization changes label behavior | Treat behavior parity as mandatory. Add tests that compare sidecar outputs before/after column-pruned ingest and fail if fixed-universe semantics change. |
| Memory telemetry becomes platform-specific | Use estimates everywhere and peak RSS opportunistically. Missing RSS should be `unknown`, not a failure, unless a benchmark gate explicitly requires it. |
| Over-abstracting plot interfaces | Keep plot contracts small: metadata, required inputs, tidy schema, manifest. Do not require a universal plotting superclass. |
| Breaking existing campaigns | Stage warnings before errors where safe for inspection surfaces; execution contracts should still fail fast. Provide explicit normalization/migration commands. |
| Stale artifacts misleading users | Manifest-first rendering and stale warnings are worth the extra manifest plumbing. Do not auto-delete without an explicit prune command. |
| Artifact pruning deletes useful local evidence | Make audit read-only by default, require `--apply` for pruning, and preserve tracked summary artifacts before deleting ignored local run roots. |
| Notebook durability under refactor | Public view-model APIs reduce brittleness, but they become semver-like contracts. Keep them narrow. |
| Campaign-set notebook becomes a second monolith | Share primitives with single-campaign notebooks. Campaign dropdowns should select a view model; they should not fork the rendering implementation. |
| Moving campaign configs breaks operator muscle memory | Prefer explicit owner metadata first if relocation is not yet decided. If moving, update docs/tests/CLI examples in one visible migration and avoid hidden compatibility shims. |
| Single-host locks in shared environments | Document local-only lock scope and surface active/stale lock state. Do not imply distributed safety without a real lease. |
| Synthetic-oracle dogfood overclaim | Treat current cipro/random pass as one scoped scenario. Broader claims require ethanol, dual/AND, leave-sigma35, and eventually real assay evidence. |
| SFXI-specific plots crowding generic ontology | Keep existing SFXI diagnostics, but make new multi-round/progress plots shape-based and configurable. |
| Manifest hashes increasing cost | Prefer mtime/size by default and content hashes only for small or critical files unless configured. |

## Q. Open Questions

1. Should the noncanonical X normalization command live under `opal x normalize`, `opal import-records`, or `opal validate --repair --out`? Recommendation: `opal x normalize` because mutation is explicit and scoped.
2. Should campaign-set notebooks add an explicit manifest/index input in addition to repeated `--campaign` flags? Recommendation: yes, once the campaign-set manifest contract stabilizes; study/probe tools should generate that manifest through OPAL public APIs rather than adding OPAL study-specific discovery.
3. Should study-specific campaign configs move out of `src/dnadesign/opal/campaigns/`, or stay temporarily with explicit `study_fixture` metadata? Recommendation: mark ownership immediately, then move live study configs once campaign discovery and docs can change without hidden shims.
4. Should legacy UMAP dashboard code be deleted, moved under `archived/`, or migrated to a study/producer-owned package? Recommendation: quarantine first, then delete after confirming no canonical command imports it.
5. What stale-artifact threshold should make review fail rather than warn? Recommendation: missing manifest-referenced files are errors; extra unreferenced files are warnings unless a configured artifact budget is exceeded.
6. What memory threshold should make `ingest-y` fail rather than warn? Recommendation: use configurable budgets with conservative defaults, and initially gate on regression benchmarks rather than platform-specific RSS alone.
7. Which dogfood gates are fast enough for PR CI versus nightly/local operator runs? Recommendation: OPAL unit/fixture gates in PR; study/probe dogfood in optional/nightly workflows.

## R. Acceptance Criteria Summary

- [ ] OPAL core imports no study/probe code and canonical OPAL surfaces contain no LatentDNA/UMAP/representation-browser content.
- [ ] Study/probe code depends only on documented OPAL public APIs.
- [x] `opal ingest-y` has a column-pruned fixed-sidecar path, emits `IngestRuntimeContract`, and does not materialize full records for small label appends unless an explicit record-write mode requires it.
- [ ] Leakage and contamination guards fail fast on train/eval overlap, duplicate prediction IDs, selected IDs outside eval, malformed label sources, prediction/label contamination, and study-owned forbidden inputs.
- [ ] `opal progress --json`, `opal review --json`, `opal status --json`, and `opal runs ... --json` expose schema versions, run scope, `event_contract.*`, warnings, ambiguity, aborted/incomplete state, and lock state where relevant.
- [ ] `opal run` progress events distinguish command, preflight, actual run, abort, and finalize phases with attempt IDs before run IDs exist.
- [ ] Raw run metrics/status are not mutated by review/report generation; derived report payloads are separate and manifest-referenced.
- [ ] X validation has one canonical physical schema and runtime execution fails fast on noncanonical X.
- [ ] Review manifests and plot manifests are authoritative; stale files are detected, reported as ignored by active manifest, and never rendered as current evidence.
- [ ] Artifact gardening surfaces inventory local-only run roots, stale siblings, byte counts, retention state, and dry-run/apply prune plans.
- [ ] Every plot plugin declares metadata, required inputs, output data shape, and failure modes.
- [ ] Generic plot primitives cover scalar over rounds, vector over rounds, attribution heatmap, overlap, composition, uncertainty/support, objective decomposition, audit table, and calibration.
- [ ] Generated marimo notebooks use public OPAL APIs, manifest-backed view models, reusable marimo component primitives, lazy accordions, plot cards, campaign/plot dropdowns, metric-definition accordions, artifact-garden panels, and smoke checks.
- [ ] Single-campaign and campaign-set notebooks share component primitives; campaign-set navigation is not a second bespoke template surface.
- [ ] Study-specific campaign config ownership is explicit: configs are either moved under the owning study or marked as study fixtures with owner metadata.
- [ ] Docs are updated for CLI config resolution, data contracts, plot authoring, review manifests, notebooks, public APIs, and study/probe boundary guidance.
- [ ] Tests cover strict config, X contracts, run_id progress, aborts, stale artifacts, plot manifests, notebook generation, public imports, and no active OPAL LatentDNA/UMAP residue.
- [ ] Dogfood evidence is labeled by coverage; cipro/random is not generalized to ethanol, dual/AND, leave-sigma35, or real assay behavior without independent evidence.
