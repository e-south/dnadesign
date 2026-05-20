# OPAL Modernization Development Specification

**Status:** Draft for engineering planning and review
**Audience:** OPAL maintainers, study integrators, and developer-experience owners
**Date:** 2026-05-20
**Scope:** OPAL campaign runtime, reporting, plot artifacts, CLI JSON surfaces, and generated marimo review notebooks
**Chosen path:** `src/dnadesign/opal/docs/maintainers/dev-spec-modernization.md`

The requested path `docs/opal/dev-spec-modernization.md` is not present in the current repository layout. OPAL's checked-in documentation lives under `src/dnadesign/opal/docs/`, with maintainer planning material under `src/dnadesign/opal/docs/maintainers/`. This spec is placed there so it stays in the tool-local OPAL docs tree instead of creating a new root-level docs island.

This document is a specification only. It intentionally does not implement production code.

## 1. Executive Summary

OPAL should become a small, contract-first active-learning campaign runtime with excellent machine-readable reporting and review artifacts. Its core identity is a campaign loop over one candidate table, one explicit X column, one label source, model/objective/selector plugins, append-only ledgers, run-scoped progress, configured plots, static review bundles, and generated marimo notebooks.

The reason to modernize now is not that the round loop is broken. Repository evidence shows OPAL already has strict config loading, channelized objectives, ledger contracts, run-aware review, plot registries, and public reporting exports. The gap is that the product surface is not yet as contract-authoritative as the runtime: notebook and review surfaces still rely on filesystem discovery in places, the plot output model lacks per-plot manifests, OPAL analysis code still carries UMAP/projection residue, progress events can be ambiguous around preflight/abort/lock boundaries, and the X contract is split between runtime permissiveness and strict Parquet validation.

Highest-priority changes:

| Priority | Change | Outcome |
| --- | --- | --- |
| P0 | Remove active LatentDNA/UMAP/projection residue from canonical OPAL review and generated notebook surfaces | OPAL reviews campaigns, not upstream representation geometry |
| P1 | Harden run/progress/review contracts around run_id, preflight events, aborts, locks, and stale artifacts | Machines and operators can trust status without log archaeology |
| P2 | Unify the X physical schema around fixed-size finite vectors, with explicit import normalization for noncanonical forms | Campaign execution and review fail fast on invalid X |
| P3 | Add per-plot artifact manifests and manifest-first review/notebook rendering | Stale files cannot masquerade as current evidence |
| P4 | Add generic, data-shape-based plot primitives | New campaign diagnostics configure reusable plot kinds instead of bespoke study plots |
| P5 | Replace generated notebooks' private imports and filesystem plot discovery with public OPAL view-model APIs | Notebooks survive internal refactors |
| P6 | Expand dogfood coverage beyond the current cipro/random synthetic-oracle slice | OPAL readiness claims are scoped to evidence |

Explicitly out of scope:

- OPAL will not become a LatentDNA geometry browser, UMAP atlas, DenseGen visualizer, or study-specific benchmark harness.
- OPAL will not own DenseGen synthetic-oracle logic, cipro/ethanol/AND biological interpretation, stress-axis aggregate reports, or scratch-only synthetic labels.
- OPAL will not silently migrate invalid data, guess run scope, or fall back from configured contracts.
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
| `git status --short` | Dirty worktree before this spec; many existing OPAL/study/retron changes and untracked OPAL reporting files were present. This spec must remain a narrow new-doc change. |
| `uv run opal --help` | Passed; CLI includes `progress`, `review`, `plot`, `notebook`, `status`, and `runs`. |
| `uv run opal progress --help` | Passed; accepts `--config`, `--round`, and `--json/--text`. |
| `uv run opal review --help` | Passed; accepts `--run-id`, `--plots/--no-plots`, and JSON output. |
| `uv run opal plot --help` | Passed; supports `--list`, `--list-config`, `--describe`, `--round`, `--run-id`, `--name`, and tags. |
| `uv run opal plot --list` | Passed; listed 10 registered plot kinds. |
| `uv run opal notebook --help` | Passed; has `generate` and `run`. |
| `uv run opal status --help` | Passed; supports `--with-ledger` and JSON. |
| `uv run opal runs --help` | Passed; has `list` and `show`. |
| `uv run opal notebook generate --help` | Passed; supports directory-capable config, `--round`, `--out`, `--name`, `--force`, and validation. |

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

### Public APIs

The public package exports are intentionally small and useful for study code. This is the correct direction. The generated notebook template still imports from `dnadesign.opal.src.*`, which is acceptable for package-owned source at generation time but brittle for durable generated artifacts and user notebooks. Future generated notebooks should import public view-model/reporting APIs from `dnadesign.opal`.

### CLI Progress And Review

`build_campaign_progress()` returns `opal.campaign_progress.v1` with generated time, campaign identity, state path, selector, status, round count, and per-round summaries. `build_campaign_review()` returns and writes `opal.campaign_review.v1`, including campaign metadata, review scope, run summary, progress summary, selection preview, plots, and artifact paths.

The review manifest is a good start, but the current review and notebook surfaces do not fully enforce manifest authority. Current generated notebook code lists `outputs/plots/*.png` and picks the latest filename prefix match for configured plot names. Current review can emit `plots: []` while old plot PNGs still exist on disk.

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
| `scatter_score_vs_rank` | Score versus rank with selected candidates highlighted |
| `percent_high_activity_over_rounds` | Thresholded score progress over rounds |
| `fold_change_vs_logic_fidelity` | SFXI score/effect versus logic fidelity diagnostics |
| `sfxi_factorial_effects` | Factorial-effects map from predicted logic vectors |
| `sfxi_setpoint_sweep` | Setpoint sweep over labels |
| `sfxi_support_diagnostics` | Distance to labeled logic support versus score |
| `sfxi_uncertainty` | Uncertainty versus score/effect diagnostics |
| `sfxi_intensity_scaling` | Denominator, clipping, and raw-effect scaling diagnostics |
| `sfxi_logic_fidelity_closeness` | Observed label closeness to setpoint |

The main plot gap is artifactization. Plots write images and optional tidy CSVs, but do not write a per-plot manifest recording parameters, inputs, run scope, status, source freshness, output files, and schema version.

### Notebook Generation

The generated marimo notebook is campaign-specific and covers records, ledgers, selected records, labels, predictions, configured plot deliverables, and CLI handoff. It already uses accordions and degrades when runs are missing. However, it imports private OPAL modules, carries optional projection/UMAP language, and discovers plot files by directory listing rather than manifest. That should change.

### Study/Probe Separation

The stress study route correctly describes OPAL as consuming the `usr_prom_eth_cip_opal_candidates` candidate feature table with explicit X column, while LatentDNA owns prior X selection and study code owns pre-assay batch-0/probe logic. The DenseGen axis probe is study-owned, exports no package-root compatibility API, uses scratch-only synthetic labels, and calls public OPAL progress/review APIs.

The current cipro/random DenseGen dogfood run is useful but narrow. Its manifest reports `PASS_CIPRO_RANDOM_GATE` and explicitly omits ethanol, dual, and leave-sigma35 coverage. Do not generalize it to real assay behavior or broader OPAL readiness.

### Known Issues And Risks

| Issue | Risk |
| --- | --- |
| Active OPAL analysis/notebook helpers still mention projection, UMAP, and LatentDNA as optional context | Boundary drift: OPAL review can feel like a representation-browser surface |
| Generated notebooks import `dnadesign.opal.src.*` | Durable notebooks break under internal refactor |
| Notebook plot gallery discovers `outputs/plots/*.png` and filename prefixes | Stale plot files can appear current |
| Review manifests can state `plots: []` while prior PNG files still exist | Review evidence can be misleading |
| `cmd_run()` logs early command/records events before lock and before run_id | Progress can mix preflight, abort, and run events |
| X docs allow `list<float>` or JSON strings, identity transform accepts scalar/list/JSON, but strict validator requires Parquet `fixed_size_list` | Runtime and validation disagree on canonical physical schema |
| `CampaignLock` and `PathLock` are local-host locks | Shared/network mutation needs a stronger lease or documented non-support |
| Plot primitives include several SFXI-specific single-round diagnostics | Useful diagnostics, but future plots should be data-shape primitives first |
| CLI docs say `progress --config <yaml-or-dir>` but config-resolution footer omits progress from directory-capable commands | Documentation mismatch |

## Evidence Ledger

| Observation | Evidence |
| --- | --- |
| OPAL intent is explicit active-learning over feature/objective/selection/ledger contracts | `src/dnadesign/opal/README.md:3-4` |
| Documented round lifecycle and runtime surfaces | `src/dnadesign/opal/docs/concepts/architecture.md:9-32` |
| OPAL's documented fail-fast model | `src/dnadesign/opal/docs/concepts/architecture.md:54-63` |
| Config top-level blocks, defaults, shared sidecar policy, and plot config wiring | `src/dnadesign/opal/docs/reference/configuration.md:12-30`, `src/dnadesign/opal/docs/reference/configuration.md:44-62`, `src/dnadesign/opal/docs/reference/configuration.md:94-101`, `src/dnadesign/opal/docs/reference/configuration.md:180-193` |
| Loader forbids unknown fields and validates plugin names and shared sidecar constraints | `src/dnadesign/opal/src/config/loader.py:179-205`, `src/dnadesign/opal/src/config/loader.py:253-291`, `src/dnadesign/opal/src/config/loader.py:315-333` |
| Public OPAL API currently exports config, predictions, progress, review, and X validation helpers | `src/dnadesign/opal/__init__.py:14-27` |
| Data-contract docs still describe X as Arrow list or JSON string while strict validator requires fixed_size_list | `src/dnadesign/opal/docs/reference/data-contracts.md:41-45`, `src/dnadesign/opal/src/storage/x_contracts.py:70-97` |
| Strict X validator tests reject variable-list and scalar physical schemas | `src/dnadesign/opal/tests/storage/test_x_contracts.py:61-90` |
| Runtime identity transform accepts scalar/list/JSON-string inputs | `src/dnadesign/opal/src/transforms_x/identity.py:29-78` |
| CLI run logs pre-lock records events and acquires campaign lock later | `src/dnadesign/opal/src/cli/commands/run.py:78-124` |
| Runtime creates run_id after training and logs later run-scoped stages | `src/dnadesign/opal/src/runtime/run_round.py:119-177`, `src/dnadesign/opal/src/runtime/run_round.py:182-268`, `src/dnadesign/opal/src/runtime/run_round.py:330-380` |
| Progress JSON builder summarizes campaign state and round logs | `src/dnadesign/opal/src/reporting/progress.py:29-58`, `src/dnadesign/opal/src/reporting/progress.py:119-156` |
| Round summary filters by run_id when supplied and slices to latest start when multiple starts exist | `src/dnadesign/opal/src/reporting/summary.py:114-135` |
| Review writer creates schema `opal.campaign_review.v1`, manifest, Markdown, HTML, selection preview, and plot status list | `src/dnadesign/opal/src/reporting/review.py:67-166`, `src/dnadesign/opal/src/reporting/review.py:188-253`, `src/dnadesign/opal/src/reporting/review.py:256-327` |
| Review tests cover manifest path, schema, run_id scope, and run-log mismatch failure | `src/dnadesign/opal/tests/reporting/test_review.py:50-117` |
| Plot docs define PlotContext, plots.yaml, strict params placement, built-in paths, and save_data | `src/dnadesign/opal/docs/reference/plots.md:7-28`, `src/dnadesign/opal/docs/reference/plots.md:83-101`, `src/dnadesign/opal/docs/reference/plots.md:163-178` |
| PlotMeta shape and registry/entry point loading | `src/dnadesign/opal/src/registries/plots.py:27-32`, `src/dnadesign/opal/src/registries/plots.py:40-55`, `src/dnadesign/opal/src/registries/plots.py:63-82` |
| Plot config rejects unknown keys and conflicting inline/external plot config | `src/dnadesign/opal/src/plots/config.py:23-42`, `src/dnadesign/opal/src/plots/config.py:75-78`, `src/dnadesign/opal/src/plots/config.py:161-216` |
| Plot runner injects built-in data paths, builds PlotContext, calls plugin, and only reports pass/fail to terminal | `src/dnadesign/opal/src/plots/runner.py:91-100`, `src/dnadesign/opal/src/plots/runner.py:170-251`, `src/dnadesign/opal/src/plots/runner.py:254-284` |
| Feature importance plot currently discovers per-round files and writes optional tidy CSV | `src/dnadesign/opal/src/plots/feature_importance_bars.py:34-53`, `src/dnadesign/opal/src/plots/feature_importance_bars.py:163-176`, `src/dnadesign/opal/src/plots/feature_importance_bars.py:213-317` |
| Generated notebook imports private OPAL internals | `src/dnadesign/opal/src/analysis/notebook_template.py:41-86` |
| Generated notebook includes optional projection/UMAP boundary text | `src/dnadesign/opal/src/analysis/notebook_template.py:270-285`, `src/dnadesign/opal/src/analysis/campaign_progress.py:22-42`, `src/dnadesign/opal/src/analysis/campaign_progress.py:125-135` |
| Generated notebook plot panel discovers filesystem PNGs by prefix | `src/dnadesign/opal/src/analysis/notebook_template.py:414-438`, `src/dnadesign/opal/src/analysis/notebook_template.py:470-505` |
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
| Current dogfood evidence is cipro/random only and explicitly omits ethanol, dual, and leave-sigma35 | `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/dogfood_cipro_random_progress_20260520T022443Z/reports/review_manifest.json:8-22`, `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/dogfood_cipro_random_progress_20260520T022443Z/reports/metrics.json:24-32`, `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/dogfood_cipro_random_progress_20260520T022443Z/reports/metrics.json:55-63` |
| Current stale artifact case: manifests say no plots but PNG files exist on disk | `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/dogfood_cipro_random_progress_20260520T022443Z/reports/review_manifest.json:24-49`, `.var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/dogfood_cipro_random_progress_20260520T022443Z/scratch_campaigns/cipro_positive_random_id/outputs/review/manifest.json:135-140` |
| CLI docs mismatch: progress usage accepts directory, but config-resolution footer omits progress | `src/dnadesign/opal/docs/reference/cli.md:490-513`, `src/dnadesign/opal/docs/reference/cli.md:780-789` |

## 3. Goals

| Goal | Target outcome |
| --- | --- |
| Maintainability | OPAL remains a small runtime kernel plus narrow extension points; internal modules can be refactored without breaking study code or generated notebooks. |
| Fail-fast contracts | Unknown config keys, ambiguous run selection, missing columns, invalid X, stale artifacts, and mixed-run review produce explicit errors or warnings with stable machine-readable codes. |
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

## 6. Proposed Target Architecture

The target architecture keeps abstractions small. Use dataclasses, protocols, JSON schemas, and registry metadata before introducing heavier frameworks.

### Contract Components

| Component | Responsibility | Minimal shape |
| --- | --- | --- |
| `CampaignConfigContract` | Validated campaign config, config source path, schema version, and plugin refs | Dataclass wrapping `RootConfig` plus `schema_version`, `config_path`, `strict_mode` |
| `XMatrixContract` | Physical and logical X schema for candidate table | `records_path`, `x_column`, `id_column`, `physical_type`, `x_dim`, `row_count`, `normalization_status` |
| `LabelSourceContract` | Label source identity, y space, dedup policy, and write lock semantics | `kind`, `dataset`, `path`, `y_space`, columns, `requires_existing_for_run`, `lock_scope` |
| `RoundRunContract` | One actual run attempt, separated from preflight | `run_id`, `round_index`, `phase`, `started_at`, `completed_at`, `aborted_at`, `status`, `lock_token` |
| `ProgressEventContract` | Structured event stream for preflight and run phases | `schema_version`, `event_id`, `phase`, `run_id`, `stage`, `severity`, `ts`, `payload` |
| `ReviewManifestContract` | Authoritative campaign review bundle | `schema_version`, `review_scope`, `campaign`, `run`, `progress`, `selection`, `plots`, `stale_artifacts`, `warnings` |
| `PlotDataContract` | Plot plugin input declaration and tidy data schema | `kind`, `required_sources`, `required_columns`, `optional_columns`, `tidy_schema`, `failure_modes` |
| `PlotArtifactManifest` | Per-plot output authority | `schema_version`, `plot_id`, `kind`, `params`, `run_id`, `rounds`, `inputs`, `outputs`, `status`, `generated_at`, `stale_state` |
| `NotebookViewModel` | Manifest-backed marimo input surface | `campaign_state`, `review_manifest`, `plot_manifests`, `warnings`, `links`, `tables` |
| `Public Reporting API` | Stable functions for progress, review, predictions, status, manifests | `build_campaign_progress`, `build_campaign_review`, `read_campaign_predictions`, `load_review_manifest`, `inspect_campaign_status` |
| `Public Plot API` | Stable functions for plot metadata and manifests | `list_plot_kinds`, `describe_plot_kind`, `load_plot_artifact_manifest`, `run_configured_plots` |
| `Public Notebook API` | Stable generated-notebook helper surface | `build_notebook_view_model`, `render_campaign_notebook`, `smoke_check_notebook` |

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
class NotebookViewModel:
    schema_version: str
    campaign: dict[str, object]
    status: dict[str, object]
    review_manifest: dict[str, object] | None
    plot_manifests: list[dict[str, object]]
    stale_artifacts: list[dict[str, object]]
    warnings: list[dict[str, object]]
```

## 7. Concrete Improvement Areas

### A. OPAL/Study Boundary Cleanup

| Field | Specification |
| --- | --- |
| Problem | OPAL analysis/notebook surfaces still include projection/UMAP/LatentDNA wording and active dashboard UMAP helpers. That blurs OPAL campaign review with upstream representation review. |
| Proposed change | Remove or quarantine active LatentDNA/UMAP/projection residue from canonical OPAL review and generated notebooks. Preserve only generic X-column provenance: `x_column`, `x_dim`, `records_path`, and optional `x_contract`. |
| Contract shape | `campaign.x_provenance = {"x_column": str, "x_dim": int | None, "source_note": str | None}`. No UMAP, cluster, LatentDNA atlas, or producer-specific readiness gates. |
| Affected modules | `src/dnadesign/opal/src/analysis/campaign_progress.py`, `src/dnadesign/opal/src/analysis/notebook_template.py`, `src/dnadesign/opal/src/analysis/dashboard/*`, notebook tests. |
| Migration notes | If UMAP dashboard code is still needed, move it to an archived/noncanonical namespace or a producer/study-owned package. Add tests that canonical OPAL notebook/review surfaces contain no `LatentDNA`, `UMAP`, `projection`, `cluster__ldn`, or `DenseGen visual` strings except in boundary docs. |
| Acceptance criteria | Generated notebooks and `opal review` mention only campaign contracts, X column provenance, ledgers, progress, selection, labels, predictions, plots, and limitations. Study/probe code remains free to link OPAL review artifacts from study-owned reports. |
| Tests | Snapshot tests for generated notebook, review Markdown, review HTML, and public JSON. Static grep/lint guard over canonical OPAL surfaces. |

### B. Public/Private API Boundary

| Field | Specification |
| --- | --- |
| Problem | Generated notebooks import `dnadesign.opal.src.*`, making durable artifacts brittle and encouraging downstream users to copy private imports. |
| Proposed change | Expand `dnadesign.opal` public APIs narrowly for reporting, plot manifest loading, status inspection, and notebook view-model construction. Generated notebooks should import only public APIs and general third-party packages. |
| Contract shape | Public exports: `build_campaign_progress`, `build_campaign_review`, `read_campaign_predictions`, `load_review_manifest`, `list_plot_kinds`, `describe_plot_kind`, `load_plot_artifact_manifest`, `build_notebook_view_model`, `render_campaign_progress_text`. |
| Affected modules | `src/dnadesign/opal/__init__.py`, `src/dnadesign/opal/src/analysis/notebook_template.py`, `src/dnadesign/opal/src/reporting/*`, `src/dnadesign/opal/src/plots/*`. |
| Migration notes | Keep internal modules intact. Add thin public adapters rather than relocating large internals. Generated notebooks should pin the public schema versions they expect. |
| Acceptance criteria | No generated notebook imports from `dnadesign.opal.src.*`. Study packages use only `dnadesign.opal` public helpers. |
| Tests | Public import tests, generated notebook text tests, architecture boundary check, study probe import tests. |

### C. Run/Progress Semantics

| Field | Specification |
| --- | --- |
| Problem | Early CLI events are useful but not cleanly separated from actual run events. Logs can include preflight/aborted entries without run_id. Progress must disclose ambiguity, incomplete runs, stale locks, and active locks. |
| Proposed change | Introduce explicit event phases: `command`, `preflight`, `run`, `abort`, `finalize`. Acquire a preflight/run lock token before writing events that imply mutation. Emit terminal abort events on operator cancellation and contract failures where possible. |
| Contract shape | `ProgressEventContract = {schema_version, event_id, phase, run_id, round, stage, severity, status, ts, lock_token, message, payload}`. `run_id` is required for `phase=run` after `run_context`. |
| Affected modules | `src/dnadesign/opal/src/cli/commands/run.py`, `src/dnadesign/opal/src/runtime/run_round.py`, `src/dnadesign/opal/src/reporting/progress.py`, `src/dnadesign/opal/src/reporting/summary.py`, `src/dnadesign/opal/src/storage/locks.py`. |
| Migration notes | Preserve old fields while adding `schema_version` and `phase`; readers should accept old logs but label them `legacy_event_contract`. Do not silently discard old preflight events. |
| Acceptance criteria | `opal progress --json` reports `run_scope`, `ambiguous_run_scope`, `active_lock`, `stale_lock`, `aborted`, `legacy_events`, `preflight_events`, and `run_events`. `opal review --json` refuses mixed-run review unless run_id is explicit. |
| Tests | Unit tests for aborted prompt, lock conflict, stale lock, multiple starts, multiple run_ids per round, missing done event, and legacy log parsing. CLI JSON snapshots. |

### D. X Contract Unification

| Field | Specification |
| --- | --- |
| Problem | Runtime `identity` accepts scalar/list/JSON string vectors, docs describe Arrow list or JSON string, and public validation requires Parquet `fixed_size_list`. This split weakens fail-fast behavior. |
| Proposed change | Define canonical physical schema as Parquet Arrow `fixed_size_list<float32 or float64>` with finite, non-null values and stable row count. Noncanonical forms are allowed only through explicit import/normalization commands, never inside campaign execution. |
| Contract shape | `XMatrixContract` with `physical_type`, `x_dim`, `canonical=true`, `validation_level=parquet_schema_and_values`, and `normalization_source` if converted. |
| Affected modules | `src/dnadesign/opal/src/storage/x_contracts.py`, `src/dnadesign/opal/src/transforms_x/identity.py`, `src/dnadesign/opal/src/runtime/round/stages.py`, `src/dnadesign/opal/docs/reference/data-contracts.md`, validate/init/run/explain paths. |
| Migration notes | Keep `identity` as a model-matrix transform, but require candidate records to validate before run/explain/review. Add `opal x normalize` or `opal import-records` if scalar/list/JSON compatibility is needed for legacy inputs. |
| Acceptance criteria | `opal run`, `opal explain`, `opal review`, and notebook view model all fail or warn on invalid/noncanonical X according to severity. Docs no longer describe JSON-string X as runtime-canonical. |
| Tests | Fixed-size list accept; variable list/scalar/JSON reject at campaign execution; normalization command accepts legacy forms and writes canonical records; review exposes x contract. |

### E. Review Manifest Authority

| Field | Specification |
| --- | --- |
| Problem | Existing review manifests can omit plots while stale PNGs remain on disk. Notebook/review surfaces that trust directory contents can mislead users. |
| Proposed change | Make `manifest.json` and `review_manifest.json` authoritative snapshots. Review/notebook readers render only artifacts referenced by the active manifest. Extra files under known artifact directories become `StaleArtifactWarning` entries. |
| Contract shape | `ReviewManifestContract = {schema_version, generated_at, review_scope, campaign, run, progress, selection, plots, stale_artifacts, warnings, artifacts}`. |
| Affected modules | `src/dnadesign/opal/src/reporting/review.py`, generated notebook, study probe review wrappers, tests. |
| Migration notes | Existing manifests without `stale_artifacts` are accepted as v1 and upgraded in memory. Do not delete stale files by default; add an explicit `opal review prune-stale` later if needed. |
| Acceptance criteria | If `outputs/review/plots/*.png` exists but manifest `plots` is empty, review JSON and notebook show a stale-artifact warning and do not render those PNGs as current evidence. |
| Tests | Fixture with manifest empty plus stale files; fixture with missing manifest-referenced file; fixture with status `failed`; JSON schema validation. |

### F. Plot Artifactization

| Field | Specification |
| --- | --- |
| Problem | Configured plots currently write media and optional CSVs, but no per-plot manifest. Review and notebooks cannot reliably explain source data, params, run scope, freshness, or failure modes. |
| Proposed change | Each configured plot writes an artifact directory or manifest next to outputs. The manifest is the authority for plot cards and downstream review. |
| Contract shape | `PlotArtifactManifest` as defined above. Inputs should include path, role, exists, size, mtime, and optional content hash where practical. Outputs should include media path, tidy CSV path, format, bytes, mtime, and status. |
| Affected modules | `src/dnadesign/opal/src/plots/runner.py`, `src/dnadesign/opal/src/plots/_context.py`, plot plugins, `src/dnadesign/opal/src/registries/plots.py`, docs/tests. |
| Migration notes | Start with a lightweight manifest writer in the runner so existing plugins need minimal changes. Plugins can optionally return richer data-shape metadata. |
| Acceptance criteria | `opal plot` emits one manifest per plot entry, an aggregate `outputs/plots/plot_manifest.json`, and `opal plot --list-config --json` references expected artifact IDs. Failed plots write failed manifests with error taxonomy and no ambiguous success. |
| Tests | Success manifest, failed manifest, save_data CSV manifest, missing input failure, stale source detection, aggregate manifest schema. |

### G. Plot Ontology And New Primitives

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

### H. Marimo Notebook UX

| Field | Specification |
| --- | --- |
| Problem | The generated notebook is useful but too tied to private helpers, filesystem plots, and optional projection context. It lacks a manifest-backed plot card model and explicit stale/fresh state. |
| Proposed change | Generate a manifest-backed marimo review notebook over `NotebookViewModel`. It should read review manifests and plot manifests first, never raw directory listings as authority. |
| Contract shape | `NotebookViewModel` plus lazy sections: `At a glance`, `Validity`, `Changes`, `Evidence`, `Distrust and limitations`, `Artifacts`, `Raw tables`. |
| Affected modules | `src/dnadesign/opal/src/analysis/notebook_template.py`, public notebook API, tests, CLI notebook command. |
| Migration notes | Preserve current ability to generate before first run. In that state, the view model reports `not_started` and missing manifest states explicitly. |
| Acceptance criteria | First viewport shows campaign, run scope, status, X column, label source, latest run_id, selected count, stale warnings, and missing artifacts. Heavy tables and plots are inside `mo.accordion(..., multiple=True, lazy=True)`. |
| Tests | `marimo check` smoke, generated Python parse, no private imports, no UMAP/LatentDNA strings, manifest-only plot rendering, missing/stale plot states. |

Notebook design rule: the user path is "Is it valid?", "What changed?", "What visual evidence supports that?", and "What should I distrust?" Plot cards must show source data, params, status, stale/fresh state, generated time, media links, and tidy CSV links.

### I. CLI UX And Machine Surfaces

| Surface | Proposed contract |
| --- | --- |
| `opal progress --json` | Add `schema_version`, `run_scope`, `rounds`, `warnings`, `locks`, `stale_artifacts`, `aborted_runs`, `legacy_events`, and stable error categories. |
| `opal review --json` | Return `ReviewManifestContract` plus write paths. Refuse ambiguous run scope unless `--run-id` is explicit. Include stale-file warnings. |
| `opal status` | Become the operator summary over state, latest run, lock state, manifest freshness, and next safe commands. |
| `opal runs` | Add JSON schemas for `runs list` and `runs show`; expose run status, done/aborted state, artifact manifest refs, and duplicate/compaction warnings. |
| `opal plot --list` | Current text is fine; add `--json` later with `PlotMeta` and schema version. |
| `opal plot --describe` | Include required data shape, tidy CSV schema, output manifest schema, and failure modes. |
| `opal notebook generate` | Accept review manifest and plot manifest options; run a smoke check by default when marimo is installed; report `NotebookViewModel` summary in JSON. |

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

### J. Error Taxonomy And Fail-Fast Behavior

| Category | When raised | Severity |
| --- | --- | --- |
| `ConfigContractError` | Unknown config keys, duplicate YAML keys, unknown plugins, invalid params, incompatible shared-label config | Error |
| `XContractError` | Missing X column, noncanonical physical schema, null/nonfinite/ragged X, mismatched X dim | Error for run/review; warning only for pre-run status if records missing |
| `LabelSourceError` | Missing configured sidecar for run, malformed labels, unknown candidate IDs, y_space mismatch | Error |
| `RunScopeAmbiguityError` | Multiple run_id values for a selected round without explicit run_id | Error for review/plots/predictions; warning for listing |
| `ProgressContractError` | Malformed round log, missing terminal state, run_id mismatch | Error or attention depending surface |
| `PlotDataContractError` | Missing required plot input, missing required columns, nonnumeric data, vector length mismatch | Error per plot, aggregate command exit nonzero |
| `ReviewManifestError` | Missing/invalid review manifest, schema mismatch, manifest references missing files | Error for manifest read; warning for stale extras |
| `StaleArtifactWarning` | Files exist in artifact dirs but are absent from active manifest or older than inputs | Warning |
| `NotebookViewModelError` | Notebook generation cannot build a valid view model | Error |
| `PublicApiBoundaryError` | Tests detect generated/study imports from private OPAL internals | Test/lint failure |

Fail-fast policy:

- Execution commands fail on invalid contracts.
- Inspection commands may return `attention` with structured warnings when a campaign has not started.
- Stale artifacts never silently render as current evidence.
- Explicit repair/prune commands may mutate artifacts; inspection commands do not.

### K. Testing And Quality Gates

Required test coverage:

| Area | Tests |
| --- | --- |
| Strict config behavior | Unknown top-level keys, duplicate YAML keys, unknown plugins, invalid plugin params, shared sidecar incompatibilities |
| X schema validation | Accept fixed-size list; reject variable list, scalar, JSON string, nulls, nonfinite values; normalization command writes canonical schema |
| Run_id-scoped progress | Multiple runs per round, explicit run_id filter, missing run_id, legacy events, latest selector |
| Interrupted/aborted runs | Prompt abort, exception during records load, exception after run_context, lock conflict, stale lock |
| Review manifest schema | Valid schema, missing referenced files, stale extras, no-plots with old PNGs, JSON schema version |
| Plot plugin metadata | Every registered plot has `PlotMeta`, required inputs, output data shape, failure modes, and `--describe` coverage |
| Plot tidy CSV shape | Each plot with `save_data` writes declared tidy schema and manifest references it |
| Marimo notebook | Python parse, `marimo check`, no private imports, no UMAP/LatentDNA residue, manifest-backed plot cards, lazy accordions |
| Public API import paths | Study code imports OPAL public API only; generated notebooks use public API only |
| Dogfood | cipro random positive/null, ethanol random, dual/AND random, leave-sigma35 variants, and real-assay workflows only when evidence exists |

Recommended targeted commands during implementation:

```bash
uv run pytest -q src/dnadesign/opal/tests/storage/test_x_contracts.py
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

## L. Documentation

Update these docs as implementation lands:

| Document | Required updates |
| --- | --- |
| `src/dnadesign/opal/docs/reference/cli.md` | Fix config resolution mismatch for `progress`; add JSON schemas for progress/review/runs/status/plot/notebook; document ambiguity errors and stale warnings. |
| `src/dnadesign/opal/docs/reference/data-contracts.md` | Declare canonical X physical schema; move noncanonical forms to import/normalization docs. |
| `src/dnadesign/opal/docs/reference/plots.md` | Add plot artifact manifests, tidy CSV schema expectations, `PlotMeta` requirements, and generic plot primitive guide. |
| `src/dnadesign/opal/docs/reference/configuration.md` | Clarify that `plot_config` configures review artifacts but runtime round execution does not depend on plots. |
| New `src/dnadesign/opal/docs/reference/review-manifests.md` | Define campaign review manifest and stale artifact behavior. |
| New `src/dnadesign/opal/docs/reference/notebooks.md` | Define generated notebook contract, public imports, view model, smoke checks, and no representation-browser boundary. |
| `docs/studies/stress_ethanol_cipro_growth/contexts/opal/*` | Keep study/probe benchmark docs outside OPAL core and remove wording that generalizes cipro/random dogfood. |

## M. Migration Plan

| Phase | Scope | Dependencies | Risk | Acceptance criteria | Rollback/migration concerns |
| --- | --- | --- | --- | --- | --- |
| P0: Boundary cleanup and spec alignment | Remove canonical OPAL notebook/review UMAP/projection/LatentDNA residue; fix CLI docs mismatch | None | Low to medium; tests may encode old strings | Generated canonical notebook/review has no representation-browser content; docs distinguish OPAL vs study/probe | Archive or quarantine old dashboard code rather than deleting if ownership is unclear |
| P1: Run/progress/review contract hardening | Add event phases, abort events, lock state, run ambiguity JSON, stale artifact warnings in review | P0 recommended | Medium; old logs exist | Progress/review JSON schemas include warnings and refuse mixed-run review | Readers accept old logs with `legacy_event_contract` warnings |
| P2: X contract unification | Enforce canonical fixed-size list at run/review; add normalization/import path for legacy X | P1 helpful | Medium/high; existing fixtures may use legacy X | Runtime, validator, docs, and tests agree on canonical schema | Provide explicit migration command; do not silently coerce during run |
| P3: Plot artifact manifests | Add per-plot manifests and aggregate plot manifest; report failures structurally | P1 | Medium; all plot plugins affected indirectly | Every configured plot produces success/failed/skipped manifest | Runner-level manifest wrapper minimizes plugin edits |
| P4: Generic plot primitives | Add `metric_over_rounds`, `feature_importance_heatmap`, `vector_summary_heatmap`, overlap/composition/uncertainty/support/calibration primitives | P3 | Medium; temptation to over-abstract | New primitives pass data-shape contracts and avoid study names | Keep SFXI-specific plots during transition as configured diagnostics |
| P5: Marimo review notebook modernization | Public view-model API, manifest-backed plot cards, lazy accordions, smoke checks | P3 | Medium; generated artifacts are durable | Notebook uses public imports and manifest authority; smoke checks pass | Keep old template behind explicit legacy flag for one release if needed |
| P6: Dogfood expansion and CI gates | Add broader synthetic and real-data smoke coverage with explicit evidence labels | P1-P5 depending gate | High; runtime cost and study availability | Reports distinguish cipro/random, random-all, leave-sigma35, and real assay evidence | Keep expensive gates optional/nightly; never block core OPAL unit tests on study-specific benchmarks |

## N. Risks And Tradeoffs

| Risk | Tradeoff and mitigation |
| --- | --- |
| Over-abstracting plot interfaces | Keep plot contracts small: metadata, required inputs, tidy schema, manifest. Do not require a universal plotting superclass. |
| Breaking existing campaigns | Stage warnings before errors where safe for inspection surfaces; execution contracts should still fail fast. Provide explicit normalization/migration commands. |
| Stale artifacts misleading users | Manifest-first rendering and stale warnings are worth the extra manifest plumbing. Do not auto-delete without an explicit prune command. |
| Notebook durability under refactor | Public view-model APIs reduce brittleness, but they become semver-like contracts. Keep them narrow. |
| Single-host locks in shared environments | Document local-only lock scope and surface active/stale lock state. Do not imply distributed safety without a real lease. |
| Synthetic-oracle dogfood overclaim | Treat current cipro/random pass as one scoped scenario. Broader claims require ethanol, dual/AND, leave-sigma35, and eventually real assay evidence. |
| SFXI-specific plots crowding generic ontology | Keep existing SFXI diagnostics, but make new multi-round/progress plots shape-based and configurable. |
| Manifest hashes increasing cost | Prefer mtime/size by default and content hashes only for small or critical files unless configured. |

## O. Open Questions

1. Should the noncanonical X normalization command live under `opal x normalize`, `opal import-records`, or `opal validate --repair --out`? Recommendation: `opal x normalize` because mutation is explicit and scoped.
2. Should plot manifests be one file per plot directory plus aggregate manifest, or aggregate-only? Recommendation: both, with per-plot manifests as source of truth and aggregate as an index.
3. Should legacy UMAP dashboard code be deleted, moved under `archived/`, or migrated to a study/producer-owned package? Recommendation: quarantine first, then delete after confirming no canonical command imports it.
4. What stale-artifact threshold should make review fail rather than warn? Recommendation: missing manifest-referenced files are errors; extra unreferenced files are warnings.
5. Which dogfood gates are fast enough for PR CI versus nightly/local operator runs? Recommendation: OPAL unit/fixture gates in PR; study/probe dogfood in optional/nightly workflows.

## P. Acceptance Criteria Summary

- [ ] OPAL core imports no study/probe code and canonical OPAL surfaces contain no LatentDNA/UMAP/representation-browser content.
- [ ] Study/probe code depends only on documented OPAL public APIs.
- [ ] `opal progress --json`, `opal review --json`, `opal status --json`, and `opal runs ... --json` expose schema versions, run scope, warnings, ambiguity, aborted/incomplete state, and lock state where relevant.
- [ ] `opal run` progress events distinguish preflight, actual run, abort, and completion.
- [ ] X validation has one canonical physical schema and runtime execution fails fast on noncanonical X.
- [ ] Review manifests and plot manifests are authoritative; stale files are detected and never rendered as current evidence.
- [ ] Every plot plugin declares metadata, required inputs, output data shape, and failure modes.
- [ ] Generic plot primitives cover scalar over rounds, vector over rounds, attribution heatmap, overlap, composition, uncertainty/support, objective decomposition, audit table, and calibration.
- [ ] Generated marimo notebooks use public OPAL APIs, manifest-backed view models, lazy accordions, plot cards, and smoke checks.
- [ ] Docs are updated for CLI config resolution, data contracts, plot authoring, review manifests, notebooks, public APIs, and study/probe boundary guidance.
- [ ] Tests cover strict config, X contracts, run_id progress, aborts, stale artifacts, plot manifests, notebook generation, public imports, and no active OPAL LatentDNA/UMAP residue.
- [ ] Dogfood evidence is labeled by coverage; cipro/random is not generalized to ethanol, dual/AND, leave-sigma35, or real assay behavior without independent evidence.
