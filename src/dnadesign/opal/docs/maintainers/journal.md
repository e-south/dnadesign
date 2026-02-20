## OPAL Dev Journal

This journal tracks ongoing Elm_UQ analysis, refactor notes, and merge-readiness decisions for OPAL.

### 2026-02-20 SFXI uncertainty canon alignment plan

Decision points captured from active issue triage:
- Analytical canon reference is `c5666a7`.
- Analytical effect moments must use log2-consistent `ln(2)` scaling with `np.exp` forms (not `np.exp2` on variance terms).
- Keep analytical default for `beta=gamma=1`, with explicit approximation notes.
- Treat #23 as partially superseded by #27, and resolve both in one coherent PR narrative.

Execution plan used for this pass:
- Add deterministic tests first for analytical log2-moment correctness and regression baselines.
- Remove Monte Carlo-based uncertainty smoke checks to reduce CI flakiness.
- Patch `sfxi_v1` analytical intensity moments to log2-consistent closed forms.
- Preserve #27 strictness invariants (required std `>0`, emitted sigma `>0`, no clip-mask variance gating).
- Update objective docs and PR #26 body to the same canon wording and issue relationship.
- Include pending formatting diffs and end with a clean working tree.

### 2026-02-16 Guided demo workflows (CLI + docs)

What changed:
- Added guided CLI surfaces:
  - `opal guide` (config-driven runbook: text/markdown/json)
  - `opal guide next` (state-aware next-command recommendation)
  - `opal demo-matrix` (isolated pressure test runner across canonical demo flows)
- Added default human-output next-step hints (opt-out with `--no-hints`) for:
  - `init`, `validate`, `ingest-y`, `run`, `explain`, `verify-outputs`
- Enhanced `explain` with a preflight block that explicitly states:
  - observed-round vs labels-as-of semantics
  - SFXI current-round label counts vs `scaling.min_n`
  - run-fail warning + fix command when under threshold
- Added tests:
  - `test_cli_guide.py`
  - `test_cli_guide_next.py`
  - `test_cli_guidance_hints.py`
  - `test_cli_demo_matrix.py`
- Updated workflow and CLI docs to surface guided usage:
  - `docs/workflows/{rf-sfxi-topn,gp-sfxi-topn,gp-sfxi-ei}.md`
  - `docs/reference/cli.md`

Why:
- Convert demo docs from command lists into a guided operator flow with explicit lifecycle checkpoints.
- Reduce round-semantic confusion and make “what to do next” available directly in CLI output.
- Provide an agent/CI-oriented pressure-test command for the canonical workflow matrix.

Validation:
- `uv run pytest -q src/dnadesign/opal/tests/test_cli_guide.py src/dnadesign/opal/tests/test_cli_guide_next.py src/dnadesign/opal/tests/test_cli_guidance_hints.py src/dnadesign/opal/tests/test_cli_demo_matrix.py`
- `uv run pytest -q src/dnadesign/opal/tests/test_cli_workflows.py src/dnadesign/opal/tests/test_workflow_matrix_cli.py src/dnadesign/opal/tests/test_verify_outputs.py src/dnadesign/opal/tests/test_cli_run_renderer.py src/dnadesign/opal/tests/test_cli_ingest_missing_x_drop.py src/dnadesign/opal/tests/test_cli_campaign_reset.py`
- `uv run pytest -q src/dnadesign/opal/tests`
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs`

### 2026-02-16 Plain-language docs pass (user-facing wording cleanup)

What changed:
- Replaced awkward or overloaded wording in user-facing docs with plain instructional language.
- Updated terms such as:
  - `switchboard` -> `main docs page` / `Documentation`
  - `canonical` -> `primary` or `main` (where appropriate)
  - `fail-fast` section labels -> `Error cases`
  - `Intent` headings -> `Purpose`
  - `deep pages` -> `detail pages`
- Touched user-facing docs under:
  - `src/dnadesign/opal/README.md`
  - `src/dnadesign/opal/docs/index.md`
  - `src/dnadesign/opal/docs/plugins/*.md`
  - `src/dnadesign/opal/docs/reference/{cli.md,data-contracts.md,schema-to-runtime.md}`

Why:
- Improve readability and reduce jargon while keeping behavior and contract meaning unchanged.

Validation:
- Local markdown link target audit across `src/dnadesign/opal/**/*.md`.
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs`.

### 2026-02-16 Plugin docs alignment pass (GP/EI deep pages + CLI reference tightening)

What changed:
- Added deep plugin behavior/math pages to match objective docs structure:
  - `src/dnadesign/opal/docs/plugins/model-gaussian-process.md`
  - `src/dnadesign/opal/docs/plugins/selection-expected-improvement.md`
- Updated docs switchboard and workflow guides to link these pages directly:
  - `src/dnadesign/opal/docs/index.md`
  - `src/dnadesign/opal/docs/workflows/gp-sfxi-topn.md`
  - `src/dnadesign/opal/docs/workflows/gp-sfxi-ei.md`
  - `src/dnadesign/opal/docs/workflows/rf-sfxi-topn.md`
- Kept registry-level pages focused on plugin contract/wiring and linked out to deep pages:
  - `src/dnadesign/opal/docs/plugins/models.md`
  - `src/dnadesign/opal/docs/plugins/selection.md`
  - `src/dnadesign/opal/docs/plugins/objectives.md`
- Tightened CLI reference scope to command contracts by removing tutorial-style workflow sections:
  - `src/dnadesign/opal/docs/reference/cli.md`
- Removed local docs `.DS_Store` cruft from working tree (these were untracked; ignore rule already exists in repo root `.gitignore`).

Why:
- Keep one consistent plugin-doc mental model:
  - registry page at `docs/plugins/<type>.md`
  - deep behavior/math pages at `docs/plugins/<type>-<plugin>.md`
- Keep `reference/cli.md` as a contract lookup page; keep run sequencing in workflows.

Validation:
- Local markdown link target audit across `src/dnadesign/opal/**/*.md`.
- `uv run ruff check src/dnadesign/opal/docs`.

### 2026-02-16 Docs IA pass (single canonical index)

What changed:
- Removed nested docs entrypoints:
  - `docs/workflows/index.md`
  - `docs/plugins/index.md`
  - `docs/reference/index.md`
- Kept `docs/index.md` as the only documentation index.
- Updated docs and README links to target `docs/index.md` anchors (`#workflows`, `#plugins`, `#reference`) and direct topic pages.

Why:
- Multiple nested index pages made navigation indirect and increased UX ambiguity.
- One canonical index provides a single stable switchboard for all docs paths.

Validation:
- `find src/dnadesign/opal/docs -type f -name 'index.md'` now returns only `src/dnadesign/opal/docs/index.md`.
- Grep checks found no markdown links to removed nested index pages.

### 2026-02-16 Docs IA pass (workflows + unified plugins surface)

What changed:
- Renamed `docs/demos/` to `docs/workflows/` and updated cross-links from docs index, concepts, campaign readmes, and maintainer testing pages.
- Removed nested plugin docs under `docs/reference/plugins/` and consolidated all plugin docs under `docs/plugins/`.
- Moved objective math pages into plugins tree:
  - `docs/plugins/objective-sfxi.md`
  - `docs/plugins/objective-spop.md`
- Added `docs/plugins/index.md` as the single plugin documentation entrypoint.
- Updated docs switchboard and reference copy to use the new mental model:
  - workflows = end-to-end run paths
  - plugins = model/objective/selection/transform semantics
  - reference = schema/CLI/data contracts/runtime crosswalk

Why:
- Remove ambiguity from `docs/reference/plugins/` nesting and objective docs being split away from plugin docs.
- Give one canonical place to find GP, EI, SFXI, and plugin wiring semantics.

Validation:
- Local markdown link target audit across `src/dnadesign/opal/**/*.md` passed.
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` passed.

### 2026-02-16 Docs UX pass (CLI semantics + concept cross-links)

What changed:
- Updated `docs/reference/cli.md` usage blocks and workflow examples to prefer explicit semantic flags:
  - `ingest-y --observed-round ... --in ...`
  - `run/explain --labels-as-of ...`
- Updated CLI mental-model section to clarify that `--round` is a shared alias and explicit flags are preferred by command surface.
- Added concise `Next steps` cross-links to concept pages:
  - `docs/concepts/architecture.md`
  - `docs/concepts/roundctx.md`
  - `docs/concepts/strategy-matrix.md`
- Renamed docs switchboard heading from `Maintainer Notes` to `Maintainers` for consistency with `docs/index.md`.

Why:
- Remove residual flag ambiguity in reference examples.
- Reduce navigation backtracking from concept pages.

Validation:
- Local markdown link target audit across `src/dnadesign/opal/**/*.md` passed with zero unresolved links.
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` passed.

### 2026-02-16 Docs UX pass (navigation + semantics alignment)

What changed:
- Fixed broken maintainer links in `docs/index.md` after maintainer path moves.
- Added maintainer switchboard and moved demo pressure-test matrix script out of `docs/workflows/index.md`:
  - `docs/index.md`
  - `docs/maintainers/testing-matrix.md`
- Added objective docs/navigation surfaces:
  - `docs/plugins/index.md`
  - `docs/plugins/objectives.md`
- Added user-facing runtime guidance pages:
  - `docs/reference/troubleshooting.md`
  - `docs/reference/schema-to-runtime.md`
- Updated demo guides to use explicit round/input flags:
  - `ingest-y --observed-round ... --in ...`
  - `run --labels-as-of ...`
  - `explain --labels-as-of ...`
- Updated SFXI doc with an operational contract header, parameterized percentile wording, and explicit distinction between objective channels and diagnostic columns.
- Updated SPOP doc status text to draft/in-progress objective semantics.

Why:
- Remove onboarding clutter from demo entrypoints.
- Reduce round-flag ambiguity in step-by-step docs.
- Make channel ref wiring and config-to-runtime behavior easier to find without source dives.

Validation:
- Local markdown link target audit across `src/dnadesign/opal/**/*.md` passed with zero unresolved links.
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` passed.

### 2026-02-16 Docs IA Overhaul (no compatibility stubs)

What changed:
- Rebased OPAL docs navigation around explicit index pages:
  - `docs/index.md` (primary docs switchboard)
  - `docs/workflows/index.md` (demo switchboard)
  - `docs/reference/index.md` (reference switchboard)
- Removed redundant `docs/guides/` routing layer and promoted demo guides to `docs/workflows/`.
- Simplified top-level `README.md` to an overview + switchboard only (removed duplicated navigation sections).
- Updated all campaign README guide pointers to `docs/workflows/*.md`.
- Updated concept references (`strategy-matrix.md`) to the new demo guide paths.

Why:
- Previous docs structure had redundant entry nodes (`Start here` + `Documentation map`) and a nested demos path that made discovery harder.
- The new structure keeps one primary narrative path at each level and removes stale/duplicated routing pages.

Validation:
- Local markdown link + anchor audit across `src/dnadesign/opal/**/*.md` passed with zero unresolved targets.

### 2026-02-16 Run-Meta Contract Hardening (channel refs + denominator semantics)

What changed:
- `build_run_meta_event` now enforces strict channel-ref invariants:
  - `selection_score_ref` must be non-empty.
  - `selection_uncertainty_ref` must be null or a non-empty string.
- `objective__denom_percentile` no longer defaults to `95` for objectives that do not provide denominator-scaling metadata.
  - Resolution order is now:
    1) `objective_summary_stats["denom_percentile"]` when present,
    2) `objective_params.scaling.percentile` when explicitly provided,
    3) otherwise `null`.
  - percentile values are validated in `[1, 100]`.
- `sfxi_v1` summary stats now emit `denom_percentile` explicitly.

Why:
- A hard-coded default of `95` in `run_meta` for non-SFXI objectives produced misleading metadata.
- Empty selection refs in run metadata create ambiguous, non-auditable channel linkage in downstream analysis.

Tests added:
- `test_build_run_meta_event_sets_denom_percentile_none_without_scaling`
- `test_build_run_meta_event_rejects_empty_selection_score_ref`
- `test_build_run_meta_event_rejects_blank_selection_uncertainty_ref`
- `test_sfxi_v1_scores_and_ctx_denom` now asserts `summary_stats.denom_percentile`.

Validation:
- `uv run pytest -q src/dnadesign/opal/tests/test_ledger_dataset_writes.py::test_build_run_meta_event_sets_denom_percentile_none_without_scaling src/dnadesign/opal/tests/test_ledger_dataset_writes.py::test_build_run_meta_event_rejects_empty_selection_score_ref src/dnadesign/opal/tests/test_ledger_dataset_writes.py::test_build_run_meta_event_rejects_blank_selection_uncertainty_ref src/dnadesign/opal/tests/test_objective_sfxi_v1.py::test_sfxi_v1_scores_and_ctx_denom`
- `uv run pytest -q src/dnadesign/opal/tests/test_ledger_dataset_writes.py src/dnadesign/opal/tests/test_objective_sfxi_v1.py src/dnadesign/opal/tests/test_run_round_integrity.py src/dnadesign/opal/tests/test_dashboard_utils.py`
- `uv run pytest -q src/dnadesign/opal/tests`
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs`

### 2026-02-16 Exception-Surface Hardening (no swallow paths)

What changed:
- Dashboard selection utilities now only catch expected coercion/reconstruction errors:
  - `coerce_selection_dataframe` catches `(TypeError, ValueError, pl.exceptions.PolarsError)` only.
  - `ensure_selection_columns` catches `(TypeError, ValueError)` only.
  - Unexpected runtime failures now propagate instead of being silently converted into empty/soft-error states.
- Round writebacks no longer swallow malformed resume state:
  - removed `try/except Exception: pass` around `allow_resume` round filtering in `update_campaign_state`.
  - malformed `round_index` now fails fast.
- Round writebacks no longer mask unexpected `model_artifacts()` failures:
  - removed broad wrapper in `write_round_artifacts`; unexpected plugin errors now surface directly.

Why:
- Broad exception swallowing in analysis/runtime paths creates silent divergence and delayed failures.
- Strict failure semantics make plugin/data contract violations visible at the exact fault site.

Tests added:
- `test_selection_coercion_does_not_swallow_unexpected_errors`
- `test_ensure_selection_columns_does_not_swallow_unexpected_overlay_errors`
- `test_run_round_allow_resume_rejects_malformed_state_round_index`
- `test_run_round_bubbles_runtime_error_from_model_artifacts`

Validation:
- `uv run pytest -q src/dnadesign/opal/tests/test_dashboard_utils.py src/dnadesign/opal/tests/test_run_round_integrity.py`
- `uv run pytest -q src/dnadesign/opal/tests`
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs`

### 2026-02-16 Run-Pred API Cleanup + Reference Invariants

What changed:
- Removed stale, unused writeback API kwargs from `build_run_pred_events`:
  - removed `y_hat_model_sd`
  - removed `y_obj_scalar_sd`
- Updated all runtime and test call sites to the reduced API surface.
- Tightened run-pred reference contracts:
  - `selected_score_ref` must be non-empty.
  - `selected_uncertainty` and `selected_uncertainty_ref` must be provided together (both present or both absent).

Why:
- The removed SD kwargs were dead surface area that implied behavior OPAL does not use.
- Missing/empty score/uncertainty refs created ambiguous ledger rows and fail-late behavior in downstream analysis paths.
- The stricter contract keeps writebacks deterministic and fail-fast.

Tests added:
- `test_build_run_pred_events_rejects_removed_sd_kwargs`
- `test_build_run_pred_events_rejects_empty_selected_score_ref`
- `test_build_run_pred_events_rejects_uncertainty_without_ref`
- `test_build_run_pred_events_rejects_uncertainty_ref_without_values`

Validation:
- `uv run pytest -q src/dnadesign/opal/tests/test_ledger_dataset_writes.py src/dnadesign/opal/tests/test_analysis_facade_setpoint.py src/dnadesign/opal/tests/test_verify_outputs.py`
- `uv run pytest -q src/dnadesign/opal/tests`
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs`

### 2026-02-16 Run-Pred Writeback Hardening (strict vector contracts)

What was tightened:
- `build_run_pred_events` now validates all row-vector inputs explicitly:
  - `y_hat_model` must be 2D, finite, and aligned to candidate count.
  - `selected_score`, `selection_score`, `selected_uncertainty` must match candidate count.
  - `selected_uncertainty` must be finite and non-negative.
  - `SelectionEmit` arrays must align to candidate count.
  - score/uncertainty channel payload vectors are prevalidated for length + finiteness.

Performance/structure improvement:
- Channel arrays are now prepared once per call, then row payloads are assembled from cached arrays.
- Removes repeated `np.asarray(...).reshape(-1)` conversions inside per-row loops.

Why:
- Previous behavior could raise incidental `IndexError` on mismatched optional vectors and allowed non-finite uncertainty payloads to leak into ledger rows.
- Prevalidating once at the function boundary improves fail-fast behavior and reduces avoidable per-row conversion overhead.

Tests added:
- `test_build_run_pred_events_rejects_selection_score_length_mismatch`
- `test_build_run_pred_events_rejects_non_finite_selected_uncertainty`

Validation:
- `uv run pytest -q src/dnadesign/opal/tests/test_ledger_dataset_writes.py`
- `uv run pytest -q src/dnadesign/opal/tests`
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs`

### 2026-02-16 Selection Contract Hardening (strict plugin output typing)

What was tightened:
- Centralized selection reserved-key handling into shared contract utilities (`core/selection_contracts.py`) so runtime and registry cannot drift on which params are runtime-reserved vs plugin-specific.
- Selection result validation now rejects coercive payloads:
  - `order_idx` must be numeric and integral (no silent float truncation).
  - `score` must be numeric and finite (no string coercion).
  - empty selections (`expected_len=0`) are handled explicitly without reduction errors.
- `normalize_selection_result` now rejects invalid provided `order_idx` instead of silently recomputing from scores.
- Runtime tie expansion now uses selection-plugin score (`validated_sel.score`) rather than objective channel score.

Why:
- Previous validation cast `order_idx` to `int` and `score` to `float`, which could silently accept malformed plugin output instead of failing at the plugin boundary.
- Runtime and registry duplicated reserved-key sets; drift risk would create hard-to-diagnose plugin wiring failures.
- Normalization fallback on malformed provided order could mask upstream plugin/data issues.
- Using objective scores for tie expansion could silently under-select when a plugin ranks on a different score domain (for example, acquisition score vs objective channel score).

Tests added:
- `test_selection_contracts.py`
  - `test_extract_selection_plugin_params_excludes_reserved_keys`
  - `test_reserved_selection_param_keys_are_complete_for_runtime_keys`
- `test_selection_result_validation.py`
  - `test_validate_selection_result_rejects_fractional_order_idx`
  - `test_validate_selection_result_rejects_non_numeric_score_payload`
  - `test_validate_selection_result_accepts_empty_payload_when_expected_len_zero`
- `test_run_round_integrity.py`
  - `test_run_round_uses_selection_score_for_tie_expansion`

Validation:
- `uv run pytest -q src/dnadesign/opal/tests/test_selection_contracts.py src/dnadesign/opal/tests/test_selection_result_validation.py`
- `uv run pytest -q src/dnadesign/opal/tests`
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs`

### 2026-02-16 UQ Correctness Hardening (GP/EI/Y-ops/SFXI)

What was broken:
- `gaussian_process.predict(std=False)` unpacked `(y, sd)` unconditionally, which breaks scalar prediction calls.
- GP scalar std payload could remain 1D and mismatch downstream 2D objective contracts.
- GP `load(params=None)` could attempt kernel rebuild from stringified estimator params.
- `expected_improvement` produced `0/0 -> NaN` for mixed cases where `sigma=0` and `improvement=0`.
- Runtime applied y-ops inverse transform to predicted means only; std dev stayed in model-space units.
- `sfxi_v1` uncertainty path used math inconsistent with the implemented score (missing `ln(2)` derivative factor, no score-exponent propagation, and all-OFF setpoint uncertainty collapsed to zero).

What changed:
- GP model:
  - `predict(std=False)` now only returns `y`, never unpacks std.
  - scalar outputs are normalized to shape `(n, 1)` for both mean and std.
  - std payload is validated finite/non-negative and appended immutably in RoundCtx.
  - `load(params=None)` now uses `est.get_params(deep=False)` and avoids kernel rebuild-from-string.
- EI:
  - keeps strict input validation, but handles per-candidate `sigma==0` with deterministic limit `A = alpha * max(I, 0)`.
  - all-zero uncertainty still fails fast.
- Y-ops uncertainty units:
  - added `inverse_std` support for `intensity_median_iqr`.
  - runtime now inverts both mean and std through y-ops in reverse order.
  - if EI is selected and a configured y-op lacks `inverse_std`, run hard-fails with explicit error.
  - for non-EI selection paths, missing `inverse_std` drops uncertainty propagation instead of using wrong units.
- SFXI uncertainty:
  - replaced prior block with delta-method variance on the actual scalar score:
    - `Var(score) ≈ Σ_k (dscore/dy_k)^2 * sigma_k^2`
  - derivatives include clipping masks and exponent terms from the implemented score.
  - all-OFF setpoint now propagates logic uncertainty correctly.

Regression tests added:
- `test_gaussian_process_model.py` (predict/std/load regressions)
- `test_expected_improvement_edge_cases.py` (mixed-zero sigma + all-zero rejection)
- `test_intensity_median_iqr.py` (inverse_std scaling semantics)
- `test_run_round_integrity.py::test_run_round_rejects_ei_when_yops_lacks_inverse_std`
- `test_run_round_integrity.py::test_run_round_gp_ei_with_intensity_yops_emits_finite_uncertainty`
- `test_objective_sfxi_v1.py` uncertainty regressions:
  - zero-std -> zero score uncertainty
  - all-OFF setpoint still reflects logic uncertainty
  - delta estimate matches Monte Carlo smoke tolerance

### 2026-02-16 Independent Math Audit Follow-up

Independent checks executed beyond the feature-targeted test suite:
- Randomized EI equation checks against direct formula evaluation (including mixed `sigma=0` cases).
- Multi-output GP predict/load shape checks (`(n,d)` mean/std payload invariants).
- Affine y-op Monte Carlo check confirming inverse mean/std propagation for `intensity_median_iqr`.
- SFXI uncertainty checks:
  - analytic implementation vs numerical finite-difference delta method (matched tightly),
  - delta-method estimate vs Monte Carlo score spread across sampled parameter regimes.

Findings:
- EI implementation matched equation-level expectations, but strict rejection of negative acquisition values was incorrect for weighted EI (`alpha != beta` can legitimately produce negative finite acquisition values).
- Fixed by removing the negative-acquisition hard-fail and adding regression coverage:
  - `test_expected_improvement_allows_finite_negative_weighted_acquisition`.
- SFXI uncertainty implementation is mathematically consistent with first-order delta method on the implemented score; observed Monte Carlo deviations are approximation error in nonlinear/clipped regions, not an implementation mismatch with the chosen method.

### 2026-02-15 Baseline Audit

- Branch: `Elm_UQ`
- Compared against: `main`
- Scope: OPAL-only diff (`src/dnadesign/opal/src/**`)
- Commit delta: 12 commits ahead, 0 behind

### Current focus

- Review Elm_UQ deltas by workstream.
- Identify integration and test failures before merge.
- Define smallest safe refactor/fix plan to merge into `main`.

### Notes

- Use this file for chronological notes during the next change set.
- Keep entries short, factual, and tied to specific files/commits.

### 2026-02-15 Contract V2 Execution Plan

### Plan

We are implementing a strict OPAL v2 contract that replaces ambiguous uncertainty semantics (`var`) with explicit channel-based abstractions for objective and selection. The approach is to define hard plugin/data contracts first, then refactor runtime and storage to consume those contracts, and finally lock behavior with tests and docs before merge.

### Scope
- In: `src/dnadesign/opal/src/**` objective plugin API, selection API, runtime round stages, config schemas, registry validation, ledger/writeback contracts, and OPAL docs/tests.
- Out: Non-OPAL packages, backward-compatibility shims, and unrelated performance tuning.

### Action items
[ ] Define explicit uncertainty semantics and naming (`y_pred_std`, `y_pred_var`, `score_ref`, `uncertainty_ref`) and remove generic `var` usage from public contracts.
[ ] Add a canonical typed objective result abstraction (v2) with named score/uncertainty channels and deterministic diagnostics fields.
[ ] Enforce objective plugin signature exactly as `(*, y_pred, params, ctx, train_view, y_pred_std)` at registry load time with fail-fast validation errors.
[ ] Extend config types/schemas from single `objective` to first-class `objectives: [...]`, validate uniqueness, and reject extra keys.
[ ] Require selection config to declare `score_ref` (and `uncertainty_ref` when acquisition policy needs it), then reject unresolved channel refs during preflight.
[ ] Refactor runtime scoring to evaluate all configured objectives, merge namespaced channels, and pass only explicit channels to selection.
[ ] Introduce a strict typed selection result validator and remove implicit `raw_sel["score"]` assumptions from staging/writeback paths.
[ ] Harden expected-improvement to fail on missing, non-finite, negative, or shape-invalid uncertainty and remove degrade-to-top-n behavior.
[ ] Update prediction ledger/writebacks to persist channel-qualified score/uncertainty outputs under strict allowlisted schema contracts.
[ ] Write failing tests first for each breaking contract (registry signature enforcement, multi-objective channel resolution, EI failure modes, selection result validation, ledger schema), then implement minimum passing code.
[ ] Run targeted OPAL tests, then full OPAL suite and CI; fix any failures at root cause before merge.
[ ] Update OPAL reference docs (configuration, models/selection plugins, objective docs) so semantics, contracts, and examples match v2 exactly.

### Open questions
- Should prediction ledger store channel outputs as wide typed columns (`pred__score__<channel>`) or as a typed map column with schema validation?
- Should objective channel names be globally unique strings, or namespaced by objective id and exposed through aliases?
- Should this land as one breaking PR (`Elm_UQ` -> `main`) or staged into two PRs (contracts first, runtime/storage second)?

### 2026-02-15 Compatibility + Docs Guardrails (Elm_UQ)

Context update:
- Elm_UQ introduces uncertainty-aware model/objective/selection behavior (GP + EI + uncertainty emissions).
- Requirement update: preserve pre-existing user-facing usage flows while documenting new UQ paths clearly.
- Primary baseline flow to preserve: SFXI demo (`random_forest` + greedy selection `top_n`).

Compatibility policy for this branch:
- Keep legacy end-to-end flows working without requiring users to learn new config fields to run old pipelines.
- Add new UQ capabilities as additive paths (`gaussian_process`, `expected_improvement`, uncertainty channels).
- Fail fast on invalid UQ configuration/data, but do not regress non-UQ default pipelines.

Documentation scope to update in lockstep:
- `src/dnadesign/opal/README.md`
- `src/dnadesign/opal/docs/reference/configuration.md`
- `src/dnadesign/opal/docs/plugins/models.md`
- `src/dnadesign/opal/docs/plugins/selection.md`
- `src/dnadesign/opal/docs/reference/data-contracts.md`
- `src/dnadesign/opal/docs/workflows/index.md`
- (optional new guide) `src/dnadesign/opal/docs/workflows/gp-sfxi-ei.md`

Usage-path matrix (must pass before merge):
- Path A (baseline): `random_forest` + `sfxi_v1` + `top_n` (no uncertainty path required).
- Path B (UQ model only): `gaussian_process` + scalar objective + `top_n`.
- Path C (full UQ): `gaussian_process` + uncertainty-emitting objective + `expected_improvement`.
- Path D (legacy campaign config): existing single-objective campaigns from pre-UQ docs still run.

Execution checkpoints:
- [ ] Add/restore compatibility handling in config/runtime for legacy objective/selection flows.
- [ ] Keep new channelized abstractions internal where possible; present simple defaults in docs.
- [ ] Update docs with explicit "Which path should I use?" decision guidance.
- [ ] Add/extend tests for all four usage paths above.
- [ ] Run targeted OPAL suites and record pass/fail status by path in this journal.

### 2026-02-15 Strict v2 Audit + Fix Pass (No Shim)

Decision update:
- This branch now treats OPAL schema migration as intentionally breaking v2.
- No backward-compatibility shim for deprecated selection aliases.
- Unified contracts: explicit objective channels + explicit selection channel refs.

Completed actions:
- [x] Removed legacy selection shim in round runtime and kept strict v2 selection contract enforcement.
- [x] Updated `verify-outputs` reporting/CLI to v2 score field (`pred__score_selected`) only.
- [x] Expanded strict GP config schema with typed kernel block and optimizer controls.
- [x] Updated objective registry signature validation to require strict v2 args while allowing optional keyword-only extensions.
- [x] Added channel-qualified prediction metrics to label history payloads (`score::<ref>`, `uncertainty::<ref>`).
- [x] Rewrote core v2 docs:
  - `README.md`
  - `docs/reference/configuration.md`
  - `docs/reference/data-contracts.md`
  - `docs/plugins/models.md`
  - `docs/plugins/selection.md`
  - `docs/workflows/rf-sfxi-topn.md`

Path matrix results (targeted tests):
- Path A: `random_forest + sfxi_v1 + top_n` -> PASS
- Path B: `gaussian_process + scalar_identity_v1 + top_n` -> PASS
- Path C: `gaussian_process + uncertainty objective + expected_improvement` -> PASS
- Strict rejection: invalid `score_ref` -> PASS
- Strict rejection: invalid `uncertainty_ref` -> PASS

Command set executed for this pass:
- `uv run pytest -q src/dnadesign/opal/tests/test_verify_outputs.py`
- `uv run pytest -q src/dnadesign/opal/tests/test_config_objectives_v2.py`
- `uv run pytest -q src/dnadesign/opal/tests/test_objective_contract_v2.py`
- `uv run pytest -q src/dnadesign/opal/tests/test_run_round_integrity.py`
- `uv run pytest -q src/dnadesign/opal/tests/test_ledger_dataset_writes.py`
- `uv run pytest -q src/dnadesign/opal/tests/test_analysis_facade_setpoint.py`
- `uv run pytest -q src/dnadesign/opal/tests/test_objective_sfxi_v1.py`
- `uv run pytest -q src/dnadesign/opal/tests/test_scalar_plugins.py`
- `uv run pytest -q src/dnadesign/opal/tests/test_roundctx_contracts.py`
- `uv run pytest -q src/dnadesign/opal/tests/test_pipeline_round_ctx.py`
- `uv run ruff check <touched OPAL src/tests files>`

### 2026-02-15 Full OPAL Validation (post-v2 cleanup)

- Fixed remaining v1-oriented test fixture assumptions by making `_cli_helpers.write_campaign_yaml` emit valid v2 selection params by default (`score_ref`, `objective_mode`, `tie_handling`, `uncertainty_ref` for EI).
- Updated dashboard utility test expectations to strict v2 behavior (ignore deprecated `objective` key, rely on `objective_mode` only).
- Confirmed no `pred__y_obj_scalar` references remain under `src/dnadesign/opal`.
- Confirmed demo workflow matrix configs exist and align with docs:
  - `campaign_rf_sfxi_topn.yaml`
  - `campaign_gp_topn.yaml`
  - `campaign_gp_ei.yaml`

Verification:
- `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests` -> PASS
- Workflow matrix test remains green:
  - `uv run pytest -q src/dnadesign/opal/tests/test_workflow_matrix_cli.py` -> PASS

Notes:
- GP workflow tests report sklearn `ConvergenceWarning` on noise-level lower bound; non-fatal and expected for current toy demo data.

### 2026-02-15 Demo CLI Pressure Test Matrix

Ran end-to-end CLI command chains against isolated copies of `src/dnadesign/opal/campaigns/demo` for each workflow config.

Command chain per workflow:
- `uv run opal init -c <config>`
- `uv run opal validate -c <config>`
- `uv run opal ingest-y -c <config> --round 0 --csv inputs/r0/vec8-b0.xlsx --unknown-sequences drop --if-exists replace --yes`
- `uv run opal run -c <config> --round 0`
- `uv run opal verify-outputs -c <config> --round latest`
- `uv run opal status -c <config>`
- `uv run opal runs list -c <config>`

Matrix outcomes:
- `campaign_rf_sfxi_topn.yaml` -> PASS
- `campaign_gp_topn.yaml` -> PASS
- `campaign_gp_ei.yaml` -> PASS

Observed behavior checks:
- `verify-outputs` mismatch count was `0` for all three runs.
- `runs list` reflected expected selection strategies:
  - RF path: `top_n`
  - GP top_n path: `top_n`
  - GP EI path: `expected_improvement`
- GP paths emitted sklearn `ConvergenceWarning` for kernel noise-level lower bound; run completed and contracts remained valid.

### 2026-02-15 Round-1 CLI Pressure Test Matrix

Initial check:
- `run --round 1` without `ingest-y --round 1` fails for SFXI flows by design because `sfxi_v1` scaling enforces `min_n` on the current-round label pool.
- Error observed:
  - `[sfxi_v1] Need at least min_n=5 labels in current round to scale intensity; got 0.`

Validated command chain per workflow for round 1:
- `uv run opal ingest-y -c <config> --round 1 --csv inputs/r0/vec8-b0.xlsx --unknown-sequences drop --if-exists replace --yes`
- `uv run opal run -c <config> --round 1 --resume`
- `uv run opal verify-outputs -c <config> --round latest`
- `uv run opal status -c <config>`

Round-1 matrix outcomes:
- `campaign_rf_sfxi_topn.yaml` -> PASS
- `campaign_gp_topn.yaml` -> PASS
- `campaign_gp_ei.yaml` -> PASS

Observed behavior checks:
- `verify-outputs` mismatch count remained `0` in all cases.
- `status` reported `num_rounds: 2` and latest round `r=1` for all three workflows.

### 2026-02-15 Campaign-Scoped Demo IA Reorganization

Goal:
- Move from one-config-many-flows to one-campaign-per-flow demos with clearer docs and command paths.

Implemented structure:
- New campaign-scoped demos:
  - `src/dnadesign/opal/campaigns/demo_rf_sfxi_topn/`
  - `src/dnadesign/opal/campaigns/demo_gp_topn/`
  - `src/dnadesign/opal/campaigns/demo_gp_ei/`
- Each includes:
  - `configs/campaign.yaml`
  - `configs/plots.yaml`
  - `inputs/r0/vec8-b0.xlsx`
  - `README.md`
- Campaign-local `records.parquet` is intentionally bootstrapped at runtime (`cp ../demo/records.parquet ./records.parquet`) so each flow stays isolated.

Docs IA changes:
- Added demos hub:
  - `src/dnadesign/opal/docs/workflows/index.md`
- Added dedicated flow guides:
  - `src/dnadesign/opal/docs/workflows/rf-sfxi-topn.md`
  - `src/dnadesign/opal/docs/workflows/gp-sfxi-topn.md`
  - `src/dnadesign/opal/docs/workflows/gp-sfxi-ei.md`
- Repositioned matrix page to route to flow guides:
  - `src/dnadesign/opal/docs/workflows/index.md`
- Updated doc entrypoints:
  - `src/dnadesign/opal/docs/index.md`
  - `src/dnadesign/opal/README.md`

File-organization guardrails:
- Added `src/dnadesign/opal/campaigns/.gitignore` to ignore campaign artifacts and local demo data copies.
- Added `src/dnadesign/opal/.gitignore` to suppress `.DS_Store` cruft.

Pressure-test result (new campaign-scoped flows):
- Executed documented round-0 sequence for all three new campaigns in isolated temp copies.
- For each flow, `init -> validate -> ingest-y -> run -> verify-outputs -> status -> runs list` passed.
- `verify-outputs` mismatch count was `0` for all three flows.

### 2026-02-15 UX/Schema/Registry Audit Follow-up

Audit-driven fixes applied:
- Fixed a GP correctness/performance footgun in batched prediction uncertainty accumulation:
  - prior behavior used `np.vstack` per batch in-model, which fails for scalar-output uneven batch sizes and scales poorly.
  - model now appends uncertainty chunks in `RoundCtx`; stage scoring coalesces chunks once.
- Clarified and enforced EI uncertainty semantics:
  - `expected_improvement` now treats `scalar_uncertainty` as standard deviation directly.
  - added strict finite/non-negative/all-zero and alpha/beta validation.
- Updated `sfxi_v1` to emit score uncertainty as standard deviation (derived from validated variance).
  - added explicit `y_pred_std` shape check against `y_pred`.
- Tightened dashboard selection reconstruction:
  - `selection.params.objective_mode` is now required (no implicit default).
  - `selection.params.top_k` is now required for reconstruction (no implicit default).

New regression coverage:
- EI contract semantics:
  - `test_expected_improvement_uses_uncertainty_as_std`
- GP multi-batch scalar-output path:
  - `test_run_round_matrix_gp_topn_handles_scalar_uncertainty_multibatch`
- Dashboard objective-mode strictness:
  - updated `test_resolve_objective_mode_aliases` to assert strict v2 behavior.

Docs updates:
- `docs/plugins/selection.md` clarifies EI requires std-dev uncertainty channel.
- `docs/reference/configuration.md` clarifies `uncertainty_ref` semantics.
- `docs/plugins/models.md` clarifies EI channel expectation.
- `docs/plugins/objective-sfxi.md` now documents `pred__uncertainty_selected`.

Validation for this pass:
- `uv run pytest -q src/dnadesign/opal/tests/test_objective_contract_v2.py::test_expected_improvement_uses_uncertainty_as_std src/dnadesign/opal/tests/test_dashboard_utils.py::test_resolve_objective_mode_aliases src/dnadesign/opal/tests/test_run_round_integrity.py::test_run_round_matrix_gp_topn_handles_scalar_uncertainty_multibatch` -> PASS
- `uv run pytest -q src/dnadesign/opal/tests/test_objective_contract_v2.py src/dnadesign/opal/tests/test_run_round_integrity.py src/dnadesign/opal/tests/test_dashboard_utils.py src/dnadesign/opal/tests/test_workflow_matrix_cli.py` -> PASS
- `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS

Residual known warning:
- sklearn GP `ConvergenceWarning` on demo data noise lower bound remains non-fatal.

### 2026-02-15 Strict hardening pass: remove remaining selection/runtime fallbacks

Goal:
- Remove two remaining fallback-style behaviors identified in audit:
  1. selection registry TypeError masking during factory/function resolution;
  2. runtime implicit default for `selection.params.objective_mode`.

Implemented changes:
- `src/dnadesign/opal/src/registries/selection.py`
  - Removed broad `TypeError` fallback path in `get_selection`.
  - Added explicit callable-shape validation (`ids`, `scores`, `top_k`) to distinguish direct strategies vs factories.
  - Factory construction errors now propagate instead of being silently treated as strategy callables.
  - Added strict validation that factory return values are callables with selection surface.
- `src/dnadesign/opal/src/runtime/round/stages.py`
  - `_resolve_selection_objective_mode` now requires explicit `selection.params.objective_mode`.
  - Missing/null objective mode now raises `OpalError` instead of defaulting to `maximize`.

Test updates (TDD):
- Added `test_get_selection_does_not_mask_factory_typeerror` in `src/dnadesign/opal/tests/test_roundctx_contracts.py`.
- Added `test_run_round_rejects_missing_objective_mode` in `src/dnadesign/opal/tests/test_run_round_integrity.py`.
- Updated helper defaults in `test_run_round_integrity.py` to include explicit `objective_mode` for standard fixtures.
- Updated `src/dnadesign/opal/tests/test_pipeline_round_ctx.py` configs to pass explicit `objective_mode`.

Validation:
- Focused red->green tests:
  - `uv run pytest -q src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_does_not_mask_factory_typeerror src/dnadesign/opal/tests/test_run_round_integrity.py::test_run_round_rejects_missing_objective_mode`
- Broader suites:
  - `uv run pytest -q src/dnadesign/opal/tests/test_roundctx_contracts.py src/dnadesign/opal/tests/test_run_round_integrity.py src/dnadesign/opal/tests/test_workflow_matrix_cli.py src/dnadesign/opal/tests/test_config_objectives_v2.py`
  - `uv run pytest -q src/dnadesign/opal/tests`
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs`

Result:
- PASS on all OPAL tests and lint.
- GP convergence warnings remain non-fatal on demo data.

### 2026-02-15 Strictness follow-up: selection mode/tie and summary rendering alignment

Goal:
- Remove remaining implicit selection defaults in runtime-adjacent paths so v2 contracts are explicit even for programmatic config construction.

Implemented changes:
- `src/dnadesign/opal/src/runtime/round/stages.py`
  - Added `_resolve_selection_tie_handling` and now requires explicit `selection.params.tie_handling`.
  - Removed implicit `top_k` fallback path when not using CLI `--k`; now requires explicit `selection.params.top_k`.
- `src/dnadesign/opal/src/registries/selection.py`
  - Selection callable wrapper now requires explicit `objective` and `tie_handling` keyword arguments (no wrapper defaults).
- `src/dnadesign/opal/src/selection/top_n.py`
  - Updated plugin signature to require explicit `objective` and `tie_handling`.
  - Updated module header authorship ordering for uncertainty-aware branch attribution.
- `src/dnadesign/opal/src/selection/expected_improvement.py`
  - Updated plugin signature to require explicit `objective` and `tie_handling`.
- `src/dnadesign/opal/src/cli/commands/run.py`
  - Removed summary-level defaults for `tie_handling` and `objective_mode`; summary now enforces required/valid values.
- `src/dnadesign/opal/docs/plugins/selection.md`
  - Updated runtime signature docs to match strict required args (no default objective/tie values).

TDD additions:
- Added `test_run_round_rejects_missing_tie_handling` in `src/dnadesign/opal/tests/test_run_round_integrity.py`.
- Added `test_selection_call_requires_explicit_mode_and_tie` in `src/dnadesign/opal/tests/test_roundctx_contracts.py`.
- Updated affected tests/config fixtures to include explicit `tie_handling` in strict v2 configs:
  - `src/dnadesign/opal/tests/test_pipeline_round_ctx.py`
  - `src/dnadesign/opal/tests/test_roundctx_contracts.py`
  - `src/dnadesign/opal/tests/test_run_round_integrity.py`

Validation:
- Red -> green (new tests):
  - `uv run pytest -q src/dnadesign/opal/tests/test_run_round_integrity.py::test_run_round_rejects_missing_tie_handling src/dnadesign/opal/tests/test_roundctx_contracts.py::test_selection_call_requires_explicit_mode_and_tie`
- Broader targeted:
  - `uv run pytest -q src/dnadesign/opal/tests/test_roundctx_contracts.py src/dnadesign/opal/tests/test_run_round_integrity.py src/dnadesign/opal/tests/test_objective_contract_v2.py src/dnadesign/opal/tests/test_workflow_matrix_cli.py`
- Full gate:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS

Notes:
- GP `ConvergenceWarning` on demo toy data remains non-fatal and unchanged.

### 2026-02-15 Renderer strictness cleanup (selection summary)

Goal:
- Remove remaining display-layer fallback defaults that could hide invalid selection summary payloads.

Implemented changes:
- `src/dnadesign/opal/src/cli/formatting/renderers/run.py`
  - Added required-field validation for `tie_handling` and `objective_mode` in `render_run_summary_human`.
  - Removed fallback display defaults (`competition_rank`, `maximize`) so summary rendering fails fast on malformed payloads.

TDD additions:
- Added `src/dnadesign/opal/tests/test_cli_run_renderer.py`:
  - `test_render_run_summary_requires_selection_mode_and_tie`
  - `test_render_run_summary_uses_explicit_mode_and_tie`

Validation:
- `uv run pytest -q src/dnadesign/opal/tests/test_cli_run_renderer.py` -> PASS
- `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS

### 2026-02-16 Strict no-fallback sweep + demo pressure retest

Goal:
- Remove remaining implicit fallback behavior in objective/selection contracts and re-validate all documented demo flows end-to-end.

Findings addressed:
- Objective score channel modes were implicitly defaulting to `maximize` when `modes_by_name` omitted a channel.
- `top_n` accepted non-contract objective aliases (for example `max`) via `startswith` semantics.
- Selection rank normalization accepted invalid `tie_handling` values by silently treating them as competition rank.
- Demo docs lacked a concise copy/paste matrix for re-running all flow guides in isolated temp workspaces.

Implemented changes:
- `src/dnadesign/opal/src/core/objective_result.py`
  - `validate_objective_result_v2` now requires explicit `modes_by_name` entries for every emitted score channel.
- `src/dnadesign/opal/src/selection/top_n.py`
  - Added strict objective mode normalization (`maximize|minimize` only), removed fuzzy prefix interpretation.
- `src/dnadesign/opal/src/registries/selection.py`
  - Added strict objective-mode validation in `_stable_sort_indices`.
  - Added strict `tie_handling` validation in `_ranks_from_sorted_scores` (invalid values now error).
- `src/dnadesign/opal/docs/workflows/index.md`
  - Added a quick pressure-test matrix with sequential commands for all three campaign-scoped demo flows.

TDD updates:
- `src/dnadesign/opal/tests/test_objective_contract_v2.py`
  - Added `test_objective_result_v2_requires_explicit_mode_per_score_channel`.
  - Added `test_top_n_rejects_invalid_objective_mode`.
  - Added `test_selection_normalizer_rejects_invalid_tie_handling`.

Validation:
- Red -> green targeted tests:
  - `uv run pytest -q src/dnadesign/opal/tests/test_objective_contract_v2.py::test_objective_result_v2_requires_explicit_mode_per_score_channel src/dnadesign/opal/tests/test_objective_contract_v2.py::test_top_n_rejects_invalid_objective_mode src/dnadesign/opal/tests/test_objective_contract_v2.py::test_selection_normalizer_rejects_invalid_tie_handling`
- Full package gate:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS

Demo flow pressure retest (isolated copies, round 0 and round 1):
- Root temp workspace: `/private/tmp/opal-demo-audit-MwfqGI`
- Flows:
  - `demo_rf_sfxi_topn`: verify mismatch count `0` at r0 and r1
  - `demo_gp_topn`: verify mismatch count `0` at r0 and r1
  - `demo_gp_ei`: verify mismatch count `0` at r0 and r1
- Data-sufficiency behavior check:
  - GP top_n vs GP EI selected IDs at round 1 diverged (`intersection=3`, `topn_only=2`, `ei_only=2`), confirming demo inputs support selector-behavior differentiation.

### 2026-02-16 Maintainer hardening pass: verify integrity + mode strictness + profiling

Goal:
- Remove remaining verification/runtime footguns and re-confirm demo ergonomics plus end-to-end behavior on current branch state.

Findings fixed:
- `verify-outputs` compared selection against ledger with an inner join and only failed when overlap was empty.
  - Risk: unknown selection IDs could be silently dropped and still pass verification.
- Selection score-channel mode propagation in scoring still used a fallback-style `.get(..., \"maximize\")` despite strict mode contracts.
  - Risk: contract intent not fully reflected in runtime.

Implemented changes:
- `src/dnadesign/opal/src/reporting/verify_outputs.py`
  - Added strict duplicate-ID rejection for selection tables.
  - Added strict unknown-ID rejection when selection IDs are absent from run ledger predictions.
- `src/dnadesign/opal/src/runtime/round/stages.py`
  - Removed fallback mode propagation; score channel modes are now read directly from validated objective output.
- `src/dnadesign/opal/docs/workflows/index.md`
  - Expanded pressure-test matrix to include round-1 continuation.
  - Added a compact GP top_n vs GP EI selected-ID comparison snippet to verify data sufficiency for selector differentiation.

TDD additions:
- `src/dnadesign/opal/tests/test_verify_outputs.py`
  - `test_compare_selection_to_ledger_rejects_unknown_selection_ids`
  - `test_compare_selection_to_ledger_rejects_duplicate_selection_ids`

Validation:
- Targeted:
  - `uv run pytest -q src/dnadesign/opal/tests/test_verify_outputs.py::test_compare_selection_to_ledger_rejects_unknown_selection_ids src/dnadesign/opal/tests/test_verify_outputs.py::test_compare_selection_to_ledger_rejects_duplicate_selection_ids`
  - `uv run pytest -q src/dnadesign/opal/tests/test_verify_outputs.py src/dnadesign/opal/tests/test_run_round_integrity.py::test_run_round_matrix_gp_ei_path src/dnadesign/opal/tests/test_objective_contract_v2.py::test_objective_result_v2_requires_explicit_mode_per_score_channel`
- Full package:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS

Demo pressure test (isolated temp copy, r0+r1):
- Root temp workspace: `/tmp/opal-demo-audit-phYyKP`
- All flows passed with verify mismatch count `0` at both round 0 and round 1:
  - `demo_rf_sfxi_topn`
  - `demo_gp_topn`
  - `demo_gp_ei`
- GP selector differentiation remained present on this rerun:
  - `intersection=3`, `topn_only=2`, `ei_only=2`

Performance profiling (cProfile on GP+EI round-1 run path):
- Command: profile `cmd_run(... round=1, resume=True)` against temp campaign config.
- Top cumulative hotspots (same pattern as prior pass):
  - `storage/label_history.py:_deep_as_py`
  - `storage/label_history.py:normalize_hist_cell`
  - `runtime/round/stages.py:stage_training`
  - `storage/label_history.py:parse_hist_cell_strict`
- Takeaway: dominant latency remains label-history normalization/parsing, not selection/EI path.

### 2026-02-16 Documentation clarity and schema-semantics pass

Goal:
- Make OPAL docs more didactic and less noisy while keeping strict semantic alignment with current code paths and schema behavior.

Findings addressed:
- Primary docs entrypoints still linked to non-runtime objective docs (`spop`) even though it is not a built-in registered objective.
- The docs did not have one concise page explaining model/selection pairings and practical flow choices.
- Configuration docs mixed strict contract language with defaults without clarifying intent, which could confuse reviewers.
- Demo docs already had command sequences but lacked explicit round-1 continuation in the matrix page.

Implemented docs changes:
- Added concise strategy matrix page:
  - `src/dnadesign/opal/docs/concepts/strategy-matrix.md`
  - Covers model/objective/selection wiring, built-in flow choices, and EI uncertainty semantics.
- Updated primary docs indexes and README maps:
  - `src/dnadesign/opal/docs/index.md`
  - `src/dnadesign/opal/README.md`
  - Removed `spop` from primary docs navigation (kept file but no longer presented as standard path).
- Tightened config/reference semantics:
  - `src/dnadesign/opal/docs/reference/configuration.md`
  - Clarified explicit selection keys and model->objective->selection wiring.
- Updated plugin references for selection/optimizer communication:
  - `src/dnadesign/opal/docs/plugins/models.md`
  - `src/dnadesign/opal/docs/plugins/selection.md`
- Updated data-contract reference to include strict verify behavior:
  - `src/dnadesign/opal/docs/reference/data-contracts.md`
- Expanded demo matrix to include full round-1 continuation and selector-difference check snippet:
  - `src/dnadesign/opal/docs/workflows/index.md`

Validation:
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS
- `uv run pytest -q src/dnadesign/opal/tests/test_verify_outputs.py src/dnadesign/opal/tests/test_workflow_matrix_cli.py` -> PASS

Pressure test (real CLI flows in isolated temp workspace):
- Temp root: `/tmp/opal-doc-pass-TVl2zG`
- Flows passed end-to-end for round 0 and round 1:
  - `demo_rf_sfxi_topn`
  - `demo_gp_topn`
  - `demo_gp_ei`
- Selector differentiation check remained stable:
  - `intersection=3`, `topn_only=2`, `ei_only=2`

### 2026-02-16 Doc semantics and demo command reproducibility pass

Goal:
- Remove doc footguns and ensure command snippets execute as written for all documented OPAL demo flows.

Findings fixed:
- Demo matrix snippet consumed `verify_r1.json` files that were never produced by the documented command loop.
- Architecture concept page still described a single-objective flow and hid channel-ref semantics.
- SFXI objective doc described `selection__score_ref` as a data-field ref (`pred__score_selected`) instead of a channel ref.
- Data-contract and CLI reference pages needed explicit wording for `selection__objective` semantics and `verify-outputs --json` stdout behavior.

Changes:
- `src/dnadesign/opal/docs/workflows/index.md`
  - Updated pressure-test loop to write `verify_r0.json` / `verify_r1.json`.
  - Added `tmp_root` echo to support the follow-up selector comparison snippet.
- `src/dnadesign/opal/docs/concepts/architecture.md`
  - Updated flow to explicit channel-driven v2 semantics (`score_ref`, `uncertainty_ref`).
- `src/dnadesign/opal/docs/plugins/objective-sfxi.md`
  - Corrected run_meta selection-field semantics.
- `src/dnadesign/opal/docs/reference/data-contracts.md`
  - Clarified `selection__objective` stores objective mode.
- `src/dnadesign/opal/docs/reference/cli.md`
  - Clarified `verify-outputs --json` writes to stdout (redirect when file output is needed).

Validation:
- `uv run ruff check src/dnadesign/opal/docs src/dnadesign/opal/src src/dnadesign/opal/tests` -> PASS
- `uv run pytest -q src/dnadesign/opal/tests/test_workflow_matrix_cli.py src/dnadesign/opal/tests/test_verify_outputs.py` -> PASS

Pressure test (live CLI demo matrix, round 0 + round 1 in isolated workspace):
- Temp root: `/tmp/opal-demo-audit-docpass-pF76c0`
- `demo_rf_sfxi_topn`: `mismatches=0`, `rows=5`
- `demo_gp_topn`: `mismatches=0`, `rows=5`
- `demo_gp_ei`: `mismatches=0`, `rows=5`
- GP selector differentiation remained present:
  - `intersection=3`
  - `topn_only=2`
  - `ei_only=2`

### 2026-02-16 Schema + demo UX hardening pass

Goal:
- Remove dead-end config surface, fail fast earlier for invalid plugin names, and align demo docs with full user-facing CLI flow.

Changes:
- Fail-fast plugin registry checks now run in `opal validate` before data checks:
  - `src/dnadesign/opal/src/cli/commands/validate.py`
  - validates configured `transform_x`, `transform_y`, `model`, `objectives[*]`, and `selection` names against loaded registries.
- Removed unused `metadata.notes` config surface:
  - `src/dnadesign/opal/src/config/types.py`
  - `src/dnadesign/opal/src/config/loader.py`
  - campaign configs updated to remove `metadata` blocks.
- Fixed stale marker cruft:
  - `src/dnadesign/opal/campaigns/demo/.opal/config` -> `configs/campaign.yaml`
  - removed stale marker file `src/dnadesign/opal/campaigns/prom60-etoh-cipro-andgate/.opal/config`
- Demo information architecture + command narratives updated:
  - `src/dnadesign/opal/docs/workflows/index.md`
  - `src/dnadesign/opal/docs/workflows/rf-sfxi-topn.md`
  - `src/dnadesign/opal/docs/workflows/gp-sfxi-topn.md`
  - `src/dnadesign/opal/docs/workflows/gp-sfxi-ei.md`
  - `src/dnadesign/opal/docs/workflows/index.md`
  - now includes optional `campaign-reset`, then `init/validate/ingest/run/verify`, and in-sequence `ctx`, `explain`, `record-show`, `predict`, `plot`.
- Added explicit RF demo coverage for `feature_importance_bars`:
  - `src/dnadesign/opal/campaigns/demo_rf_sfxi_topn/configs/plots.yaml`
  - `src/dnadesign/opal/docs/reference/plots.md`

TDD + validation:
- Added failing test first, then implementation:
  - `src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_unknown_plugin_names`
- Targeted suite:
  - `uv run pytest -q src/dnadesign/opal/tests/test_cli_workflows.py src/dnadesign/opal/tests/test_cli_config_discovery.py src/dnadesign/opal/tests/test_run_round_integrity.py src/dnadesign/opal/tests/test_pipeline_round_ctx.py` -> PASS
- Full OPAL suite:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS

Pressure-test matrix (updated command path) in isolated workspace:
- temp root: `/tmp/opal-demo-audit-Xh1cfV`
- all flows passed round 0 and round 1 with mismatch count `0`:
  - `demo_rf_sfxi_topn`
  - `demo_gp_topn`
  - `demo_gp_ei`
- GP selector differentiation still present on r1:
  - `intersection=3`, `topn_only=2`, `ei_only=2`

### 2026-02-16 Demo doc simplification pass (no one-off flags, no Python snippets)

Goal:
- Keep demo guides on a strict, simple happy path with only standard CLI/shell commands.

Changes:
- Removed `opal campaign-reset --allow-non-demo --yes` from demo guides and matrix docs.
- Replaced reset step with explicit generated-artifact cleanup:
  - `rm -rf outputs state.json notebooks`
- Removed embedded Python heredoc snippets from demos.
- Replaced selected-id extraction with shell-only CSV parsing:
  - `tail -n +2 .../selection_top_k.csv | head -n 1 | cut -d, -f1`
- Removed the Python-based GP selector comparison block from demo index.

Touched docs:
- `src/dnadesign/opal/docs/workflows/index.md`
- `src/dnadesign/opal/docs/workflows/rf-sfxi-topn.md`
- `src/dnadesign/opal/docs/workflows/gp-sfxi-topn.md`
- `src/dnadesign/opal/docs/workflows/gp-sfxi-ei.md`
- `src/dnadesign/opal/docs/workflows/index.md`

Validation:
- `rg -n "allow-non-demo|uv run python|<<'PY'" src/dnadesign/opal/docs/workflows src/dnadesign/opal/docs/workflows/index.md` -> no matches
- `uv run ruff check src/dnadesign/opal/docs` -> PASS

### 2026-02-16 YAGNI config-resolution cleanup pass

Goal:
- Remove implicit campaign marker/discovery behavior and keep campaign resolution explicit (`--config` or `$OPAL_CONFIG`).

Changes:
- Removed workspace marker creation from init:
  - `src/dnadesign/opal/src/cli/commands/init.py`
- Updated init human output to only report `outputs/` scaffolding:
  - `src/dnadesign/opal/src/cli/formatting/renderers/init.py`
- Removed marker/cwd/fallback config resolution paths; now strict explicit config/env only:
  - `src/dnadesign/opal/src/core/config_resolve.py`
- Removed unused workspace properties:
  - `src/dnadesign/opal/src/storage/workspace.py`
  - removed `inputs_dir`
  - removed `marker_path`
- Removed hidden demo fallback in `campaign-reset`; now also requires explicit config/env:
  - `src/dnadesign/opal/src/cli/commands/campaign_reset.py`
- Updated tests for strict resolver behavior and init scaffolding:
  - `src/dnadesign/opal/tests/test_cli_config_discovery.py`
  - `src/dnadesign/opal/tests/test_cli_workflows.py`
  - `src/dnadesign/opal/tests/test_cli_campaign_reset.py`
- Updated user-facing docs to match behavior:
  - `src/dnadesign/opal/README.md`
  - `src/dnadesign/opal/docs/reference/cli.md`

Validation:
- `uv run pytest -q src/dnadesign/opal/tests/test_cli_campaign_reset.py::test_campaign_reset_requires_config_or_env` -> FAIL (expected, before implementation)
- `uv run pytest -q src/dnadesign/opal/tests/test_cli_campaign_reset.py::test_campaign_reset_requires_config_or_env` -> PASS (after implementation)
- `uv run pytest -q src/dnadesign/opal/tests/test_cli_config_discovery.py src/dnadesign/opal/tests/test_cli_workflows.py src/dnadesign/opal/tests/test_cli_campaign_reset.py` -> PASS
- `uv run ruff check src/dnadesign/opal/src/cli/commands/campaign_reset.py src/dnadesign/opal/src/core/config_resolve.py src/dnadesign/opal/src/cli/commands/init.py src/dnadesign/opal/src/cli/formatting/renderers/init.py src/dnadesign/opal/src/storage/workspace.py src/dnadesign/opal/tests/test_cli_config_discovery.py src/dnadesign/opal/tests/test_cli_workflows.py src/dnadesign/opal/tests/test_cli_campaign_reset.py` -> PASS

### 2026-02-16 YAGNI + correctness verification pass (post-cleanup)

Goal:
- Verify strict config-resolution changes under full OPAL coverage and real demo workflows, and remove notebook crash footgun when label history is empty.

Changes:
- Notebook robustness for empty label history in SFXI label panel:
  - `src/dnadesign/opal/notebooks/prom60_eda.py`
  - when `opal_labels_view_df` is empty, show explicit notice and keep notebook runnable.
- Restored accidental mutation of demo records after validating root cause:
  - root cause was prior `campaign-reset` default-config fallback path (now removed).
  - `src/dnadesign/opal/campaigns/demo/records.parquet` restored to branch baseline.

Validation:
- `uv run pytest -q src/dnadesign/opal/tests/test_promoter_eda_notebook_smoke.py::test_prom60_eda_headless` -> PASS
- `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs src/dnadesign/opal/notebooks` -> PASS
- Demo pressure matrix (RF+top_n, GP+top_n, GP+EI; rounds 0 and 1) -> PASS
  - `verify_r0.json` and `verify_r1.json` mismatch arrays were empty for all flows.

### 2026-02-16 docs IA + UX footgun cleanup pass

Goal:
- Remove remaining documentation UX footguns and ensure docs semantics match current v2 runtime behavior.

Changes:
- Fixed docs hub maintainer navigation:
  - `src/dnadesign/opal/docs/index.md`
  - `Dev journal` link now points to `docs/dev/journal.md` (was stale `docs/internal/journal.md`).
- Corrected stale plot example field:
  - `src/dnadesign/opal/docs/reference/plots.md`
  - `scatter_score_vs_rank` example now uses `score_field: "pred__score_selected"` with explicit note on expected default usage.
- Tightened demo terminology consistency:
  - `src/dnadesign/opal/docs/workflows/gp-sfxi-topn.md`
  - `src/dnadesign/opal/docs/workflows/gp-sfxi-ei.md`
  - Link label standardized to `Model plugins`.

Validation:
- Docs stale-term sweep:
  - `rg -n -e "internal/journal\\.md|score_sfxi|Models plugins" src/dnadesign/opal/docs src/dnadesign/opal/README.md --glob '!src/dnadesign/opal/docs/dev/**' --glob '!src/dnadesign/opal/docs/internal/**'`
  - result: no matches.
- Markdown relative-link check across `src/dnadesign/opal/README.md` and `src/dnadesign/opal/docs/**/*.md`:
  - result: `OK: no missing relative markdown links`.
- Runtime contract spot checks:
  - `uv run opal plot --describe scatter_score_vs_rank` confirms default `score_field=pred__score_selected`.
  - `uv run opal campaign-reset --help` confirms documented `--apply` / `--allow-non-demo` semantics.
- Full demo workflow pressure test (isolated temp copies, round 0 flow path):
  - temp root: `/tmp/opal-doc-pass-6RkXlq`
  - flows passed: `demo_rf_sfxi_topn`, `demo_gp_topn`, `demo_gp_ei`
  - `verify-outputs` JSON mismatch count: `0` for all three flows.

### 2026-02-16 strict selection-contract hardening + overlay refactor pass

Goal:
- Remove remaining implicit selection defaults across config validation and dashboard reconstruction, and tighten channel-selection helpers to explicit contracts.

Changes:
- Selection schema strictness (no implicit defaults for required contract fields):
  - `src/dnadesign/opal/src/config/plugin_schemas.py`
  - `top_n` and `expected_improvement` now require explicit `top_k`, `objective_mode`, and `tie_handling`.
- Dashboard selection reconstruction strictness:
  - `src/dnadesign/opal/src/analysis/dashboard/selection.py`
  - `compute_selection_overlay` now requires explicit `selection.params.tie_handling` and validates allowed modes.
- Selection normalizer contract + hot-path optimization:
  - `src/dnadesign/opal/src/registries/selection.py`
  - removed fallback defaults from `normalize_selection_result` signature (`tie_handling`, `objective` now explicit),
  - avoids duplicate rank computation when requested tie mode is already `competition_rank`.
- Docs alignment:
  - `src/dnadesign/opal/docs/reference/configuration.md`
  - updated selection-contract wording to state no schema defaults for required fields.
- Fixture/test updates for explicit selection params:
  - `src/dnadesign/opal/tests/test_cli_workflows.py`
  - `src/dnadesign/opal/tests/test_dashboard_utils.py`
  - `src/dnadesign/opal/tests/test_config_objectives_v2.py`
  - `src/dnadesign/opal/tests/test_verify_outputs.py`

TDD:
- Added failing tests first:
  - `test_validate_requires_explicit_selection_contract_fields`
  - `test_compute_selection_overlay_requires_explicit_tie_handling`
- Confirmed failures before implementation, then implemented minimal changes until green.

Validation:
- Targeted strictness tests:
  - `uv run pytest -q src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_requires_explicit_selection_contract_fields src/dnadesign/opal/tests/test_dashboard_utils.py::test_compute_selection_overlay_requires_explicit_tie_handling` -> PASS
- Broader impacted suites:
  - `uv run pytest -q src/dnadesign/opal/tests/test_cli_workflows.py src/dnadesign/opal/tests/test_dashboard_utils.py src/dnadesign/opal/tests/test_objective_contract_v2.py src/dnadesign/opal/tests/test_workflow_matrix_cli.py` -> PASS
  - `uv run pytest -q src/dnadesign/opal/tests/test_config_objectives_v2.py src/dnadesign/opal/tests/test_verify_outputs.py::test_end_to_end_run_and_verify_outputs` -> PASS
- Full OPAL suite:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests` -> PASS

### 2026-02-16 critical audit pass (UX footguns + strict history contracts)

Goal:
- Run a full OPAL package audit with real CLI usage paths, then remove stale tracked campaign artifacts and harden label-history correctness on malformed existing cells.

Changes:
- Strict label-history handling (no silent normalization of malformed existing history cells):
  - `src/dnadesign/opal/src/storage/label_history.py`
  - `append_labels_from_df`, `training_labels_with_round`, and `append_predictions_from_arrays` now parse existing `label_hist` with `parse_hist_cell_strict` and raise `OpalError` on malformed cells.
- Added tests for malformed-history fail-fast behavior:
  - `src/dnadesign/opal/tests/test_training_policy.py`
  - `src/dnadesign/opal/tests/test_label_hist_validation.py`
- Removed stale tracked campaign artifacts:
  - deleted `src/dnadesign/opal/campaigns/demo/.opal/config` (marker path is intentionally ignored by current resolver contract).
  - deleted `src/dnadesign/opal/campaigns/prom60-etoh-cipro-andgate/state.json` (run artifact; absolute-path stale data).

Validation:
- Targeted TDD checks:
  - `uv run pytest -q src/dnadesign/opal/tests/test_training_policy.py::test_training_policy_rejects_malformed_label_history src/dnadesign/opal/tests/test_label_hist_validation.py::test_append_predictions_rejects_malformed_existing_label_history src/dnadesign/opal/tests/test_label_hist_validation.py::test_append_labels_rejects_malformed_existing_label_history` -> PASS
- Full OPAL suite:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS
- End-to-end CLI pressure test against demo docs command path (RF+top_n, GP+top_n, GP+EI; rounds 0 and 1):
  - all flows reported `FLOW_OK`.
  - temp run root: `/tmp/opal-demo-audit-m8EawP`.
  - `verify-outputs` JSON files written for r0/r1 in each temp flow and passed.

### 2026-02-16 plugin-name fail-fast preflight in shared CLI config loader

Goal:
- Eliminate fail-late unknown-plugin behavior where bad plugin names could pass `init` and only fail deep inside `run`.

Changes:
- Added shared plugin registry name preflight in CLI common loader:
  - `src/dnadesign/opal/src/cli/commands/_common.py`
  - `load_cli_config` now validates configured `transform_x`, `transform_y`, `model`, `selection`, and each `objective` name against active registries before command execution.
- Removed duplicated plugin-name validation logic from `validate` command:
  - `src/dnadesign/opal/src/cli/commands/validate.py`
  - `validate` now relies on shared loader preflight for the same strict contract.
- Added TDD coverage for unknown-model rejection at `init` time:
  - `src/dnadesign/opal/tests/test_cli_config_discovery.py::test_init_rejects_unknown_model_plugin`

Root-cause reproduction:
- Unknown model names were accepted by config load and `init`, and `run` failed only after expensive pipeline prep (`X_train`/`X_pool` materialization and label ops).

Validation:
- Red/green:
  - `uv run pytest -q src/dnadesign/opal/tests/test_cli_config_discovery.py::test_init_rejects_unknown_model_plugin` -> FAIL (before), PASS (after).
- Impacted suites:
  - `uv run pytest -q src/dnadesign/opal/tests/test_cli_config_discovery.py src/dnadesign/opal/tests/test_cli_workflows.py src/dnadesign/opal/tests/test_cli_campaign_reset.py src/dnadesign/opal/tests/test_workflow_matrix_cli.py` -> PASS
- Full OPAL suite:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src/cli/commands/_common.py src/dnadesign/opal/src/cli/commands/validate.py src/dnadesign/opal/tests/test_cli_config_discovery.py` -> PASS

### 2026-02-16 strict UQ path hardening (no std-drop fallback + selection signature fail-fast)

Goal:
- Remove remaining fallback-style behavior in uncertainty and selection plugin contracts.

Changes:
- Runtime y-ops inversion now hard-fails whenever model std-dev payload is present and a configured y-op lacks `inverse_std`:
  - `src/dnadesign/opal/src/runtime/round/stages.py`
  - removed path that silently dropped std-dev on non-EI flows.
- Selection registry callable resolution is stricter and rejects malformed selection callables at registry resolution time:
  - `src/dnadesign/opal/src/registries/selection.py`
  - required callable args now include `ids`, `scores`, `top_k`, `objective`, `tie_handling`.
- SFXI diagnostics summary removed broad exception fallback:
  - `src/dnadesign/opal/src/objectives/sfxi_v1.py`
  - clip-fraction summary now computes directly and fails if diagnostics contract is broken.
- Docs aligned to strict y-ops/std contract:
  - `src/dnadesign/opal/docs/plugins/models.md`
  - `src/dnadesign/opal/docs/concepts/strategy-matrix.md`

TDD:
- Added failing tests first:
  - `src/dnadesign/opal/tests/test_run_round_integrity.py::test_run_round_rejects_topn_when_yops_lacks_inverse_std_with_gp`
  - `src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_rejects_incomplete_selection_callable_signature`
- Confirmed both failed before implementation.

Validation:
- Targeted regression tests:
  - `uv run pytest -q src/dnadesign/opal/tests/test_run_round_integrity.py::test_run_round_rejects_topn_when_yops_lacks_inverse_std_with_gp src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_rejects_incomplete_selection_callable_signature` -> PASS
- Full OPAL suite:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS

### 2026-02-16 CLI/run + selection registry fail-fast hardening pass

Goal:
- Remove additional fail-late/fallback behavior in the run guard and selection plugin wiring path.

Changes:
- `run` now fails fast on malformed `state.json` instead of silently ignoring load errors:
  - `src/dnadesign/opal/src/cli/commands/run.py`
  - corrupt state metadata is now surfaced as `OpalError(BAD_ARGS)` before execution.
- Selection registry now validates required callable parameters against runtime-provided args plus `selection.params`:
  - `src/dnadesign/opal/src/registries/selection.py`
  - catches selection callables with required unbound kwargs at `get_selection(...)` time.
  - still allows required plugin args when those keys are provided via `selection.params`.

TDD:
- Added failing tests first:
  - `src/dnadesign/opal/tests/test_cli_workflows.py::test_run_rejects_corrupt_state_json`
  - `src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_rejects_unbound_required_parameters`
  - `src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_allows_required_parameters_bound_from_params`
- Confirmed failures before implementation, then implemented minimal fixes to green.

Validation:
- Targeted:
  - `uv run pytest -q src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_rejects_unbound_required_parameters src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_allows_required_parameters_bound_from_params src/dnadesign/opal/tests/test_cli_workflows.py::test_run_rejects_corrupt_state_json` -> PASS
- Full OPAL suite:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS

### 2026-02-16 selection contract refactor + reserved-key hardening pass

Goal:
- Implement shared selection contract parsing to remove duplicated mode/tie validators and tighten selection plugin parameter validation.

Changes:
- Added shared selection contract parsers:
  - `src/dnadesign/opal/src/core/selection_contracts.py`
  - `resolve_selection_objective_mode(...)`
  - `resolve_selection_tie_handling(...)`
- Rewired callers to shared parser module:
  - `src/dnadesign/opal/src/runtime/round/stages.py`
  - `src/dnadesign/opal/src/cli/commands/run.py`
  - `src/dnadesign/opal/src/analysis/dashboard/selection.py`
- Tightened selection registry required-parameter analysis to ignore runtime-reserved selection keys:
  - `src/dnadesign/opal/src/registries/selection.py`
  - reserved keys (`score_ref`, `uncertainty_ref`, `objective_mode`, `tie_handling`, `top_k`, `exclude_already_labeled`) no longer count as plugin-call kwargs during preflight.

TDD:
- Added failing tests first:
  - `src/dnadesign/opal/tests/test_selection_contracts.py`
  - `src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_rejects_required_param_if_only_reserved_key_is_present`
- Confirmed failures before implementation:
  - missing `core.selection_contracts` module failed test collection.
  - reserved-key case was accepted before strict registry filtering.

Validation:
- Targeted:
  - `uv run pytest -q src/dnadesign/opal/tests/test_selection_contracts.py src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_rejects_required_param_if_only_reserved_key_is_present src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_rejects_unbound_required_parameters src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_allows_required_parameters_bound_from_params src/dnadesign/opal/tests/test_run_round_integrity.py::test_run_round_rejects_missing_objective_mode src/dnadesign/opal/tests/test_run_round_integrity.py::test_run_round_rejects_missing_tie_handling src/dnadesign/opal/tests/test_dashboard_utils.py::test_compute_selection_overlay_requires_explicit_tie_handling src/dnadesign/opal/tests/test_dashboard_utils.py::test_resolve_objective_mode_aliases src/dnadesign/opal/tests/test_cli_workflows.py::test_run_rejects_corrupt_state_json` -> PASS
- Full OPAL suite:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS

### 2026-02-16 config + selection registry strictness pass (no factory heuristics)

Goal:
- Remove one remaining heuristic path in selection registration and enforce unknown-plugin rejection at config load for non-CLI API usage.

Changes:
- Selection registry now requires explicit factory registration mode:
  - `src/dnadesign/opal/src/registries/selection.py`
  - `@register_selection(name, factory=True)` is required for factory-style plugins.
  - Non-factory registrations that look like factories now fail with an explicit error.
- Config loader now validates plugin names against active registries:
  - `src/dnadesign/opal/src/config/loader.py`
  - Validates `transforms_x`, `transforms_y`, `model`, `selection`, each `objective`, and each `training.y_ops` entry.
- CLI config loader now maps `ConfigError` to user-facing `OpalError(BAD_ARGS)`:
  - `src/dnadesign/opal/src/cli/commands/_common.py`
- Config reference docs updated to reflect fail-fast plugin-name checks at config load:
  - `src/dnadesign/opal/docs/reference/configuration.md`

TDD:
- Added failing tests first:
  - `src/dnadesign/opal/tests/test_config_objectives_v2.py::test_load_config_rejects_unknown_model_plugin_name`
  - `src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_does_not_mask_factory_typeerror`
  - `src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_rejects_unmarked_factory`
- Confirmed failures before implementation, then implemented minimal fixes.

Validation:
- Targeted:
  - `uv run pytest -q src/dnadesign/opal/tests/test_config_objectives_v2.py::test_load_config_rejects_unknown_model_plugin_name src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_does_not_mask_factory_typeerror src/dnadesign/opal/tests/test_roundctx_contracts.py::test_get_selection_rejects_unmarked_factory src/dnadesign/opal/tests/test_cli_config_discovery.py::test_init_rejects_unknown_model_plugin` -> PASS
- Broader:
  - `uv run pytest -q src/dnadesign/opal/tests/test_roundctx_contracts.py src/dnadesign/opal/tests/test_config_objectives_v2.py src/dnadesign/opal/tests/test_selection_contracts.py src/dnadesign/opal/tests/test_run_round_integrity.py` -> PASS
- Full OPAL:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS

### 2026-02-16 strict y-op state + run-pred diagnostics hardening pass

Goal:
- Remove remaining fallback-style behavior in y-op state access and row diagnostics flattening; make run-level uncertainty summary reflect emitted uncertainty.

Changes:
- `intensity_median_iqr` no longer uses implicit defaults for fitted state:
  - `src/dnadesign/opal/src/transforms_y/intensity_median_iqr.py`
  - `transform/inverse/inverse_std` now require explicit ctx keys (`enabled`, `center`, `scale`, `eps`) and validate state shapes/finiteness.
  - Missing or invalid fitted state now raises explicit errors instead of silently using placeholder vectors.
- `build_run_pred_events` now rejects row diagnostic length mismatches:
  - `src/dnadesign/opal/src/storage/writebacks.py`
  - Removed fallback that broadcasted `nanmean` when diagnostic vector length did not match candidate count.
- Run meta uncertainty summary now uses emitted selected uncertainty:
  - `src/dnadesign/opal/src/runtime/round/writebacks.py`
  - `stats__unc_mean_sd_targets` is populated from `score.uq_scalar` when available.
- Data-contract docs updated for `stats__unc_mean_sd_targets` semantics:
  - `src/dnadesign/opal/docs/reference/data-contracts.md`

TDD:
- Added failing tests first:
  - `src/dnadesign/opal/tests/test_intensity_median_iqr.py::test_intensity_median_iqr_transform_rejects_missing_fit_state`
  - `src/dnadesign/opal/tests/test_intensity_median_iqr.py::test_intensity_median_iqr_inverse_std_rejects_nonpositive_scale`
  - `src/dnadesign/opal/tests/test_ledger_dataset_writes.py::test_build_run_pred_events_rejects_diagnostic_length_mismatch`
  - `src/dnadesign/opal/tests/test_workflow_matrix_cli.py` EI case now asserts non-null `stats__unc_mean_sd_targets`.

Validation:
- Targeted:
  - `uv run pytest -q src/dnadesign/opal/tests/test_intensity_median_iqr.py::test_intensity_median_iqr_transform_rejects_missing_fit_state src/dnadesign/opal/tests/test_intensity_median_iqr.py::test_intensity_median_iqr_inverse_std_rejects_nonpositive_scale src/dnadesign/opal/tests/test_ledger_dataset_writes.py::test_build_run_pred_events_rejects_diagnostic_length_mismatch 'src/dnadesign/opal/tests/test_workflow_matrix_cli.py::test_cli_workflow_matrix[gp_sfxi_ei-gaussian_process-model_params2-expected_improvement-selection_params2]'` -> PASS
- Broader:
  - `uv run pytest -q src/dnadesign/opal/tests/test_intensity_median_iqr.py src/dnadesign/opal/tests/test_ledger_dataset_writes.py src/dnadesign/opal/tests/test_workflow_matrix_cli.py src/dnadesign/opal/tests/test_run_round_integrity.py` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src/transforms_y/intensity_median_iqr.py src/dnadesign/opal/src/storage/writebacks.py src/dnadesign/opal/src/runtime/round/writebacks.py src/dnadesign/opal/tests/test_intensity_median_iqr.py src/dnadesign/opal/tests/test_ledger_dataset_writes.py src/dnadesign/opal/tests/test_workflow_matrix_cli.py` -> PASS

### 2026-02-16 strict summary-stats logging + selection ctx wrapper cleanup

Goal:
- Close the two remaining suggestions from the prior audit with fail-fast semantics preserved and no fallback behavior added.

Changes:
- Added explicit summary-stat log serializer in scoring stage:
  - `src/dnadesign/opal/src/runtime/round/stages.py`
  - New `_format_summary_stats_for_log(...)` formats numeric values deterministically and avoids broad exception fallback during rendering.
- Removed redundant catch-and-reraise wrappers in selection ctx enforcement:
  - `src/dnadesign/opal/src/registries/selection.py`
  - `precheck_requires` and `postcheck_produces` now raise directly.
  - Kept `reset_stage_state()` behavior on actual plugin execution exceptions.
- Added dedicated regression test for mixed summary-stat value types:
  - `src/dnadesign/opal/tests/test_run_round_integrity.py::test_format_summary_stats_for_log_handles_mixed_types`

TDD:
- Added failing test first for summary-stat formatting helper import/behavior.
- Confirmed red:
  - `uv run pytest -q src/dnadesign/opal/tests/test_run_round_integrity.py::test_format_summary_stats_for_log_handles_mixed_types` -> FAIL (missing helper import)
- Implemented minimal helper + call-site replacement.
- Re-ran targeted test to green.

Validation:
- Targeted:
  - `uv run pytest -q src/dnadesign/opal/tests/test_run_round_integrity.py::test_format_summary_stats_for_log_handles_mixed_types` -> PASS
  - `uv run pytest -q src/dnadesign/opal/tests/test_selection_contracts.py src/dnadesign/opal/tests/test_selection_result_validation.py src/dnadesign/opal/tests/test_selection_exclusion.py src/dnadesign/opal/tests/test_run_round_integrity.py src/dnadesign/opal/tests/test_objective_contract_v2.py src/dnadesign/opal/tests/test_dashboard_utils.py` -> PASS
- Full OPAL:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS

### 2026-02-16 demo pressure-test pass + validate fail-fast hardening

Goal:
- Pressure-test all documented OPAL demo flows for UX and numerical stability, then eliminate fail-late selection/UQ config paths.

Pressure-test execution:
- Ran full OPAL tests and all three demo flows from clean temp campaign copies:
  - RF + SFXI + top_n
  - GP + SFXI + top_n
  - GP + SFXI + expected_improvement
- Reproduced two fail-late config paths:
  - `selection.params.uncertainty_ref` channel typo passed `opal validate` and failed in `opal run`.
  - `selection=expected_improvement` with `model=random_forest` passed `opal validate` and failed in `opal run` due missing predictive uncertainty emission.
- UX observation:
  - `ingest-y` preview rendered unresolved IDs as `id=nan`, which is ambiguous for users following demo docs.

Changes:
- Added validate-time selection/channel contract checks:
  - `src/dnadesign/opal/src/cli/commands/validate.py`
  - `score_ref` and `uncertainty_ref` must be `<objective>/<channel>` and reference configured objectives.
  - For objectives that declare channels, validate now rejects undeclared channel names at validate-time.
  - `selection=expected_improvement` now requires model contract `model/<self>/std_devs` at validate-time.
- Added objective channel declarations for built-ins:
  - `src/dnadesign/opal/src/objectives/sfxi_v1.py`
  - `src/dnadesign/opal/src/objectives/scalar_identity_v1.py`
- Added objective declaration access helper in registry:
  - `src/dnadesign/opal/src/registries/objectives.py`
- Improved ingest preview clarity:
  - `src/dnadesign/opal/src/cli/formatting/renderers/ingest.py`
  - unresolved IDs now render as `<unresolved>` instead of `nan`.

TDD:
- Added failing tests first:
  - `src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_unknown_selection_score_channel_for_declared_objective`
  - `src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_ei_channel_typo_before_runtime`
  - `src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_ei_with_model_without_predictive_std`
  - `src/dnadesign/opal/tests/test_ingest_preview_counts.py::test_ingest_preview_renders_unresolved_id_instead_of_nan`
- Confirmed red, implemented minimal fixes, re-ran to green.

Validation:
- Targeted:
  - `uv run pytest -q src/dnadesign/opal/tests/test_cli_workflows.py src/dnadesign/opal/tests/test_ingest_preview_counts.py` -> PASS
- Full OPAL:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS
- Demo matrix recheck:
  - round-0 `init/validate/ingest-y/run/verify-outputs` for all 3 flows -> PASS
- Edge-case recheck:
  - EI bad `uncertainty_ref` now fails at `opal validate` with declared-channel error.
  - RF + EI now fails at `opal validate` with model uncertainty contract error.

### 2026-02-16 pass-2 pressure test: objective_mode fail-late closure

Goal:
- Re-run full demo matrix and adversarial configs; remove remaining fail-late selection semantic mismatch.

Finding:
- `selection.params.objective_mode` mismatch (for example `score_ref=sfxi_v1/sfxi` + `objective_mode=minimize`) passed `opal validate` and failed during `opal run`.

Changes:
- Added objective-declared score mode metadata for built-ins:
  - `src/dnadesign/opal/src/objectives/sfxi_v1.py`
  - `src/dnadesign/opal/src/objectives/scalar_identity_v1.py`
- Extended objective registry metadata exposure:
  - `src/dnadesign/opal/src/registries/objectives.py`
- Added validate-time objective mode/channel coherence check:
  - `src/dnadesign/opal/src/cli/commands/validate.py`

TDD:
- Added failing test first:
  - `src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_objective_mode_mismatch_for_score_ref`
- Confirmed red before implementation, then green after fix.

Validation:
- Targeted:
  - `uv run pytest -q src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_objective_mode_mismatch_for_score_ref src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_unknown_selection_score_channel_for_declared_objective src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_ei_channel_typo_before_runtime src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_ei_with_model_without_predictive_std` -> PASS
- Full OPAL:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS
- Adversarial recheck:
  - objective mode mismatch now fails at `opal validate` with explicit mode mismatch error.

### 2026-02-16 pass-3 pressure test: EI weight fail-late closure + demo matrix recheck

Goal:
- Re-run the full demo matrix as a real-user command path and probe additional adversarial EI config edges for fail-fast behavior.

Pressure-test execution:
- Re-ran all three campaign-scoped demo flows from clean temp copies with full sequence:
  - `init`, `validate`, `ingest-y`, `run`, `verify-outputs`, `status`, `runs list`, `ctx audit`, `explain`, `record-show`, `predict`, `plot`, then round-1 `ingest-y` + `run --resume` + `verify-outputs`.
- Flow outcomes:
  - RF + SFXI + top_n: round-0/round-1 verify mismatches empty.
  - GP + SFXI + top_n: round-0/round-1 verify mismatches empty.
  - GP + SFXI + expected_improvement: round-0/round-1 verify mismatches empty.
- Numeric checks on ledger predictions:
  - selected scores finite with no NaN/Inf in all three flows.
  - GP+EI selected uncertainty finite and non-negative.
  - GP+top_n records uncertainty channels but keeps selected uncertainty null when no `uncertainty_ref` is configured (expected contract).

Finding:
- Additional fail-late path remained:
  - `selection.params.alpha` / `selection.params.beta` negative values passed `opal validate` but failed at runtime in `expected_improvement` (`alpha must be finite and >= 0`).

Changes:
- Added validate-time strict checks for EI weights in selection schema:
  - `src/dnadesign/opal/src/config/plugin_schemas.py`
  - `alpha` and `beta` now require finite and `>= 0`.

TDD:
- Added failing test first:
  - `src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_ei_negative_weights`
- Confirmed red before code change, then green after schema validator update.

Validation:
- Targeted:
  - `uv run pytest -q src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_ei_negative_weights src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_ei_channel_typo_before_runtime src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_ei_with_model_without_predictive_std src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_rejects_objective_mode_mismatch_for_score_ref` -> PASS
- Full OPAL:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- Lint:
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS
- Adversarial recheck:
  - demo GP+EI config with negative `alpha` now fails at `opal validate` (schema error) instead of run-time.

### 2026-02-16 pass-4 adversarial UX hardening (demo reset path + plugin error clarity)

Goal:
- Pressure-test demo workflows again from user docs, then remove CLI UX footguns found during adversarial runs.

Findings:
- `opal campaign-reset` rejected demo campaign slugs used by current demo flows (`demo_gp_topn`, `demo_gp_ei`, `demo_rf_sfxi_topn`) unless `--allow-non-demo` was passed.
- Unknown plugin validation errors used bracketed lists in the error message, which rendered poorly in default Rich output (available plugin names looked blank).

Changes:
- Expanded demo slug detection for `campaign-reset`:
  - `src/dnadesign/opal/src/cli/commands/campaign_reset.py`
  - Demo slugs now include `demo`, `demo_*`, and `demo-*`.
  - Updated guard error message accordingly.
- Updated unknown plugin message format to avoid Rich-markup ambiguity:
  - `src/dnadesign/opal/src/config/loader.py`
  - Message now uses `Available plugins: ...` (no bracketed list markup).
- Updated demo docs to use a single semantic reset command instead of ad hoc `rm -rf` cleanup:
  - `src/dnadesign/opal/docs/workflows/index.md`
  - `src/dnadesign/opal/docs/workflows/rf-sfxi-topn.md`
  - `src/dnadesign/opal/docs/workflows/gp-sfxi-topn.md`
  - `src/dnadesign/opal/docs/workflows/gp-sfxi-ei.md`
- Updated CLI reference for reset guard semantics:
  - `src/dnadesign/opal/docs/reference/cli.md`

TDD:
- Added failing tests first:
  - `src/dnadesign/opal/tests/test_cli_campaign_reset.py::test_campaign_reset_allows_demo_prefixed_slug_without_flag`
  - `src/dnadesign/opal/tests/test_config_objectives_v2.py::test_load_config_rejects_unknown_model_plugin_name` (tightened assertion for explicit `Available plugins` text)
  - `src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_unknown_model_error_lists_available_plugins_in_default_output`
- Confirmed red, implemented minimal fixes, re-ran to green.

Validation:
- Targeted tests:
  - `uv run pytest -q src/dnadesign/opal/tests/test_cli_campaign_reset.py src/dnadesign/opal/tests/test_cli_workflows.py::test_validate_unknown_model_error_lists_available_plugins_in_default_output src/dnadesign/opal/tests/test_config_objectives_v2.py::test_load_config_rejects_unknown_model_plugin_name` -> PASS
- Demo reset command check for all three demo campaigns:
  - `uv run opal campaign-reset -c <demo_config> --apply --no-backup` -> PASS for `demo_rf_sfxi_topn`, `demo_gp_topn`, `demo_gp_ei`.
- Full demo matrix re-run (reset-first path from docs, round-0 + round-1):
  - RF + SFXI + top_n -> PASS
  - GP + SFXI + top_n -> PASS
  - GP + SFXI + expected_improvement -> PASS
- Full OPAL gate:
  - `uv run pytest -q src/dnadesign/opal/tests` -> PASS
  - `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS

### 2026-02-16 pass-5 campaign-reset policy simplification (remove non-demo gate)

Goal:
- Remove unnecessary campaign-reset scope gating and simplify command semantics.

Changes:
- Removed `--allow-non-demo` from the command surface:
  - `src/dnadesign/opal/src/cli/commands/campaign_reset.py`
- Removed slug-based reset restriction; `campaign-reset` now supports any campaign when explicitly invoked with config + confirmation (`--apply` or slug typing in TTY).
- Kept assertive destructive safeguards:
  - explicit `--config`/`OPAL_CONFIG` resolution,
  - preview output,
  - interactive slug confirmation when `--apply` is not set.
- Updated CLI docs:
  - `src/dnadesign/opal/docs/reference/cli.md`
- Updated tests:
  - `src/dnadesign/opal/tests/test_cli_campaign_reset.py`
    - non-demo reset without extra flags now passes,
    - removed flag path now rejected as unknown option.

Validation:
- `uv run pytest -q src/dnadesign/opal/tests/test_cli_campaign_reset.py` -> PASS
- `uv run pytest -q src/dnadesign/opal/tests` -> PASS
- `uv run ruff check src/dnadesign/opal/src src/dnadesign/opal/tests src/dnadesign/opal/docs` -> PASS
- Manual CLI check:
  - non-demo slug (`alpha_campaign`) reset via `uv run opal campaign-reset -c <config> --apply --no-backup` -> PASS
