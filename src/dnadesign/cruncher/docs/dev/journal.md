# cruncher dev journal


## Contents
- [2026-02-04](#2026-02-04)
- [2026-02-05](#2026-02-05)
- [2026-02-06](#2026-02-06)

## 2026-02-04
- Start refactor to remove core I/O dependencies and progress UI from optimizers.
- Enforce strict no-fallback behaviors (numba cache location, auto-opt override).
- Split sample/analyze workflows into focused modules while keeping public entrypoints.
- Add core-import contract test and update docs to match strict behavior.
- Extracted sample workflow helpers into app/sample/* modules and slimmed app/sample_workflow to orchestration + re-exports.
- Split analysis helpers into app/analyze/* modules and kept app/analyze_workflow as the entrypoint.
- Reviewed cruncher docs (architecture, internals spec, sampling guide, config reference) for no-fallback alignment.
- Flagged explicit fallbacks (warm-start elites fallback, analyze partial outputs) as contradictions to "no fallbacks" goal.
- Proposed follow-ups: make warm-start strict, fail analyze on missing artifacts unless explicitly allowed, and migrate tests to new app submodules.
- Enforced warm-start to require sequences.parquet and removed elite fallbacks; missing seeds now error.
- Analyze now errors if trace.nc is missing when trace-based diagnostics are requested.
- Updated docs to reflect strict analyze artifacts and warm-start requirements.
- Reorganized tests into analysis/app/cli/core/ingest/store subfolders and fixed path-based tests.
- Profiled demo_basics_two_tf with a reduced budget; font-cache build dominated the first run, with scoring/pvalue setup next.
- Added analysis plan + manifest helpers to trim analyze_workflow and keep manifest writing cohesive.
- Made fetch CLI default its source from motif_store.source_preference and documented the requirement.
- Tightened auto-opt candidate validation: warn candidates now require allow_warn, and quality=fail is rejected.
- Added a core import contract test to keep core free of artifacts/cli/filesystem dependencies.
- Profiled `cruncher analyze --summary` (demo_basics_two_tf); runtime dominated by import overhead (~3s total).
- Audited demo_basics_two_tf outputs: auto_opt candidates=40 (20 specs x 2 budgets), 35 fail (ESS<10), 5 warn; winner selected via allow_warn. auto_opt outputs ~145M vs sample ~7.7M.
- Added learning metrics to analysis objective components + report (best score draw, plateau, early-stop simulation).
- Enforced early_stop.min_delta <= 0.1 for normalized-llr configs and added tests.
- Enabled early_stop and analysis.extra_tables in demo config; demo now emits auto_opt_pilots table.
- Updated docs (config, sampling guide, demo) to reflect learning metrics + early-stop guidance.
- Auto-opt quality grading now uses trace draw counts + ESS ratio; short pilots are marked warn and ESS ratio only warns (no hard-fail).
- Auto-opt candidate payload now records ess_ratio + trace_draws/expected to make pilot diagnostics scale-aware in auto_opt_pilots tables.
- Reran demo_basics_two_tf with full default config; auto-opt selected PT with warnings and produced sample run lexA-cpxR_20260204_140346_cb1935.
- Latest demo analysis warns only on low elite count (n_elites=1) and shows early-stop per-chain around draw 1500-1633.
- Auto-opt pilot summary: PT swap_prob=0.15 + aggressive/boosted cooling yields top top_k_median; ESS ratios remain very low (~0.001) across grid.
- Added `min_norm` to sequences.parquet for fast per-draw consensus filtering.
- Lowered demo `min_per_tf_norm` / `success_min_per_tf_norm` to 0.60 after inspecting draw percentiles (median ~0.28, 95th ~0.55) and unique counts.
- Updated CLI docs to describe the MMR scorecard as the default auto-opt selector.
- Allow NUMBA_CACHE_DIR overrides when an explicit cache_dir is provided (warn + overwrite) to keep CLI runs deterministic in workspaces.
- Reordered MMR selection to run after final candidate pool assembly so final elites preserve diversity (MMR meta now matches elites).
- Reran demo_basics_two_tf after the MMR ordering change: latest run lexA-cpxR_20260204_174159_fdeee2 produced 10 elites with 10 unique canonical sequences and unique_successes=24.
- Design Section 4 plan of action: (1) audit + prune config schema to essential knobs (enforce init.length >= max PWM width; canonicalize MMR when bidirectional), (2) behavior cleanup (PT-only, no auto-disable paths), (3) refactor cohesion (split auto_opt and run_set by responsibility), (4) tests (CLI smoke test, fixed-length vs max PWM), (5) docs alignment (sampling/config docs; MMR TFBS-core behavior), (6) performance sanity (note scorer DP caching; no core kernel changes now).

## 2026-02-05
- Brainstorming: Intent + Plan of Action (200-300 words). Cruncher's intent is to generate short, fixed-length DNA sequences that jointly satisfy multiple PWMs while returning a diverse elite set, with diversity defined across sequences rather than within a sequence's motif windows. The MCMC kernel should remain unchanged; selection, scoring, and stopping should be assertive and predictable. The hard-break simplification focuses on keeping only essential knobs so users can reason about outcomes with minimal configuration. Plan of action: first, prune config schema to essential knobs and enforce sample.init.length >= max PWM width with canonical MMR under bidirectional scoring. Second, behavior cleanup removes auto-disable or warm-start paths and keeps the optimizer PT-only. Third, improve cohesion by splitting app/sample/auto_opt.py into candidate generation, scoring/selection, and orchestration, and splitting app/sample/run_set.py into run layout, candidate pool creation, MMR selection/metadata, and manifest writing. Fourth, add tests that assert fixed-length constraints and include a CLI smoke test using the two-TF demo. Fifth, align docs (sampling + config) to state fixed length, TFBS-core MMR behavior, and canonicalization under bidirectional scoring. Sixth, add lightweight profiling to identify hot paths, with an eye toward reusing scorer caches across pilots without touching the kernel.
- Fixed a recursion bug in `_assert_init_length_fits_pwms` and centralized the max-PWM length check.
- Removed warm-start seed injection in PT; chains now always start from a shared seed plus small perturbations.
- Dropped `dsdna_hamming` from analysis metadata; diversity metrics now follow `dsdna_canonicalize`.
- Removed unused MMR sequence-distance export and updated tests to align with the slimmer API.
- Reworked the sequence-normalization test to assert the helper is absent.
- Added a minimal CLI smoke test that runs a two-TF matrix sample with tiny budgets.
- Tightened analysis fixtures to include explicit MMR selection policy and fixed demo config indentation.
- Added `min_per_tf_norm` to sequences.parquet and asserted it in the CLI smoke test.
- Gated early-stop reporting on unique-success requirements and surfaced a diagnostics warning when unique_successes < min_unique.
- Lowered demo `min_per_tf_norm` / `success_min_per_tf_norm` to 0.40 and reran demo_basics_two_tf; elites now populate (n=10) and early-stop triggers cleanly.
- Ran the demo end-to-end (fetch motifs/sites -> lock -> parse -> sample -> analyze) using the workspace config.
- Profiled `cruncher sample` on demo_basics_two_tf; dominant cost is PWM log-odds -> p-value lookup construction (`core/pvalue.py`), with scorer init next (candidate for cross-pilot cache reuse).
- Updated docs (demo, sampling guide, config reference, architecture) to reflect PT-only language, TFBS-core MMR behavior, and sequences column naming.
- Removed sample.budget.restarts from schema/config/code to eliminate PT-only fallback overrides; updated tests, docs, and demo configs accordingly.
- Added an in-memory LRU cache for PWM log-odds -> p-value lookup tables (core/pvalue.py) to reuse DP results across auto-opt pilots; added a minimal cache-hit test.
- Added audit design doc for Cruncher end-to-end review (docs/plans/2026-02-05-cruncher-audit-design.md), emphasizing PT-only, fixed-length invariants, MMR canonical selection, and doc/workspace alignment over test bloat.
- Recorded p-value cache hit/miss stats in run manifests and surfaced them in diagnostics metrics for transparency.
- Hardened bidirectional logp correction (counts both strands) and codified deterministic best-hit tie-breaking.
- Added atomic artifact writes (status/manifest/config, parquet, analysis summaries) and retry-on-read for run status with clear CLI errors.
- Persisted effective PT ladder details in elites metadata and added tests for p-seq math, tie-breaking, atomic writes, and PT stats.
- Added Contents TOCs across docs and aligned demo/reference text with current fixed-length PT behavior.
- Ran cruncher tests and the two-TF demo flow (fetch -> lock -> parse -> sample -> analyze); removed an ArviZ warning by increasing draws in the regulator set test and clarified dashboard-only plot outputs in the sampling guide.
- Removed auto-opt scorecard `k` as a config knob; auto-opt now derives the scorecard size from `elites.k` and fails fast if `elites.k < 1` when optimizer is auto.
- Raised default auto-opt pilot budgets to 2000/3000 and aligned the multi-TF demo workspace to match.
- Added auto-opt confidence highlights to analysis reports and CLI summaries so pilots are easier to interpret at a glance.
- Updated sampling/CLI/demo docs to state `elites.k` drives the scorecard size and to guide pilots when elites fall short.
- Added a v3 single-path schema design: `sequence_length`, `compute.total_sweeps`, and `compute.adapt_sweep_frac` with explicit numeric elite gates (`min_per_tf_norm`, `mmr_alpha`), removing pilot grids and inference-style tune/draws.
- Removed auto-opt orchestration modules and CLI/report wiring; updated demos/workspaces and sampling docs to the fixed-length compute schema with no pilot references.

## 2026-02-06
- Continued v3 work: updated schema_v3 to be self-contained (removed schema_v2 imports) and aligned CLI config/analyze commands with the v3 surface.
- Simplified analysis CLI to v3 behavior (run selection + summary only) and removed plot/tf-pair override flags.
- Updated v3 docs (config reference, sampling/analysis guide, demos, ingestion/meme suite guides, architecture, internals spec) to match the curated analysis suite and v3 schema terminology.
- Added a local encode helper in analysis/per_pwm to remove dependency on deprecated scatter utils.
- Added baseline hits artifacts for analysis-only plots and updated sampling to emit `random_baseline_hits.parquet`.
- Replaced the run dashboard + diagnostics plots with a curated v3 plot suite: run summary + trajectory + NN distance + overlap panel + health panel.
- Updated analysis workflow and plot registry to use baseline NN references, new plot names, and health-only diagnostics.
- Simplified overlap plotting to a single panel figure with best-hit labels and readability rules.
- Updated analysis tests and docs to match the new plot suite and baseline artifacts.
- Added an intent + lifecycle guide and linked it from docs index, demo, config, and architecture references.
