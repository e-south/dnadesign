# cruncher dev journal

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
