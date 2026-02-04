# cruncher dev journal

## 2026-02-04
- Start refactor to remove core I/O dependencies and progress UI from optimizers.
- Enforce strict no-fallback behaviors (numba cache location, auto-opt override).
- Split sample/analyze workflows into focused modules while keeping public entrypoints.
- Add core-import contract test and update docs to match strict behavior.
- Extracted sample workflow helpers into app/sample/* modules and slimmed app/sample_workflow to orchestration + re-exports.
- Split analysis helpers into app/analyze/* modules and kept app/analyze_workflow as the entrypoint.
