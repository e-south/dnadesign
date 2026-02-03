# DenseGen Dev Journal

## 2026-02-03
- Started refactor to reduce DenseGen monoliths (orchestrator/CLI/config).
- Decisions: enforce strict no-fallback behavior across CLI/reporting; split config into submodules; extract pipeline phases; consolidate Stage-A modules; keep public imports stable via re-exports; add background progress + logos (prior commit).
- Extracted pipeline validation/usage helpers and Stage-A pool preparation into dedicated modules.
- Added shared sequence GC utility and corrected Stage-A import layering to eliminate circular imports.
- Extracted library artifact loading/writing into a dedicated pipeline helper with assertive parquet handling.
- Extracted resume-state loading into a dedicated pipeline helper.
- Extracted run-state initialization/writing into a dedicated pipeline helper.
- Split reporting into data and rendering modules; keep public facade stable.
- Added shared record value coercion helpers and removed duplicated list parsing.
- Fixed a run_metrics circular import by deferring plan_pools import to call site.
