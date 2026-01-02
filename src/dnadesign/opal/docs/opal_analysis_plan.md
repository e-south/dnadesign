# OPAL Analysis + Plot UX Plan

Summary: centralize campaign/ledger resolution, make plot UX less config-bound,
and add a campaign-tied marimo notebook workflow for richer analysis.

Plan:
1) Extract shared config/workspace resolution and RecordsStore creation.
2) Add an analysis facade with ledger loaders + round/run helpers (polars-first).
3) Refactor `opal plot` to reuse the facade, add `--quick/--describe`, and improve errors.
4) Add `opal notebook generate` (and optional `run`) with a marimo template.
5) Update CLI docs + add smoke tests for notebook generation.
