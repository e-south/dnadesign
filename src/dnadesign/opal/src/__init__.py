"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/__init__.py

OPAL — Optimization with Active Learning.

This package hosts OPAL's core modules and the opal CLI entrypoint.
OPAL orchestrates round-based active learning on biological sequences:

- reads a records table (USR or local Parquet),
- trains a top-layer model on an explicit X (representation) and Y (label),
- scores the candidate universe, ranks by Ŷ, and selects top-k,
- writes per-round predictions/ranks/flags back to the source table,
- persists round artifacts and an append-only campaign state.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

__all__ = [
    "cli",
    "config",
    "data_access",
    "preflight",
    "models",
    "transforms",
    "selection",
    "ranking",
    "artifacts",
    "writebacks",
    "state",
    "status",
    "explain",
    "predict",
    "record_show",
    "logging_utils",
    "locks",
    "utils",
]
