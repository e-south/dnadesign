"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/__init__.py

OPAL — Optimization with Active Learning.

Round-based active learning on biological sequences:
- reads a records table (USR or local Parquet),
- trains a top-layer model on explicit X (representation) and Y (label),
- scores the candidate universe, ranks by a scalar selection score, selects top-k,
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
    "ingest",
    "registries",
    "objectives",
]

# Trigger plugin auto-registration when package is imported programmatically
from . import (
    objectives,  # noqa: F401
    selection,  # noqa: F401
    transforms_x,  # noqa: F401
    transforms_y,  # noqa: F401
)
