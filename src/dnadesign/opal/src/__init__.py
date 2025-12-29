"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/__init__.py

OPAL — Optimization with Active Learning.

Round-based active learning on biological sequences:
- reads a records table (USR or local Parquet),
- trains a top-layer model on explicit X (representation) and Y (label),
- scores the candidate universe, ranks by a scalar selection score, selects top-k,
- appends canonical events to flat **ledger** sinks under outputs/,
- persists minimal round artifacts and an append-only campaign state.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

# Trigger plugin auto-registration when package is imported programmatically
from . import (
    models,
    objectives,  # noqa: F401
    plots,  # noqa: F401
    selection,  # noqa: F401
    transforms_x,  # noqa: F401
    transforms_y,  # noqa: F401
)

__version__ = "0.1.0"
LEDGER_SCHEMA_VERSION = "1.1"


__all__ = [
    "cli",
    "config",
    "data_access",
    "preflight",
    "models",
    "transforms_x",
    "transforms_y",
    "selection",
    "artifacts",
    "ledger",
    "workspace",
    "round_plan",
    "state",
    "status",
    "explain",
    "predict",
    "record_show",
    "locks",
    "utils",
    "ingest",
    "registries",
    "objectives",
]
