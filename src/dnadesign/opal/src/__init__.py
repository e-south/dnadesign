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

# Plugin modules are loaded lazily by registries to avoid import-time side effects.

__version__ = "0.1.0"
LEDGER_SCHEMA_VERSION = "1.1"


# Intentionally omit __all__ to avoid exporting lazy submodules implicitly.
