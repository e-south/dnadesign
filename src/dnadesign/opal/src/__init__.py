"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/__init__.py

Package exports for OPAL.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""


# Plugin modules are loaded lazily by registries to avoid import-time side effects.

__version__ = "0.1.0"
LEDGER_SCHEMA_VERSION = "2.0"

# Intentionally omit __all__ to avoid exporting lazy submodules implicitly.
