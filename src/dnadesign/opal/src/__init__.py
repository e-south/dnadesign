"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/__init__.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# Plugin modules are loaded lazily by registries to avoid import-time side effects.

__version__ = "0.1.0"
LEDGER_SCHEMA_VERSION = "1.1"

# Intentionally omit __all__ to avoid exporting lazy submodules implicitly.
