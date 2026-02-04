"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/config/__init__.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .loader import load_config  # noqa: F401
from .types import LocationLocal, LocationUSR, RootConfig  # noqa: F401

# Re-export common config types for convenient imports in the CLI layer.
__all__ = ["load_config", "RootConfig", "LocationLocal", "LocationUSR"]
