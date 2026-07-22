"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/__init__.py

Internal runtime package for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .cli import main
from .version import __version__

__all__ = ["__version__", "main"]
