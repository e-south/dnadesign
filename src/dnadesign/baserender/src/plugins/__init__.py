"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/plugins/__init__.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .registry import DerivedAnnotationPlugin, load_plugins

__all__ = ["load_plugins", "DerivedAnnotationPlugin"]
