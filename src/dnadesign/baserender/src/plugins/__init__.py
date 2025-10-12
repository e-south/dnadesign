"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/plugins/__init__.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .registry import DerivedAnnotationPlugin, PalettePlugin, load_plugins

__all__ = ["load_plugins", "DerivedAnnotationPlugin", "PalettePlugin"]
