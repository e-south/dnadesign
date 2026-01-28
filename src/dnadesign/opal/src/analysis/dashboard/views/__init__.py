"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/views/__init__.py

Exposes dashboard view-layer utilities for data preparation. Provides SFXI and
plot view builders for notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from . import derived_metrics, plots, sfxi

__all__ = ["derived_metrics", "plots", "sfxi"]
