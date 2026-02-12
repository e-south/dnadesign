"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/runtime.py

Explicit runtime bootstrap for baserender contracts and effect drawers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .core import register_builtin_contracts


def initialize_runtime() -> None:
    from .render.effects import register_builtin_effect_drawers

    register_builtin_contracts()
    register_builtin_effect_drawers()
