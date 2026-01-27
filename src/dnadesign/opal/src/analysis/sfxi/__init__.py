"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/sfxi/__init__.py

SFXI diagnostics helpers for analysis workflows (non-UI, non-plotting).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .state_order import STATE_ORDER, assert_state_order, require_state_order

__all__ = ["STATE_ORDER", "assert_state_order", "require_state_order"]
