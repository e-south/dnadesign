"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/__init__.py

Eco1 panel-selection materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import MaterializedSelectionReadiness
from .review_axis_contracts import NA_FACING_CHEMISTRY_METRICS

if TYPE_CHECKING:
    from .pipeline import materialize_selection_readiness as materialize_selection_readiness

__all__ = [
    "MaterializedSelectionReadiness",
    "NA_FACING_CHEMISTRY_METRICS",
    "materialize_selection_readiness",
]


def __getattr__(name: str) -> Any:
    if name == "materialize_selection_readiness":
        from .pipeline import materialize_selection_readiness

        return materialize_selection_readiness
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
