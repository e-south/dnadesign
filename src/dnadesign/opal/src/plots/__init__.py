"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/__init__.py

Importing this package triggers registration side-effects for built-in plots.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

# Register built-in plot plugins
from . import (  # noqa: F401
    feature_importance_bars,
    fold_change_vs_logic_fidelity,
    percent_high_activity_over_rounds,
    scatter_score_vs_rank,
    sfxi_logic_fidelity_closeness,
)

__all__ = []
