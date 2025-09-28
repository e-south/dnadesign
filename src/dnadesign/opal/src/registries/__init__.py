"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/__init__.py

This module re-exports the public registry APIs from submodules.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from .models import get_model, list_models, register_model
from .objectives import get_objective, list_objectives, register_objective
from .plot import get_plot, list_plots, register_plot
from .selections import get_selection, list_selections, register_selection
from .transforms_x import get_transform_x, list_transforms_x, register_transform_x
from .transforms_y import get_transform_y, list_transforms_y, register_transform_y

__all__ = [
    # X
    "register_transform_x",
    "get_transform_x",
    "list_transforms_x",
    # Y
    "register_transform_y",
    "get_transform_y",
    "list_transforms_y",
    # Models
    "register_model",
    "get_model",
    "list_models",
    # Objectives
    "register_objective",
    "get_objective",
    "list_objectives",
    # Selections
    "register_selection",
    "get_selection",
    "list_selections",
    # Plots
    "register_plot",
    "get_plot",
    "list_plots",
]
