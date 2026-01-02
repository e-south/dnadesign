"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/__init__.py

Public registry APIs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

# Models
from .models import get_model, list_models, register_model

# Objectives
from .objectives import get_objective, list_objectives, register_objective

# Plots
from .plots import get_plot, list_plots, register_plot

# Selections
from .selection import get_selection, list_selections, register_selection

# X transforms
from .transforms_x import get_transform_x, list_transforms_x, register_transform_x

# Y ingest transforms + Y-ops (training-time)
from .transforms_y import (
    get_transform_y,
    list_transforms_y,
    list_y_ops,
    register_transform_y,
    register_y_op,
    run_y_ops_pipeline,
)

__all__ = [
    # X
    "register_transform_x",
    "get_transform_x",
    "list_transforms_x",
    # Y (ingest)
    "register_transform_y",
    "get_transform_y",
    "list_transforms_y",
    # Y-ops (training-time)
    "register_y_op",
    "list_y_ops",
    "run_y_ops_pipeline",
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
