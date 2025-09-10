"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/__init__.py

Re-export convenience decorators/getters for plugin registries.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from .models import get_model, list_models, register_model
from .objectives import get_objective, list_objectives, register_objective
from .selections import get_selection, list_selections, register_selection
from .transforms_x import get_rep_transform, register_rep_transform
from .transforms_y import get_ingest_transform, register_ingest_transform

__all__ = [
    # X
    "get_rep_transform",
    "register_rep_transform",
    # Y
    "get_ingest_transform",
    "register_ingest_transform",
    # Models
    "get_model",
    "list_models",
    "register_model",
    # Objectives
    "get_objective",
    "list_objectives",
    "register_objective",
    # Selections
    "get_selection",
    "list_selections",
    "register_selection",
]
