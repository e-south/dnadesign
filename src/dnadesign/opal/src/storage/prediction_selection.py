"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/prediction_selection.py

Validates whether a prediction row was selected by any declared view.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from ..core.utils import ExitCodes, OpalError


def selected_by_any_view(payload: object) -> bool:
    """Return the strict logical union of selection-view memberships."""

    if isinstance(payload, np.ndarray):
        payload = payload.tolist()
    if not isinstance(payload, (list, tuple)):
        raise OpalError(
            "pred__selection_views must be a sequence for artifact retention.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    selected = False
    for item in payload:
        if not isinstance(item, dict) or "selection_view_id" not in item or "is_selected" not in item:
            raise OpalError(
                "Each pred__selection_views entry requires selection_view_id and is_selected.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        value = item["is_selected"]
        if not isinstance(value, (bool, np.bool_)):
            raise OpalError(
                "pred__selection_views is_selected values must be boolean.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        selected = selected or bool(value)
    return selected


__all__ = ["selected_by_any_view"]
