"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/core/objective_result.py

Defines typed objective result contracts and validation helpers for OPAL.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from .utils import OpalError


@dataclass(frozen=True)
class ObjectiveResultV2:
    scores_by_name: Dict[str, np.ndarray]
    uncertainty_by_name: Dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    modes_by_name: Dict[str, str] = field(default_factory=dict)


def validate_objective_result_v2(
    *,
    result: ObjectiveResultV2,
    objective_name: str,
    n_rows: int,
) -> ObjectiveResultV2:
    if not isinstance(result, ObjectiveResultV2):
        raise OpalError(f"Objective '{objective_name}' must return ObjectiveResultV2, got {type(result).__name__}.")

    if not result.scores_by_name:
        raise OpalError(f"Objective '{objective_name}' returned no score channels.")

    normalized_scores: Dict[str, np.ndarray] = {}
    for channel, values in result.scores_by_name.items():
        if not str(channel).strip():
            raise OpalError(f"Objective '{objective_name}' returned an empty score channel name.")
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != n_rows:
            raise OpalError(
                f"Objective '{objective_name}' score channel '{channel}' has length {arr.size}, expected {n_rows}."
            )
        if not np.all(np.isfinite(arr)):
            raise OpalError(f"Objective '{objective_name}' score channel '{channel}' contains non-finite values.")
        normalized_scores[str(channel)] = arr

    normalized_uncertainty: Dict[str, np.ndarray] = {}
    for channel, values in (result.uncertainty_by_name or {}).items():
        if not str(channel).strip():
            raise OpalError(f"Objective '{objective_name}' returned an empty uncertainty channel name.")
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != n_rows:
            raise OpalError(
                (
                    f"Objective '{objective_name}' uncertainty channel '{channel}' has length {arr.size}, "
                    f"expected {n_rows}."
                )
            )
        if not np.all(np.isfinite(arr)):
            raise OpalError(f"Objective '{objective_name}' uncertainty channel '{channel}' contains non-finite values.")
        if np.any(arr < 0.0):
            raise OpalError(f"Objective '{objective_name}' uncertainty channel '{channel}' contains negative values.")
        normalized_uncertainty[str(channel)] = arr

    modes = dict(result.modes_by_name or {})
    normalized_modes: Dict[str, str] = {}
    for channel in normalized_scores:
        if channel not in modes:
            raise OpalError(
                (
                    f"Objective '{objective_name}' score channel '{channel}' is missing a modes_by_name entry; "
                    "each score channel must declare maximize|minimize explicitly."
                )
            )
        mode = str(modes.get(channel)).strip().lower()
        if mode not in {"maximize", "minimize"}:
            raise OpalError(
                (
                    f"Objective '{objective_name}' channel '{channel}' has invalid mode {mode!r}; "
                    "expected maximize|minimize."
                )
            )
        normalized_modes[channel] = mode

    return ObjectiveResultV2(
        scores_by_name=normalized_scores,
        uncertainty_by_name=normalized_uncertainty,
        diagnostics=dict(result.diagnostics or {}),
        modes_by_name=normalized_modes,
    )
