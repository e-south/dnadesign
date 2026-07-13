"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/runtime/round/stages/selection_types.py

Typed results for selection views and logical selection batches.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectionEvaluation:
    selection_view_id: str
    y_obj_scalar: np.ndarray
    diag: Dict[str, Any]
    obj_summary_stats: Optional[Dict[str, Any]]
    obj_name: str
    obj_params: Dict[str, Any]
    obj_mode: str
    score_ref: str
    uncertainty_ref: Optional[str]
    sel_name: str
    sel_params: Dict[str, Any]
    tie_handling: str
    mode: str
    ranks_competition: np.ndarray
    selected_bool: np.ndarray
    selected_effective: int
    top_k: int
    obj_sha: str
    scores: np.ndarray
    uq_scalar: Optional[np.ndarray]


@dataclass(frozen=True)
class SelectionBatchEvaluation:
    rows: pd.DataFrame
    deduplicate_by: str
    unique_count: int
    expected_unique_count: Optional[int]


__all__ = ["SelectionBatchEvaluation", "SelectionEvaluation"]
