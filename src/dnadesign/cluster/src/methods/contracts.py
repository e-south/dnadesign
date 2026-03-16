"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/methods/contracts.py

Clustering-method contracts.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

FitParamsResolver = Callable[[Mapping[str, Any] | None, Mapping[str, Any] | None], dict[str, Any]]
FitRunner = Callable[..., np.ndarray]
SlugParamsBuilder = Callable[[Mapping[str, Any]], dict[str, Any]]
ResolutionSweepRunner = Callable[[np.ndarray, Mapping[str, Any], float, float, float, Sequence[int], Path], Any]


def parse_method_param_assignments(assignments: Sequence[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for assignment in assignments:
        text = assignment.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Invalid --method-param '{assignment}'. Expected key=value.")
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid --method-param '{assignment}'. Parameter name cannot be empty.")
        params[key] = value
    return params


@dataclass(frozen=True)
class ClusteringMethod:
    method_id: str
    display_name: str
    default_run_prefix: str
    fit_param_names: frozenset[str]
    resolve_fit_params: FitParamsResolver
    fit: FitRunner
    slug_params: SlugParamsBuilder
    resolution_sweep: ResolutionSweepRunner | None = None
