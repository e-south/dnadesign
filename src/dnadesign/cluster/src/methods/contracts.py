"""
--------------------------------------------------------------------------------
dnadesign
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
MethodOperationRunner = Callable[..., Any]


@dataclass(frozen=True)
class ClusteringMethod:
    method_id: str
    display_name: str
    default_run_prefix: str
    fit_param_names: frozenset[str]
    resolve_fit_params: FitParamsResolver
    fit: FitRunner
    slug_params: SlugParamsBuilder
    operations: dict[str, MethodOperationRunner] | None = None

    def get_operation(self, operation_id: str) -> MethodOperationRunner | None:
        if not self.operations:
            return None
        return self.operations.get(operation_id)
