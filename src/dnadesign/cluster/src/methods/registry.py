"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/methods/registry.py

Registry of supported clustering methods.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ClusteringMethod
from .leiden import LEIDEN_FIT_PARAM_NAMES
from .leiden import resolve_fit_params as resolve_leiden_fit_params
from .leiden import run as run_leiden
from .leiden import slug_params as leiden_slug_params


def _run_leiden_resolution_sweep(
    X: np.ndarray,
    method_params: Mapping[str, Any],
    res_min: float,
    res_max: float,
    step: float,
    seeds: Sequence[int],
    out_dir: Path,
):
    from .leiden_resolution_sweep import run_resolution_sweep

    return run_resolution_sweep(
        X,
        method_params=method_params,
        res_min=res_min,
        res_max=res_max,
        step=step,
        seeds=seeds,
        out_dir=out_dir,
    )


METHODS: dict[str, ClusteringMethod] = {
    "leiden": ClusteringMethod(
        method_id="leiden",
        display_name="Leiden community clustering",
        default_run_prefix="leiden",
        fit_param_names=LEIDEN_FIT_PARAM_NAMES,
        resolve_fit_params=resolve_leiden_fit_params,
        fit=run_leiden,
        slug_params=leiden_slug_params,
        resolution_sweep=_run_leiden_resolution_sweep,
    )
}


def supported_method_ids() -> list[str]:
    return sorted(METHODS)


def get_method(method_id: str) -> ClusteringMethod:
    method = METHODS.get(method_id)
    if method is None:
        supported = ", ".join(supported_method_ids())
        raise ValueError(f"Unsupported clustering method '{method_id}'. Supported methods: {supported}.")
    return method
