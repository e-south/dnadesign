"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/methods/registry.py

Registry of supported clustering methods.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import ClusteringMethod
from .kmeans import KMEANS_FIT_PARAM_NAMES
from .kmeans import resolve_fit_params as resolve_kmeans_fit_params
from .kmeans import run as run_kmeans
from .kmeans import slug_params as kmeans_slug_params
from .leiden import LEIDEN_FIT_PARAM_NAMES
from .leiden import resolve_fit_params as resolve_leiden_fit_params
from .leiden import run as run_leiden
from .leiden import slug_params as leiden_slug_params

BuiltinMethodFactory = Callable[[], ClusteringMethod]


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


def _builtin_leiden_method() -> ClusteringMethod:
    return ClusteringMethod(
        method_id="leiden",
        display_name="Leiden community clustering",
        default_run_prefix="leiden",
        fit_param_names=LEIDEN_FIT_PARAM_NAMES,
        resolve_fit_params=resolve_leiden_fit_params,
        fit=run_leiden,
        slug_params=leiden_slug_params,
        operations={"resolution_sweep": _run_leiden_resolution_sweep},
    )


def _builtin_kmeans_method() -> ClusteringMethod:
    return ClusteringMethod(
        method_id="kmeans",
        display_name="MiniBatch K-Means clustering",
        default_run_prefix="kmeans",
        fit_param_names=KMEANS_FIT_PARAM_NAMES,
        resolve_fit_params=resolve_kmeans_fit_params,
        fit=run_kmeans,
        slug_params=kmeans_slug_params,
        operations={},
    )


class MethodRegistry:
    def __init__(self, *, builtin_factories: Sequence[BuiltinMethodFactory] = ()) -> None:
        self._builtin_factories = tuple(builtin_factories)
        self._methods: dict[str, ClusteringMethod] = {}

    def ensure_builtin_methods(self) -> None:
        for factory in self._builtin_factories:
            method = factory()
            self._methods.setdefault(method.method_id, method)

    def register_method(self, method: ClusteringMethod, *, replace: bool = False) -> None:
        method_id = str(method.method_id).strip()
        if not method_id:
            raise ValueError("Cluster method id must be a non-empty string.")
        self.ensure_builtin_methods()
        if method_id in self._methods and not replace:
            raise ValueError(f"Clustering method '{method_id}' is already registered.")
        self._methods[method_id] = method

    def registered_methods(self) -> dict[str, ClusteringMethod]:
        self.ensure_builtin_methods()
        return dict(self._methods)

    def supported_method_ids(self) -> list[str]:
        self.ensure_builtin_methods()
        return sorted(self._methods)

    def get_method(self, method_id: str) -> ClusteringMethod:
        self.ensure_builtin_methods()
        method = self._methods.get(method_id)
        if method is None:
            supported = ", ".join(self.supported_method_ids())
            raise ValueError(f"Unsupported clustering method '{method_id}'. Supported methods: {supported}.")
        return method


_DEFAULT_REGISTRY = MethodRegistry(builtin_factories=(_builtin_leiden_method, _builtin_kmeans_method))


def default_method_registry() -> MethodRegistry:
    return _DEFAULT_REGISTRY


def register_method(method: ClusteringMethod, *, replace: bool = False) -> None:
    default_method_registry().register_method(method, replace=replace)


def registered_methods() -> dict[str, ClusteringMethod]:
    return default_method_registry().registered_methods()


def supported_method_ids() -> list[str]:
    return default_method_registry().supported_method_ids()


def get_method(method_id: str) -> ClusteringMethod:
    return default_method_registry().get_method(method_id)


__all__ = [
    "MethodRegistry",
    "default_method_registry",
    "get_method",
    "register_method",
    "registered_methods",
    "supported_method_ids",
]
