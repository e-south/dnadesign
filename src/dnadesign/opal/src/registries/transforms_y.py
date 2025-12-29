"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/transforms_y.py

Registry for:
  1) Y ingest transforms: CSV/Parquet rows -> DataFrame(['sequence'(, 'id'), 'y'])
  2) Y-ops (runtime): (fit, transform, inverse, ParamModel) applied during training & prediction

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

# ---- Ingest transforms ----

_TRANSFORM_Y: Dict[str, Callable[..., Any]] = {}


def register_transform_y(name: str):
    def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in _TRANSFORM_Y:
            raise ValueError(f"Y transform already registered: {name!r}")
        _TRANSFORM_Y[name] = func
        return func

    return _wrap


def get_transform_y(name: str) -> Callable[..., Any]:
    try:
        return _TRANSFORM_Y[name]
    except KeyError as e:
        raise ValueError(f"Unknown Y transform: {name!r}") from e


def list_transforms_y() -> List[str]:
    return sorted(_TRANSFORM_Y.keys())


# ---- Y-ops (runtime) ----

_Y_OPS: Dict[
    str,
    Tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Optional[Type[Any]]],
] = {}


def register_y_op(name: str):
    """
    Register a Y-op. The factory should return (fit_fn, transform_fn, inverse_fn, ParamModel|None)
    where:
      - fit_fn(Y: np.ndarray, params: ParamModel, ctx) -> None
      - transform_fn(Y: np.ndarray, params: ParamModel, ctx) -> np.ndarray
      - inverse_fn(Y: np.ndarray, params: ParamModel, ctx) -> np.ndarray
    """

    def _wrap(
        factory: Callable[[], Tuple[Callable, Callable, Callable, Optional[Type[Any]]]],
    ):
        if name in _Y_OPS:
            raise ValueError(f"Y-op already registered: {name!r}")
        spec = factory()
        if not (isinstance(spec, tuple) and len(spec) == 4):
            raise ValueError("Y-op factory must return (fit, transform, inverse, ParamModel)")
        _Y_OPS[name] = spec
        return factory

    return _wrap


def get_y_op(name: str):
    try:
        return _Y_OPS[name]
    except KeyError as e:
        raise ValueError(f"Unknown Y-op: {name!r}") from e


def list_y_ops() -> List[str]:
    return sorted(_Y_OPS.keys())


# ---- Pipeline runner for Y-ops ----


def run_y_ops_pipeline(
    *,
    stage: str,  # "fit_transform" | "inverse"
    y_ops: List[Any],  # list of config.PluginRef (has .name and .params)
    Y: np.ndarray,
    ctx,  # RoundCtx (or compatible) with .get(key, default), .set(key, val)
) -> np.ndarray:
    """
    Apply Y-ops according to the configured pipeline.

    - stage == "fit_transform":
        for each op:
          1) validate params (ParamModel if available)
          2) fit_fn(Yt, params, ctx)    -> writes state under yops/<name>/*
          3) Yt = transform_fn(Yt, ...) -> transformed target space

      (stores pipeline names/params under yops/pipeline/*)

    - stage == "inverse":
        iterate pipeline in forward order, applying inverse_fn to map predictions
        back to objective space.

    Returns the transformed Y array.
    """
    Yt = np.asarray(Y, dtype=float)

    if stage not in ("fit_transform", "inverse"):
        raise ValueError(f"Invalid stage: {stage}")

    names: List[str] = []
    params_used: List[Dict[str, Any]] = []

    if stage == "fit_transform":
        for entry in y_ops or []:
            fit_fn, xform_fn, inv_fn, ParamT = get_y_op(entry.name)
            params = ParamT(**(entry.params or {})).model_dump() if ParamT is not None else dict(entry.params or {})
            # side-effect: write fitted stats into ctx under yops/<name>/*
            fit_fn(Yt, ParamT(**params) if ParamT is not None else params, ctx=ctx)
            Yt = np.asarray(
                xform_fn(Yt, ParamT(**params) if ParamT is not None else params, ctx=ctx),
                dtype=float,
            )
            names.append(entry.name)
            params_used.append(dict(params))

        ctx.set("yops/pipeline/names", names)
        ctx.set("yops/pipeline/params", params_used)
        return Yt

    # inverse
    names = ctx.get("yops/pipeline/names", default=[])
    params_used = ctx.get("yops/pipeline/params", default=[])
    if len(names) != len(params_used):
        raise ValueError("Malformed Y-ops pipeline: names/params length mismatch.")

    # Invert in reverse order of fit/transform
    for name, params in zip(reversed(names), reversed(params_used)):
        fit_fn, xform_fn, inv_fn, ParamT = get_y_op(name)
        Yt = np.asarray(
            inv_fn(Yt, ParamT(**params) if ParamT is not None else params, ctx=ctx),
            dtype=float,
        )
    return Yt
