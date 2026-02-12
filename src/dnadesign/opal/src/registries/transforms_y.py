"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/transforms_y.py

Registers Y transforms and loads built-in transform modules. Provides transform
factories with schema and PluginCtx handling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from ..core.round_context import Contract, PluginCtx, RoundCtx
from .loader import load_builtin_modules

# ---- Ingest transforms ----

_TRANSFORM_Y: Dict[str, Callable[..., Any]] = {}
_BUILTINS_LOADED = False


def _dbg(msg: str) -> None:
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on"):
        print(f"[opal.debug.transforms_y] {msg}", file=sys.stderr)


def _ensure_builtins_loaded() -> None:
    """Import package-shipped transforms_y modules that self-register via registry decorators."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    load_builtin_modules("dnadesign.opal.src.transforms_y", label="transform_y", debug=_dbg)
    _BUILTINS_LOADED = True


def register_transform_y(name: str):
    def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in _TRANSFORM_Y:
            raise ValueError(f"Y transform already registered: {name!r}")
        _TRANSFORM_Y[name] = func
        return func

    return _wrap


def _wrap_for_ctx_enforcement(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Enforce PluginCtx contract pre/post checks if ctx is provided.
    """
    contract = getattr(fn, "__opal_contract__", None)

    def _wrapped(df_tidy, params, *, ctx: PluginCtx | None = None):
        if contract is not None and ctx is None:
            raise ValueError(f"transform_y[{name}] requires ctx for contract enforcement.")
        if ctx is not None:
            ctx.precheck_requires(stage="transform_y")
        try:
            out = fn(df_tidy, params, ctx=ctx)
        except Exception:
            if ctx is not None:
                ctx.reset_stage_state()
            raise
        if ctx is not None:
            ctx.postcheck_produces(stage="transform_y")
        return out

    if contract is not None:
        setattr(_wrapped, "__opal_contract__", contract)
    return _wrapped


def get_transform_y(name: str) -> Callable[..., Any]:
    _ensure_builtins_loaded()
    try:
        fn = _TRANSFORM_Y[name]
    except KeyError as e:
        raise ValueError(f"Unknown Y transform: {name!r}") from e
    return _wrap_for_ctx_enforcement(name, fn)


def list_transforms_y() -> List[str]:
    _ensure_builtins_loaded()
    return sorted(_TRANSFORM_Y.keys())


# ---- Y-ops (runtime) ----


@dataclass(frozen=True)
class YOpSpec:
    fit_fn: Callable[..., Any]
    transform_fn: Callable[..., Any]
    inverse_fn: Callable[..., Any]
    ParamModel: Optional[Type[Any]]
    requires: Tuple[str, ...]
    produces: Tuple[str, ...]


_Y_OPS: Dict[str, YOpSpec] = {}


def register_y_op(
    name: str,
    *,
    requires: Optional[List[str]] = None,
    produces: Optional[List[str]] = None,
):
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
        fit_fn, transform_fn, inverse_fn, ParamT = spec
        _Y_OPS[name] = YOpSpec(
            fit_fn=fit_fn,
            transform_fn=transform_fn,
            inverse_fn=inverse_fn,
            ParamModel=ParamT,
            requires=tuple(requires or ()),
            produces=tuple(produces or ()),
        )
        return factory

    return _wrap


def get_y_op(name: str):
    _ensure_builtins_loaded()
    try:
        return _Y_OPS[name]
    except KeyError as e:
        raise ValueError(f"Unknown Y-op: {name!r}") from e


def list_y_ops() -> List[str]:
    _ensure_builtins_loaded()
    return sorted(_Y_OPS.keys())


# ---- Pipeline runner for Y-ops ----


def run_y_ops_pipeline(
    *,
    stage: str,  # "fit_transform" | "inverse"
    y_ops: List[Any],  # list of config.PluginRef (has .name and .params)
    Y: np.ndarray,
    ctx: RoundCtx,
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
    _ensure_builtins_loaded()
    Yt = np.asarray(Y, dtype=float)

    if stage not in ("fit_transform", "inverse"):
        raise ValueError(f"Invalid stage: {stage}")

    names: List[str] = []
    params_used: List[Dict[str, Any]] = []

    if stage == "fit_transform":
        for entry in y_ops or []:
            spec = get_y_op(entry.name)
            ParamT = spec.ParamModel
            params = ParamT(**(entry.params or {})).model_dump() if ParamT is not None else dict(entry.params or {})
            params_obj = ParamT(**params) if ParamT is not None else params
            # Contract: enforce requires before fit; enforce produces after transform.
            fit_contract = Contract(
                category="yops",
                requires=spec.requires,
                produces=spec.produces,
            )
            fit_ctx = ctx.for_plugin(category="yops", name=entry.name, contract=fit_contract)
            fit_ctx.precheck_requires(stage="fit_transform")
            # side-effect: write fitted stats into ctx under yops/<name>/*
            spec.fit_fn(Yt, params_obj, ctx=fit_ctx)
            Yt = np.asarray(spec.transform_fn(Yt, params_obj, ctx=fit_ctx), dtype=float)
            fit_ctx.postcheck_produces(stage="fit_transform")

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
        spec = get_y_op(name)
        ParamT = spec.ParamModel
        params_obj = ParamT(**params) if ParamT is not None else params
        inv_contract = Contract(
            category="yops",
            requires=spec.requires + spec.produces,
            produces=tuple(),
        )
        inv_ctx = ctx.for_plugin(category="yops", name=name, contract=inv_contract)
        inv_ctx.precheck_requires(stage="inverse")
        Yt = np.asarray(spec.inverse_fn(Yt, params_obj, ctx=inv_ctx), dtype=float)
    return Yt
