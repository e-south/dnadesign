"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/objectives.py

Registers objective functions and loads built-in and plugin objectives. Provides
objective access with PluginCtx contract enforcement.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any, Dict, List, Protocol

from ..core.round_context import PluginCtx
from .loader import load_builtin_modules, load_entry_points


def _dbg(msg: str) -> None:
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on"):
        print(f"[opal.debug.objectives] {msg}", file=sys.stderr)


class _ObjectiveFn(Protocol):
    def __call__(self, *, y_pred, params: Dict[str, Any], ctx, train_view, y_pred_std) -> Any: ...


# Registry: name -> callable(y_pred, *, params, ctx, train_view) -> ObjectiveResult
_REG_O: Dict[str, _ObjectiveFn] = {}

_BUILTINS_LOADED = False
_PLUGINS_LOADED = False

_DECLARED_SCORE_CHANNELS_ATTR = "__opal_score_channels__"
_DECLARED_UNCERTAINTY_CHANNELS_ATTR = "__opal_uncertainty_channels__"
_DECLARED_SCORE_MODES_ATTR = "__opal_score_modes__"


def _validate_objective_signature(func: _ObjectiveFn, *, name: str) -> None:
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    expected = ["y_pred", "params", "ctx", "train_view", "y_pred_std"]
    if len(params) < len(expected):
        raise ValueError(
            f"objective '{name}' must define keyword-only signature (*, y_pred, params, ctx, train_view, y_pred_std)."
        )
    got_names = [p.name for p in params[: len(expected)]]
    if got_names != expected:
        raise ValueError(f"objective '{name}' has invalid required signature names {got_names}; expected {expected}.")
    if any(p.kind is not inspect.Parameter.KEYWORD_ONLY for p in params):
        raise ValueError(
            f"objective '{name}' must define keyword-only signature (*, y_pred, params, ctx, train_view, y_pred_std)."
        )
    for p in params[len(expected) :]:
        if p.default is inspect.Parameter.empty:
            raise ValueError(
                f"objective '{name}' extra keyword-only parameter '{p.name}' must provide a default value."
            )


def _ensure_builtins_loaded() -> None:
    """Import package-shipped objective modules that self-register via @register_objective."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    load_builtin_modules("dnadesign.opal.src.objectives", label="objective", debug=_dbg)
    _BUILTINS_LOADED = True


def _ensure_plugins_loaded() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    load_entry_points("dnadesign.opal.objectives", label="objective", debug=_dbg)
    _PLUGINS_LOADED = True


def _ensure_all_loaded() -> None:
    _ensure_builtins_loaded()
    _ensure_plugins_loaded()


def register_objective(name: str):
    """Decorator to register an objective by name."""

    def _wrap(func: _ObjectiveFn):
        if name in _REG_O:
            raise ValueError(f"objective '{name}' already registered")
        _validate_objective_signature(func, name=name)
        _REG_O[name] = func
        _dbg(f"registered objective: {name}")
        return func

    return _wrap


def _wrap_for_ctx_enforcement(name: str, fn: _ObjectiveFn) -> _ObjectiveFn:
    """
    Return a fn that enforces PluginCtx contract pre/post checks if ctx is provided.
    Preserve __opal_contract__ so run_round can build a PluginCtx.
    """
    contract = getattr(fn, "__opal_contract__", None)

    def _wrapped(
        *,
        y_pred,
        params: Dict[str, Any],
        ctx: PluginCtx | None = None,
        train_view=None,
        y_pred_std=None,
    ):
        if ctx is not None:
            try:
                ctx.precheck_requires(stage="objective")
            except Exception:
                raise
        try:
            out = fn(y_pred=y_pred, params=params, ctx=ctx, train_view=train_view, y_pred_std=y_pred_std)
        except Exception:
            if ctx is not None:
                ctx.reset_stage_state()
            raise
        if ctx is not None:
            try:
                ctx.postcheck_produces(stage="objective")
            except Exception:
                raise
        return out

    if contract is not None:
        setattr(_wrapped, "__opal_contract__", contract)
    for attr in (
        _DECLARED_SCORE_CHANNELS_ATTR,
        _DECLARED_UNCERTAINTY_CHANNELS_ATTR,
        _DECLARED_SCORE_MODES_ATTR,
    ):
        if hasattr(fn, attr):
            setattr(_wrapped, attr, getattr(fn, attr))
    return _wrapped  # type: ignore[return-value]


def get_objective(name: str) -> _ObjectiveFn:
    _ensure_all_loaded()
    try:
        fn = _REG_O[name]
    except KeyError:
        avail_list = sorted(_REG_O)
        avail = ", ".join(avail_list)
        hint = (
            " Built-ins failed to load or registry is empty. Ensure the package is installed"
            " and that 'dnadesign.opal.src.objectives' is importable; or install/register an objective plugin"
            " exposing the 'dnadesign.opal.objectives' entry point group."
            if not avail_list
            else " Did you mean one of the available objectives above?"
        )
        raise KeyError(f"objective '{name}' not found. Available: [{avail}].{hint}")
    return _wrap_for_ctx_enforcement(name, fn)


def list_objectives() -> List[str]:
    _ensure_all_loaded()
    return sorted(_REG_O)


def _declared_channels(fn: _ObjectiveFn, *, attr_name: str, label: str) -> tuple[str, ...]:
    raw = getattr(fn, attr_name, ())
    if raw is None:
        return tuple()
    if isinstance(raw, str):
        raw = (raw,)
    if not isinstance(raw, (list, tuple, set)):
        raise ValueError(f"objective declared {label} channels must be a sequence of strings.")
    out: List[str] = []
    for ch in raw:
        c = str(ch).strip()
        if not c:
            raise ValueError(f"objective declared {label} channels must be non-empty strings.")
        out.append(c)
    return tuple(out)


def _declared_score_modes(fn: _ObjectiveFn) -> Dict[str, str]:
    raw = getattr(fn, _DECLARED_SCORE_MODES_ATTR, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("objective declared score modes must be a mapping: channel -> mode.")
    out: Dict[str, str] = {}
    for channel, mode in raw.items():
        key = str(channel).strip()
        val = str(mode).strip().lower()
        if not key:
            raise ValueError("objective declared score mode channel names must be non-empty strings.")
        if val not in {"maximize", "minimize"}:
            raise ValueError(
                f"objective declared score mode for channel '{key}' must be 'maximize' or 'minimize'; got {mode!r}."
            )
        out[key] = val
    return out


def get_objective_declared_channels(name: str) -> Dict[str, Any]:
    _ensure_all_loaded()
    try:
        fn = _REG_O[name]
    except KeyError:
        avail = ", ".join(sorted(_REG_O))
        raise KeyError(f"objective '{name}' not found. Available: [{avail}].")
    return {
        "score": _declared_channels(fn, attr_name=_DECLARED_SCORE_CHANNELS_ATTR, label="score"),
        "uncertainty": _declared_channels(
            fn,
            attr_name=_DECLARED_UNCERTAINTY_CHANNELS_ATTR,
            label="uncertainty",
        ),
        "score_modes": _declared_score_modes(fn),
    }
