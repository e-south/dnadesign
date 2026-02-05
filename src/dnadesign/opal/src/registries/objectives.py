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

import os
import sys
from typing import Any, Dict, List, Protocol

from ..core.round_context import PluginCtx
from .loader import load_builtin_modules, load_entry_points


def _dbg(msg: str) -> None:
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on"):
        print(f"[opal.debug.objectives] {msg}", file=sys.stderr)


class _ObjectiveFn(Protocol):
    def __call__(self, *, y_pred, params: Dict[str, Any], ctx=None, train_view=None) -> Any: ...


# Registry: name -> callable(y_pred, *, params, ctx, train_view) -> ObjectiveResult
_REG_O: Dict[str, _ObjectiveFn] = {}

_BUILTINS_LOADED = False
_PLUGINS_LOADED = False


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

    def _wrapped(*, y_pred, params: Dict[str, Any], var, ctx: PluginCtx | None = None, train_view=None):
        if ctx is not None:
            try:
                ctx.precheck_requires(stage="objective")
            except Exception:
                raise
        try:
            out = fn(y_pred=y_pred, params=params, ctx=ctx, train_view=train_view)
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
