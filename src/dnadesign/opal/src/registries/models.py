"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/models.py

Registers model factories and resolves built-in and plugin models. Provides
model construction and loading helpers with contract checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Protocol

from .loader import load_builtin_modules, load_entry_points


def _dbg(msg: str) -> None:
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on"):
        print(f"[opal.debug.models] {msg}", file=sys.stderr)


class _ModelFactory(Protocol):
    def __call__(self, *args, **kwargs): ...


# Registry: name -> factory(params) -> model_instance
_REG_M: Dict[str, _ModelFactory] = {}

_BUILTINS_LOADED = False
_PLUGINS_LOADED = False


def _ensure_builtins_loaded() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    load_builtin_modules("dnadesign.opal.src.models", label="models", debug=_dbg)
    _BUILTINS_LOADED = True


def _ensure_plugins_loaded() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    load_entry_points("dnadesign.opal.models", label="models", debug=_dbg)
    _PLUGINS_LOADED = True


def _ensure_all_loaded() -> None:
    _ensure_builtins_loaded()
    _ensure_plugins_loaded()


def register_model(name: str):
    """Decorator to register a model factory. The factory is called with (params)."""

    def _wrap(factory: _ModelFactory):
        if name in _REG_M:
            raise ValueError(f"model '{name}' already registered")
        _REG_M[name] = factory
        _dbg(f"registered model: {name}")
        return factory

    return _wrap


def _wrap_model_for_ctx(name: str, model: Any) -> Any:
    """
    Wrap model.fit/predict to enforce RoundCtx contracts when a PluginCtx is provided.
    """
    contract = getattr(model, "__opal_contract__", None)
    if contract is None:
        return model

    if getattr(model, "__opal_ctx_wrapped__", False):
        return model

    def _wrap_method(method_name: str) -> None:
        orig = getattr(model, method_name, None)
        if not callable(orig):
            return

        def _wrapped(*args, **kwargs):
            ctx = kwargs.get("ctx")
            if ctx is not None:
                ctx.precheck_requires()
            out = orig(*args, **kwargs)
            if ctx is not None:
                ctx.postcheck_produces()
            return out

        setattr(model, method_name, _wrapped)

    _wrap_method("fit")
    _wrap_method("predict")
    setattr(model, "__opal_ctx_wrapped__", True)
    return model


def get_model(name: str, params: dict):
    """
    Instantiate a model via its registered factory.
    Required signature: factory(params: dict) -> model_instance.
    """
    _ensure_all_loaded()
    if name not in _REG_M:
        avail_list = sorted(_REG_M)
        avail = ", ".join(avail_list)
        hint = (
            " Built-ins failed to load or registry is empty. Ensure the package is installed"
            " and that 'dnadesign.opal.src.models' is importable; or install/register a model plugin"
            " exposing the 'dnadesign.opal.models' entry point group."
            if not avail_list
            else " Did you mean one of the available models above?"
        )
        raise KeyError(f"model '{name}' not found. Available: [{avail}].{hint}")

    factory: Any = _REG_M[name]
    try:
        model = factory(params)
    except TypeError as e:
        raise TypeError(f"model factory '{name}' must accept a params dict.") from e
    return _wrap_model_for_ctx(name, model)


def list_models() -> List[str]:
    _ensure_all_loaded()
    return sorted(_REG_M)


def load_model(name: str, path: str, params: dict | None = None):
    """
    Load a persisted model by name.
    Required interface: model class must implement `load(path: str, params: dict | None = None)`.
    """
    _ensure_all_loaded()
    if name not in _REG_M:
        avail_list = sorted(_REG_M)
        avail = ", ".join(avail_list)
        raise KeyError(f"model '{name}' not found. Available: [{avail}].")
    factory: Any = _REG_M[name]
    loader = getattr(factory, "load", None)
    if not callable(loader):
        raise TypeError(f"model '{name}' does not implement required load(path, params=None) interface.")
    model = loader(path, params=params)
    return _wrap_model_for_ctx(name, model)
