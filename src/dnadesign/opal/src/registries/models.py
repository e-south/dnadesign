"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/models.py

Model registry.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from typing import Any, Dict, List, Protocol


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
    try:
        pkg = importlib.import_module("dnadesign.opal.src.models")
        _dbg(f"imported package: {pkg.__name__} ({getattr(pkg, '__file__', '?')})")
        try:
            pkg_path = pkg.__path__  # type: ignore[attr-defined]
        except Exception:
            pkg_path = []
        for mod in pkgutil.iter_modules(pkg_path):
            if mod.name.startswith("_"):
                continue
            fq = f"{pkg.__name__}.{mod.name}"
            try:
                importlib.import_module(fq)
                _dbg(f"imported built-in model module: {fq}")
            except Exception as e:
                _dbg(f"FAILED importing {fq}: {e!r}")
                continue
    except Exception as e:
        _dbg(f"FAILED importing package dnadesign.opal.src.models: {e!r}")
    _BUILTINS_LOADED = True


def _iter_entry_points(group: str):
    try:
        import importlib.metadata as _ilmd

        eps = _ilmd.entry_points()
        try:
            yield from eps.select(group=group)  # type: ignore[attr-defined]
            return
        except Exception:
            for ep in eps.get(group, []):  # type: ignore[index]
                yield ep
            return
    except Exception as e:
        _dbg(f"importlib.metadata unavailable or failed: {e!r}")
    try:
        import pkg_resources

        for ep in pkg_resources.iter_entry_points(group):
            yield ep
        return
    except Exception as e:
        _dbg(f"pkg_resources fallback unavailable or failed: {e!r}")


def _ensure_plugins_loaded() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    for ep in _iter_entry_points("dnadesign.opal.models"):
        try:
            ep.load()
            _dbg(f"loaded plugin entry point: {getattr(ep, 'name', '?')} from {getattr(ep, 'module', '?')}")
        except Exception as e:
            _dbg(f"FAILED loading plugin entry point {ep!r}: {e!r}")
            continue
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


def get_model(name: str, params: dict):
    """
    Instantiate a model via its registered factory.

    We try tolerant call patterns:
      • factory(params=params)
      • class with .from_params(params)        (if factory provides it)
      • factory(params)                        (positional)
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

    # try kwargs
    try:
        return factory(params=params)
    except TypeError:
        pass

    # try classmethod from_params
    fp = getattr(factory, "from_params", None)
    if callable(fp):
        try:
            return fp(params)
        except TypeError:
            pass

    # try positional
    try:
        return factory(params)
    except Exception as e:
        raise TypeError(f"cannot construct model '{name}': {e}") from e


def list_models() -> List[str]:
    _ensure_all_loaded()
    return sorted(_REG_M)
