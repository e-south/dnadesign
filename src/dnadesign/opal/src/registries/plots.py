"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/plots.py

OPAL — Plot Registry (auto-discovers built-ins and plugin entry points)

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_PLOTS: Dict[str, Callable] = {}
_PLOT_META: Dict[str, "PlotMeta"] = {}


@dataclass(frozen=True)
class PlotMeta:
    summary: str
    params: Dict[str, str] = field(default_factory=dict)
    requires: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _dbg(msg: str) -> None:
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on"):
        print(f"[opal.debug.plots] {msg}", file=sys.stderr)


def register_plot(name: str, *, meta: Optional[PlotMeta] = None):
    """Decorator to register a plot render function under a stable string id."""
    if not isinstance(name, str) or not name:
        raise ValueError("plot name must be a non-empty string")

    def _wrap(fn: Callable):
        if name in _PLOTS:
            raise ValueError(f"plot '{name}' already registered")
        _PLOTS[name] = fn
        if meta is not None:
            _PLOT_META[name] = meta
            setattr(fn, "__opal_plot_meta__", meta)
        _dbg(f"registered plot: {name}")
        return fn

    return _wrap


# -------- auto-discovery of built-ins & entry-point plugins -------------------
_BUILTINS_LOADED = False
_PLUGINS_LOADED = False


def _ensure_builtins_loaded() -> None:
    """Import package-shipped plot modules that self-register via @register_plot."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    try:
        pkg = importlib.import_module("dnadesign.opal.src.plots")
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
                _dbg(f"imported built-in plot module: {fq}")
            except Exception as e:
                _dbg(f"FAILED importing {fq}: {e!r}")
                continue
    except Exception as e:
        _dbg(f"FAILED importing package dnadesign.opal.src.plots: {e!r}")
    _BUILTINS_LOADED = True


def _iter_entry_points(group: str):
    # Prefer importlib.metadata; fall back to pkg_resources if needed
    try:
        import importlib.metadata as _ilmd

        eps = _ilmd.entry_points()
        try:
            # Python 3.10+ interface
            yield from eps.select(group=group)  # type: ignore[attr-defined]
            return
        except Exception:
            # Older interface
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
    for ep in _iter_entry_points("dnadesign.opal.plots"):
        try:
            ep.load()
            _dbg(f"loaded plot plugin entry point: {getattr(ep, 'name', '?')} from {getattr(ep, 'module', '?')}")
        except Exception as e:
            _dbg(f"FAILED loading plot plugin entry point {ep!r}: {e!r}")
            continue
    _PLUGINS_LOADED = True


def _ensure_all_loaded() -> None:
    _ensure_builtins_loaded()
    _ensure_plugins_loaded()


def get_plot(name: str) -> Callable:
    _ensure_all_loaded()
    try:
        return _PLOTS[name]
    except KeyError:
        avail_list = sorted(_PLOTS.keys())
        avail = ", ".join(avail_list)
        hint = (
            " Built-ins failed to load or registry is empty. Ensure the package is installed"
            " and that 'dnadesign.opal.src.plots' is importable; or install/register a plot plugin"
            " exposing the 'dnadesign.opal.plots' entry point group."
            if not avail_list
            else " Did you mean one of the available plots above?"
        )
        raise KeyError(f"Unknown plot kind '{name}'. Registered: [{avail}].{hint}")


def list_plots() -> List[str]:
    _ensure_all_loaded()
    return sorted(_PLOTS.keys())


def get_plot_meta(name: str) -> Optional[PlotMeta]:
    _ensure_all_loaded()
    fn = _PLOTS.get(name)
    if fn is None:
        raise KeyError(f"Unknown plot kind '{name}'")
    meta = _PLOT_META.get(name) or getattr(fn, "__opal_plot_meta__", None)
    if meta is not None:
        return meta
    doc = inspect.getdoc(fn)
    if doc:
        summary = doc.strip().splitlines()[0].strip()
        return PlotMeta(summary=summary)
    return None
