# ABOUTME: Registers plot functions and loads built-in and plugin plot modules.
# ABOUTME: Provides plot lookup and metadata for Opal reporting.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/plots.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .loader import load_builtin_modules, load_entry_points

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
    load_builtin_modules("dnadesign.opal.src.plots", label="plot", debug=_dbg)
    _BUILTINS_LOADED = True


def _ensure_plugins_loaded() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    load_entry_points("dnadesign.opal.plots", label="plot", debug=_dbg)
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
