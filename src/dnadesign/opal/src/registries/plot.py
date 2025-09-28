"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/plot.py

OPAL — Plot Registry

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable, Dict, List

_PLOTS: Dict[str, Callable] = {}


def register_plot(kind: str):
    """
    Decorator to register a plot plugin.

    Usage:
        @register_plot("scatter_score_vs_rank")
        def render(context, params): ...
    """
    if not isinstance(kind, str) or not kind:
        raise ValueError("register_plot(kind): 'kind' must be a non-empty string")

    def _wrap(func: Callable):
        if kind in _PLOTS:
            raise RuntimeError(f"Plot kind '{kind}' already registered")
        _PLOTS[kind] = func
        return func

    return _wrap


def get_plot(kind: str) -> Callable:
    """Resolve a plot function by kind or raise KeyError."""
    try:
        return _PLOTS[kind]
    except KeyError:
        raise KeyError(
            f"Unknown plot kind: '{kind}'. "
            f"Available: {', '.join(sorted(_PLOTS.keys())) or '(none registered)'}"
        )


def list_plots() -> List[str]:
    """List registered plot kinds."""
    return sorted(_PLOTS.keys())
