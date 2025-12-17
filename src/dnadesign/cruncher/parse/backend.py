"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/parse/backend.py

Generic registry for PWM parsers.

Add a new format by:
    1. creating parse/parsers/<fmt>.py
    2. decorating a function with  @register("MYFMT")

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Callable, Dict

from .model import PWM

_REGISTRY: Dict[str, Callable[[Path], PWM]] = {}
_PARSERS_IMPORTED: bool = False


def register(fmt: str) -> Callable[[Callable[[Path], PWM]], Callable[[Path], PWM]]:
    fmt = fmt.upper()

    def _decorator(fn):
        _REGISTRY[fmt] = fn
        return fn

    return _decorator


def guess_format(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def _ensure_parsers_imported() -> None:
    """
    Ensure built-in parsers are imported so their @register(...) decorators run.

    This makes Registry/load_pwm robust: callers do not need to remember to import
    dnadesign.cruncher.parse (or individual parser modules) for side effects.
    """
    global _PARSERS_IMPORTED
    if _PARSERS_IMPORTED:
        return
    # Import the parsers *package*; its __init__.py imports built-ins.
    import_module("dnadesign.cruncher.parse.parsers")
    _PARSERS_IMPORTED = True


def load_pwm(path: Path, fmt: str | None = None) -> PWM:
    _ensure_parsers_imported()
    fmt = (fmt or guess_format(path)).upper()
    try:
        return _REGISTRY[fmt](path)
    except KeyError as e:
        raise ValueError(f"No parser registered for format '{fmt}'") from e
