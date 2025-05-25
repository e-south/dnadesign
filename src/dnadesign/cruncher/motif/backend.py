"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/motif/backend.py

Generic registry for PWM parsers.

Add a new format by:
    1. creating motif/parsers/<fmt>.py
    2. decorating a function with  @register("MYFMT")

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict

from .model import PWM

_REGISTRY: Dict[str, Callable[[Path], PWM]] = {}


def register(fmt: str) -> Callable[[Callable[[Path], PWM]], Callable[[Path], PWM]]:
    fmt = fmt.upper()

    def _decorator(fn):
        _REGISTRY[fmt] = fn
        return fn

    return _decorator


def guess_format(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def load_pwm(path: Path, fmt: str | None = None) -> PWM:
    fmt = (fmt or guess_format(path)).upper()
    try:
        return _REGISTRY[fmt](path)
    except KeyError as e:
        raise ValueError(f"No parser registered for format '{fmt}'") from e
