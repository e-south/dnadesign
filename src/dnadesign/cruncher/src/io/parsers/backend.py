"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/io/parsers/backend.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Sequence

from dnadesign.cruncher.core.pwm import PWM

_REGISTRY: Dict[str, Callable[[Path], PWM]] = {}
_PARSERS_IMPORTED: bool = False
_EXTRA_MODULES_IMPORTED: set[str] = set()


def register(fmt: str) -> Callable[[Callable[[Path], PWM]], Callable[[Path], PWM]]:
    fmt = fmt.upper()

    def _decorator(fn):
        _REGISTRY[fmt] = fn
        return fn

    return _decorator


def guess_format(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def _import_extra_modules(extra_modules: Sequence[str]) -> None:
    for module in extra_modules:
        name = str(module).strip()
        if not name:
            raise ValueError("extra parser module names must be non-empty strings")
        if name in _EXTRA_MODULES_IMPORTED:
            continue
        try:
            import_module(name)
        except Exception as exc:  # pragma: no cover - propagate with context
            raise ImportError(f"Failed to import parser module '{name}': {exc}") from exc
        _EXTRA_MODULES_IMPORTED.add(name)


def _ensure_parsers_imported(extra_modules: Sequence[str] | None = None) -> None:
    """
    Ensure built-in parsers are imported so their @register(...) decorators run.

    This makes load_pwm robust: callers do not need to remember to import
    dnadesign.cruncher.io.parsers (or individual parser modules) for side effects.
    """
    global _PARSERS_IMPORTED
    if _PARSERS_IMPORTED:
        if extra_modules:
            _import_extra_modules(extra_modules)
        return
    # Import the parsers *package*; its __init__.py imports built-ins.
    import_module("dnadesign.cruncher.io.parsers")
    if extra_modules:
        _import_extra_modules(extra_modules)
    _PARSERS_IMPORTED = True


def load_pwm(path: Path, fmt: str | None = None, *, extra_modules: Sequence[str] | None = None) -> PWM:
    _ensure_parsers_imported(extra_modules)
    fmt = (fmt or guess_format(path)).upper()
    try:
        return _REGISTRY[fmt](path)
    except KeyError as e:
        raise ValueError(f"No parser registered for format '{fmt}'") from e
