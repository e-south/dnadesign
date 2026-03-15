"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/bootstrap.py

Explicit infer registry bootstrap contract for model/function adapter defaults.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .adapters import register_defaults

_BOOTSTRAPPED = False


def initialize_registry() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    register_defaults()
    _BOOTSTRAPPED = True

