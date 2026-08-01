"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/folding/__init__.py

Neutral folding-contract exports, loaded on first use.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .secondary_structure_prediction_v1 import (  # noqa: F401
        SecondaryStructurePredictionRequestV1,
        SecondaryStructurePredictionV1,
    )

__all__ = ["SecondaryStructurePredictionRequestV1", "SecondaryStructurePredictionV1"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".secondary_structure_prediction_v1", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
