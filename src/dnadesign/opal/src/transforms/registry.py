"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/transforms/registry.py

Transform factory/registry.

Resolves a transform name and parameters into a concrete object.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict

from .concat_stub import ConcatStubTransform
from .identity import IdentityTransform


def get_transform(name: str, params: Dict[str, Any]):
    name = name.lower()
    if name == "identity":
        return IdentityTransform(**(params or {}))
    if name == "concat":
        return ConcatStubTransform(**(params or {}))
    raise ValueError(f"Unknown transform: {name}")
