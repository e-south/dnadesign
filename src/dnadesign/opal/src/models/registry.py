"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/models/registry.py

Resolves a model name (e.g., "random_forest") to a concrete implementation.
Keeps the CLI/config decoupled from sklearn specifics and makes it easy to
add new top-layer models without touching callers.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict

from .random_forest import RandomForestModel


def get_model(name: str, params: Dict[str, Any]):
    name = name.lower()
    if name == "random_forest":
        return RandomForestModel(**(params or {}))
    raise ValueError(f"Unknown model type: {name}")
