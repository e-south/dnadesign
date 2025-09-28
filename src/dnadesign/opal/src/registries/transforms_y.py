"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/transforms_y.py

OPAL registry: Y transforms (CSV/tidy → model-ready y)

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable, Dict, List

# Registry: name -> callable(records_df, csv_df, *, params, y_expected_length, setpoint_vector, y_column_name) -> (labels_df, preview_dict)  # noqa
_REG_Y: Dict[str, Callable] = {}


def register_transform_y(name: str):
    """Decorator to register a Y ingest transform by name."""

    def _wrap(func: Callable):
        if name in _REG_Y:
            raise ValueError(f"transform_y '{name}' already registered")
        _REG_Y[name] = func
        return func

    return _wrap


def get_transform_y(name: str) -> Callable:
    try:
        return _REG_Y[name]
    except KeyError:
        raise KeyError(f"transform_y '{name}' not found. Available: {sorted(_REG_Y)}")


def list_transforms_y() -> List[str]:
    return sorted(_REG_Y)
