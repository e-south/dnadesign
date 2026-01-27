"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/runtime/y_ops_inverse.py

Y-ops normalization and inverse-transform helpers for runtime scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..registries.transforms_y import run_y_ops_pipeline


@dataclass(frozen=True)
class YOpEntry:
    name: str
    params: dict


def normalize_y_ops_config(y_ops: Sequence[Mapping[str, Any]]) -> list[YOpEntry]:
    out: list[YOpEntry] = []
    for entry in y_ops or []:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        if not name:
            continue
        params = dict(entry.get("params") or {})
        out.append(YOpEntry(name=str(name), params=params))
    return out


def apply_y_ops_inverse(
    *,
    y_ops: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    ctx: Any,
) -> np.ndarray:
    entries = normalize_y_ops_config(y_ops)
    return run_y_ops_pipeline(stage="inverse", y_ops=entries, Y=y, ctx=ctx)
