"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/runtime/memory_guard.py

Memory guardrails for OPAL round execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.utils import OpalError

DEFAULT_X_MATRIX_MAX_GIB = 8.0
DEFAULT_X_MATRIX_OVERHEAD_FACTOR = 4.0


@dataclass(frozen=True)
class XMatrixMemoryEstimate:
    row_count: int
    x_dim: int
    item_size_bytes: int
    raw_bytes: int
    estimated_bytes: int
    overhead_factor: float
    max_bytes: int
    max_gib: float

    @property
    def raw_gib(self) -> float:
        return _bytes_to_gib(self.raw_bytes)

    @property
    def estimated_gib(self) -> float:
        return _bytes_to_gib(self.estimated_bytes)


def enforce_x_matrix_memory_budget(
    *,
    row_count: int,
    x_dim: int,
    max_gib: float | None,
    item_size_bytes: int = 4,
    overhead_factor: float = DEFAULT_X_MATRIX_OVERHEAD_FACTOR,
    context: str = "OPAL X matrix",
) -> XMatrixMemoryEstimate:
    """Fail before OPAL materializes an X matrix batch that exceeds the configured safety budget."""

    estimate = estimate_x_matrix_memory(
        row_count=row_count,
        x_dim=x_dim,
        item_size_bytes=item_size_bytes,
        max_gib=max_gib,
        overhead_factor=overhead_factor,
    )
    if estimate.estimated_bytes > estimate.max_bytes:
        raise OpalError(
            f"{context} exceeds safety.max_x_matrix_gib. "
            f"rows={estimate.row_count}, x_dim={estimate.x_dim}, "
            f"raw={estimate.raw_gib:.2f} GiB, estimated_in_memory={estimate.estimated_gib:.2f} GiB, "
            f"budget={estimate.max_gib:.2f} GiB. Reduce scoring.score_batch_size, narrow the candidate universe, "
            "or use a larger explicit safety.max_x_matrix_gib only on a host with enough RAM."
        )
    return estimate


def estimate_x_matrix_memory(
    *,
    row_count: int,
    x_dim: int,
    item_size_bytes: int,
    max_gib: float | None,
    overhead_factor: float = DEFAULT_X_MATRIX_OVERHEAD_FACTOR,
) -> XMatrixMemoryEstimate:
    rows = _positive_int(row_count, "row_count")
    dim = _positive_int(x_dim, "x_dim")
    item_size = _positive_int(item_size_bytes, "item_size_bytes")
    budget_gib = DEFAULT_X_MATRIX_MAX_GIB if max_gib is None else _positive_float(max_gib, "max_x_matrix_gib")
    overhead = _positive_float(overhead_factor, "overhead_factor")
    raw_bytes = int(rows * dim * item_size)
    estimated_bytes = int(raw_bytes * overhead)
    max_bytes = int(budget_gib * (1024**3))
    return XMatrixMemoryEstimate(
        row_count=rows,
        x_dim=dim,
        item_size_bytes=item_size,
        raw_bytes=raw_bytes,
        estimated_bytes=estimated_bytes,
        overhead_factor=overhead,
        max_bytes=max_bytes,
        max_gib=budget_gib,
    )


def infer_x_dim_from_series(series: pd.Series, *, x_column: str) -> int:
    """Infer vector width from a loaded X column without building a dense matrix."""

    for value in series.tolist():
        if _is_missing_cell(value):
            continue
        return _coerce_x_width(value, x_column=x_column)
    raise OpalError(f"X column '{x_column}' has no non-null values for memory guard estimation.")


def _coerce_x_width(value: Any, *, x_column: str) -> int:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            import json

            try:
                value = json.loads(stripped)
            except Exception as exc:
                raise OpalError(f"X column '{x_column}' contains an invalid JSON vector cell.") from exc
        else:
            return 1
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value, dtype=object).ravel()
        if arr.size <= 0:
            raise OpalError(f"X column '{x_column}' contains an empty vector cell.")
        return int(arr.size)
    try:
        float(value)
    except (TypeError, ValueError) as exc:
        raise OpalError(f"X column '{x_column}' contains a non-vector, nonnumeric cell for memory estimation.") from exc
    return 1


def _positive_int(value: int, label: str) -> int:
    out = int(value)
    if out <= 0:
        raise OpalError(f"{label} must be positive; got {value!r}.")
    return out


def _positive_float(value: float, label: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise OpalError(f"{label} must be a positive finite number; got {value!r}.")
    return out


def _bytes_to_gib(value: int) -> float:
    return float(value) / float(1024**3)


def _is_missing_cell(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return bool(np.isnan(value))
    try:
        marker = pd.isna(value)
    except Exception:
        return False
    if isinstance(marker, (bool, np.bool_)):
        return bool(marker)
    return False
