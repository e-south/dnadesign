"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_protocol_fields.py

Strict field parsers for the persisted behavior shadow protocol.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class BehaviorProtocolError(ValueError):
    """Raised when the study-owned shadow protocol is missing or drifts."""


@dataclass(frozen=True)
class BehaviorTargetView:
    id: str
    target_mask: tuple[float, ...]


def parse_target_views(value: object, *, state_count: int) -> tuple[BehaviorTargetView, ...]:
    """Parse ordered, unique, nondegenerate binary target masks."""

    if not isinstance(value, list) or not value:
        raise BehaviorProtocolError("target_views must be a non-empty list.")
    views: list[BehaviorTargetView] = []
    for index, raw in enumerate(value):
        row = require_mapping(raw, context=f"target_views[{index}]")
        require_exact_fields(row, {"id", "target_mask"}, context=f"target_views[{index}]")
        view_id = nonempty_string(row["id"], field=f"target_views[{index}].id")
        raw_mask = row["target_mask"]
        if not isinstance(raw_mask, list) or len(raw_mask) != state_count:
            raise BehaviorProtocolError(f"target view {view_id!r} must define {state_count} mask values.")
        if any(isinstance(item, bool) for item in raw_mask):
            raise BehaviorProtocolError(f"target view {view_id!r} mask must use numeric zero or one.")
        try:
            mask = tuple(float(item) for item in raw_mask)
        except (TypeError, ValueError) as exc:
            raise BehaviorProtocolError(f"target view {view_id!r} mask must be numeric.") from exc
        if set(mask) - {0.0, 1.0} or not any(mask) or all(mask):
            raise BehaviorProtocolError(f"target view {view_id!r} must contain at least one ON and OFF state.")
        views.append(BehaviorTargetView(id=view_id, target_mask=mask))
    ids = tuple(view.id for view in views)
    if len(ids) != len(set(ids)):
        raise BehaviorProtocolError("target view ids must be unique.")
    return tuple(views)


def parse_state_ids(value: object) -> tuple[str, ...]:
    """Parse an ordered, unique state identity sequence."""

    if not isinstance(value, list) or len(value) < 2:
        raise BehaviorProtocolError("assay.state_ids must contain at least two ordered states.")
    state_ids = tuple(nonempty_string(item, field="state_ids") for item in value)
    if len(state_ids) != len(set(state_ids)):
        raise BehaviorProtocolError("assay.state_ids must be unique.")
    return state_ids


def require_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BehaviorProtocolError(f"{context} must be a mapping.")
    return value


def require_exact_fields(payload: dict[str, object], expected: set[str], *, context: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise BehaviorProtocolError(
            f"{context} fields must be exactly {sorted(expected)}; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}."
        )


def require_literal(payload: dict[str, object], field: str, expected: object, *, context: str) -> None:
    if payload.get(field) != expected:
        raise BehaviorProtocolError(f"{context}.{field} must be {expected!r}.")


def nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BehaviorProtocolError(f"{field} must be a non-empty string.")
    return value.strip()


def positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BehaviorProtocolError(f"{field} must be a positive finite number.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise BehaviorProtocolError(f"{field} must be a positive finite number.")
    return parsed


__all__ = [
    "BehaviorProtocolError",
    "BehaviorTargetView",
    "nonempty_string",
    "parse_state_ids",
    "parse_target_views",
    "positive_float",
    "require_exact_fields",
    "require_literal",
    "require_mapping",
]
