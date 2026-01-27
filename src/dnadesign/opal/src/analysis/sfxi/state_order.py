"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/sfxi/state_order.py

Canonical SFXI state order and helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

from ...objectives.sfxi_math import STATE_ORDER as _STATE_ORDER
from ...objectives.sfxi_math import assert_state_order as _assert_state_order

STATE_ORDER = _STATE_ORDER


def assert_state_order(order: Sequence[str]) -> None:
    _assert_state_order(order)


def require_state_order(order: Sequence[str] | None) -> Sequence[str]:
    if order is None:
        raise ValueError("state_order is required and must be [00, 10, 01, 11].")
    assert_state_order(order)
    return order


def index_for_state(state: str, *, order: Sequence[str] = STATE_ORDER) -> int:
    assert_state_order(order)
    try:
        return list(order).index(state)
    except ValueError as exc:
        raise ValueError(f"Unknown state: {state!r}") from exc
