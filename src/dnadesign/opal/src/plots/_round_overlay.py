"""Generic round-overlay helpers for OPAL plot primitives."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def resolve_highlight_round(value: Any, available_rounds: Iterable[int]) -> int | None:
    rounds = sorted({int(round_index) for round_index in available_rounds})
    if not rounds or value is None:
        return None
    if isinstance(value, bool):
        return rounds[-1] if value else None
    token = str(value).strip().lower()
    if token in {"", "none", "off", "false", "no"}:
        return None
    if token in {"latest", "current", "true"}:
        return rounds[-1]
    try:
        round_index = int(token)
    except ValueError as exc:
        raise ValueError("highlight_round must be 'latest', true/false, or an integer round.") from exc
    if round_index not in rounds:
        raise ValueError(f"highlight_round={round_index} is not present in plotted rounds: {rounds}")
    return round_index


def add_round_vline(ax: Any, round_index: int, *, label: str = "highlight round") -> None:
    ax.axvline(
        int(round_index),
        color="#202020",
        linestyle="-",
        linewidth=1.1,
        alpha=0.55,
        label=label,
        zorder=1,
    )
