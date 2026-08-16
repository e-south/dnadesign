"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/selection_round_encoding.py

Maps selection rounds to stable categorical plot encodings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_SELECTION_ROUND_MARKERS = ("D", "P", "X", "s", "^", "v", "<", ">")


def selection_round_palette_index(contract: Mapping[str, Any]) -> dict[int, int]:
    """Return each declared round's stable position in the categorical palette."""

    rounds = [int(value) for value in contract.get("selection_rounds") or []]
    if len(rounds) != len(set(rounds)):
        raise ValueError("Selection rounds must be unique.")
    return {round_k: index for index, round_k in enumerate(rounds)}


def selection_round_marker(*, round_index: int, palette_index: int) -> str:
    """Return a stable marker without cycling mature round categories."""

    if palette_index < 0:
        raise ValueError("Selection-round palette indices must be non-negative.")
    if palette_index < len(_SELECTION_ROUND_MARKERS):
        return _SELECTION_ROUND_MARKERS[palette_index]
    return f"${round_index}$"


__all__ = ["selection_round_marker", "selection_round_palette_index"]
