"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/realization/parts.py

Realized construct-part records shared across realization contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RealizedPart:
    name: str
    role: str
    kind: str
    sequence_source: str
    sequence_field: str | None
    orientation: str
    start: int
    end: int
    sequence: str
    realized_start: int
    realized_end: int
