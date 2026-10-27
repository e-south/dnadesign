"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/core/envelopes.py

Resource envelopes shared by render contracts and integrations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputEnvelope:
    max_bytes: int
    max_records: int
    max_bases: int
    base_field_path: tuple[str, ...]
    accepted_input_kinds: tuple[str, ...]


__all__ = ["InputEnvelope"]
