"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_json.py

Strict JSON loading for behavior shadow receipts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_strict_behavior_json(path: Path) -> dict[str, object]:
    """Reject invalid, duplicate-key, non-finite, or non-mapping JSON."""

    if not path.is_file():
        raise FileNotFoundError(path)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"multistate behavior JSON contains duplicate key {key!r}.")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON value {value!r} is prohibited.")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number {value!r} is prohibited.")
        return parsed

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"multistate behavior JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON record must be a mapping: {path}")
    return payload


__all__ = ["load_strict_behavior_json"]
