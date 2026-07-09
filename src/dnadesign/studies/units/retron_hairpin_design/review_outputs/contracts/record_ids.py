"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/record_ids.py

MSD-only record-id parsing for Retron review-output contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Mapping

from ...compiler.exceptions import RetronMsdCompilerError

MSD_RECORD_ID_RE = re.compile(r"^msd-retron-[0-9]+$")


def parse_record_ids(raw: object) -> Mapping[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise RetronMsdCompilerError("Retron Benchling GenBank import expected non-empty mapping for record_ids")
    parsed: dict[str, str] = {}
    observed = set()
    for key, value in raw.items():
        variant_id = _string_value(key, label="record_ids key")
        record_id = _string_value(value, label=f"record_ids.{variant_id}")
        if MSD_RECORD_ID_RE.match(record_id) is None:
            raise RetronMsdCompilerError(
                f"Retron Benchling GenBank record_ids.{variant_id} must look like msd-retron-201: {record_id}"
            )
        if record_id in observed:
            raise RetronMsdCompilerError(f"Retron Benchling GenBank record id is duplicated: {record_id}")
        parsed[variant_id] = record_id
        observed.add(record_id)
    return parsed


def _string_value(raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip() or "\n" in raw:
        raise RetronMsdCompilerError(f"Retron Benchling GenBank import has invalid value for {label}")
    return raw.strip()


__all__ = ["parse_record_ids"]
