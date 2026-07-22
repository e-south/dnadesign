"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/benchling_ids.py

Benchling reviewer-id parsing for Retron hairpin review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Mapping

from ...compiler.exceptions import RetronMsdCompilerError

BENCHLING_VARIANT_ID_RE = re.compile(r"^r\d+-w\d{2}-\d{2}$")
ASSIGNED_CONSTRUCT_ID_RE = re.compile(r"^pES-retron-\d+$")


def parse_assigned_retron_ids(raw: Mapping[str, object]) -> dict[str, str]:
    assigned: dict[str, str] = {}
    observed_construct_ids: set[str] = set()
    for raw_variant_id, raw_construct_id in raw.items():
        variant_id = str(raw_variant_id).strip()
        construct_id = str(raw_construct_id).strip()
        if BENCHLING_VARIANT_ID_RE.match(variant_id) is None:
            raise RetronMsdCompilerError(f"Retron Benchling variant id is not compact reviewer form: {variant_id}")
        if ASSIGNED_CONSTRUCT_ID_RE.match(construct_id) is None:
            raise RetronMsdCompilerError(f"Retron Benchling assigned construct id is invalid: {construct_id}")
        if construct_id in observed_construct_ids:
            raise RetronMsdCompilerError(f"Retron Benchling assigned construct id is duplicated: {construct_id}")
        assigned[variant_id] = construct_id
        observed_construct_ids.add(construct_id)
    if not assigned:
        raise RetronMsdCompilerError("Retron Benchling GenBank import assigned_retron_ids cannot be empty")
    return assigned


def parse_source_precedent_ids(raw: Mapping[str, object]) -> dict[str, str]:
    precedents: dict[str, str] = {}
    for raw_variant_id, raw_construct_id in raw.items():
        variant_id = str(raw_variant_id).strip()
        construct_id = str(raw_construct_id).strip()
        if BENCHLING_VARIANT_ID_RE.match(variant_id) is None:
            raise RetronMsdCompilerError(f"Retron Benchling precedent variant id is invalid: {variant_id}")
        if ASSIGNED_CONSTRUCT_ID_RE.match(construct_id) is None:
            raise RetronMsdCompilerError(f"Retron Benchling source precedent id is invalid: {construct_id}")
        precedents[variant_id] = construct_id
    if not precedents:
        raise RetronMsdCompilerError("Retron Benchling GenBank import source_precedent_ids cannot be empty")
    return precedents


__all__ = [
    "ASSIGNED_CONSTRUCT_ID_RE",
    "BENCHLING_VARIANT_ID_RE",
    "parse_assigned_retron_ids",
    "parse_source_precedent_ids",
]
