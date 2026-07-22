"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/review_variant_ids.py

Review-frame retron-id contract parsing for Retron hairpin review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Mapping

from ...compiler.exceptions import RetronMsdCompilerError
from .benchling_ids import ASSIGNED_CONSTRUCT_ID_RE, BENCHLING_VARIANT_ID_RE
from .benchling_import import BenchlingGenbankImportPlan

CONSTRUCT_PREFIXES = ("pES-tetr-", "pES-teto-")


def parse_review_variant_ids(
    families: Mapping[str, object],
    *,
    design_set: Mapping[str, object],
    benchling_import: BenchlingGenbankImportPlan,
) -> dict[str, str]:
    stills_plan = _require_mapping(families.get("msd_sequence_review_stills"), "msd_sequence_review_stills")
    review_ids = _parse_review_id_mapping(_require_mapping(stills_plan.get("review_variant_ids"), "review_variant_ids"))
    expected_variant_ids = _expected_design_variant_ids(design_set)
    if tuple(review_ids) != expected_variant_ids:
        raise RetronMsdCompilerError(
            f"Retron review_variant_ids must match design-set order: {list(review_ids)} != {list(expected_variant_ids)}"
        )
    for variant_id, assigned_id in benchling_import.assigned_retron_ids.items():
        if review_ids.get(variant_id) != assigned_id:
            raise RetronMsdCompilerError(
                "Retron review_variant_ids must match Benchling assigned_retron_ids for imported trims: "
                f"{variant_id}={review_ids.get(variant_id)!r} != {assigned_id!r}"
            )
    return review_ids


def _parse_review_id_mapping(raw: Mapping[str, object]) -> dict[str, str]:
    review_ids: dict[str, str] = {}
    observed_construct_ids: set[str] = set()
    for raw_variant_id, raw_construct_id in raw.items():
        variant_id = str(raw_variant_id).strip()
        construct_id = str(raw_construct_id).strip()
        if BENCHLING_VARIANT_ID_RE.match(variant_id) is None:
            raise RetronMsdCompilerError(f"Retron review variant id is not compact reviewer form: {variant_id}")
        if ASSIGNED_CONSTRUCT_ID_RE.match(construct_id) is None:
            raise RetronMsdCompilerError(f"Retron review construct id is invalid: {construct_id}")
        if construct_id in observed_construct_ids:
            raise RetronMsdCompilerError(f"Retron review construct id is duplicated: {construct_id}")
        review_ids[variant_id] = construct_id
        observed_construct_ids.add(construct_id)
    if not review_ids:
        raise RetronMsdCompilerError("Retron review_variant_ids cannot be empty")
    return review_ids


def _expected_design_variant_ids(design_set: Mapping[str, object]) -> tuple[str, ...]:
    designs = design_set.get("designs")
    if not isinstance(designs, list):
        raise RetronMsdCompilerError("Retron design set must declare designs for review_variant_ids validation")
    return tuple(_variant_id_from_design(_require_mapping(design, "design-set design")) for design in designs)


def _variant_id_from_design(design: Mapping[str, object]) -> str:
    construct_id = str(design.get("construct_id") or "").strip()
    prefix = next((candidate for candidate in CONSTRUCT_PREFIXES if construct_id.startswith(candidate)), None)
    if prefix is None:
        raise RetronMsdCompilerError(
            f"Retron design construct_id must start with one of {CONSTRUCT_PREFIXES}: {construct_id}"
        )
    variant_id = construct_id.removeprefix(prefix)
    if BENCHLING_VARIANT_ID_RE.match(variant_id) is None:
        raise RetronMsdCompilerError(f"Retron design construct_id does not encode compact review id: {construct_id}")
    return variant_id


def _require_mapping(raw: object, label: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise RetronMsdCompilerError(f"Retron review output expected mapping for {label}")
    return raw


__all__ = ["parse_review_variant_ids"]
