"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/views/promoter_metadata_sequence.py

Sequence-derived promoter metadata for LatentDNA view rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..contracts.errors import ContractViolationError
from .promoter_metadata_common import coerce_list_of_dict_entries, normalize_text
from .promoter_metadata_stress import SIG35_PATTERN, is_control_row


def sig35_variant(row: dict[str, object]) -> str:
    plan = normalize_text(row.get("densegen__plan")) or ""
    match = SIG35_PATTERN.search(plan)
    if match is not None:
        return match.group(1).lower()
    annotated = (
        _sig35_variant_from_feature_detail(row)
        or _sig35_variant_from_seq_annot_features(row)
        or _sig35_variant_from_derived_retention(row)
    )
    if annotated is not None:
        return annotated
    if is_control_row(row):
        return "control"
    raise ContractViolationError(
        "sig35_variant could not be derived for a synthetic promoter row; expected densegen__plan to contain "
        "__sig35=, densegen__used_tfbs_detail to contain an upstream sigma70_core fixed element, "
        "or seq_annot__features to contain a Sigma-35 feature sequence"
    )


def _sig35_variant_from_feature_detail(row: dict[str, object]) -> str | None:
    for entry in used_tfbs_detail_entries(row.get("densegen__used_tfbs_detail")):
        if str(entry.get("part_kind") or "").strip().lower() != "fixed_element":
            continue
        if str(entry.get("role") or "").strip().lower() != "upstream":
            continue
        if str(entry.get("constraint_name") or "").strip().lower() != "sigma70_core":
            continue
        variant = normalize_text(entry.get("variant_id")) or normalize_text(entry.get("core_sequence"))
        if variant is None:
            variant = normalize_text(entry.get("sequence"))
        if variant is not None:
            return variant.lower() if len(variant) == 1 else variant.upper()
    return None


def used_tfbs_detail_entries(value: object) -> list[dict[str, object]]:
    return coerce_list_of_dict_entries(value, field_name="densegen__used_tfbs_detail")


def _sig35_variant_from_seq_annot_features(row: dict[str, object]) -> str | None:
    sequence = normalize_text(row.get("sequence"))
    bounds_are_current_sequence = _annotation_bounds_match_current_sequence(row, sequence=sequence)
    matches: set[str] = set()
    for feature in _seq_annot_feature_entries(row.get("seq_annot__features")):
        role_hint = str(feature.get("role_hint") or "").strip().lower()
        label = str(feature.get("label") or "").strip().lower()
        if role_hint != "sigma70_minus35" and label != "-35":
            continue
        feature_sequence = _feature_sequence_from_qualifiers(feature)
        if feature_sequence is None and sequence is not None and bounds_are_current_sequence:
            feature_sequence = _feature_sequence_from_bounds(feature, sequence=sequence)
        if feature_sequence is not None:
            matches.add(feature_sequence.upper())
    if len(matches) > 1:
        raise ContractViolationError(f"seq_annot__features contains multiple Sigma-35 feature sequences: {matches}")
    return next(iter(matches), None)


def _sig35_variant_from_derived_retention(row: dict[str, object]) -> str | None:
    sequence = normalize_text(row.get("sequence"))
    if sequence is None:
        return None
    target_length = row.get("derived__target_length")
    if target_length is not None:
        try:
            if int(target_length) != len(sequence):
                return None
        except (TypeError, ValueError):
            return None
    matches: set[str] = set()
    for feature in _generic_feature_entries(
        row.get("derived__features_retained"),
        field_name="derived__features_retained",
    ):
        role_hint = str(feature.get("role_hint") or "").strip().lower()
        label = str(feature.get("label") or "").strip().lower()
        if role_hint != "sigma70_minus35" and label != "-35":
            continue
        for interval in _generic_feature_entries(
            feature.get("derived_intervals_0"),
            field_name="derived_intervals_0",
        ):
            feature_sequence = _feature_sequence_from_bounds(interval, sequence=sequence)
            if feature_sequence is not None:
                matches.add(feature_sequence.upper())
    if len(matches) > 1:
        raise ContractViolationError(
            f"derived__features_retained contains multiple Sigma-35 feature sequences: {matches}"
        )
    return next(iter(matches), None)


def _annotation_bounds_match_current_sequence(row: dict[str, object], *, sequence: str | None) -> bool:
    if sequence is None:
        return False
    start = row.get("seq_annot__sequence_region_start_0")
    end = row.get("seq_annot__sequence_region_end_0")
    if start is None or end is None:
        return False
    try:
        return int(start) == 0 and int(end) == len(sequence)
    except (TypeError, ValueError):
        return False


def _seq_annot_feature_entries(value: object) -> list[dict[str, object]]:
    return _generic_feature_entries(value, field_name="seq_annot__features")


def _generic_feature_entries(value: object, *, field_name: str) -> list[dict[str, object]]:
    return coerce_list_of_dict_entries(value, field_name=field_name)


def _feature_sequence_from_qualifiers(feature: dict[str, object]) -> str | None:
    qualifiers = feature.get("qualifiers")
    if qualifiers is None:
        return None
    if hasattr(qualifiers, "as_py"):
        qualifiers = qualifiers.as_py()
    if not isinstance(qualifiers, list) and hasattr(qualifiers, "tolist"):
        converted = qualifiers.tolist()
        if isinstance(converted, list):
            qualifiers = converted
    if not isinstance(qualifiers, list):
        raise ContractViolationError("seq_annot__features qualifiers must decode to a list")
    for qualifier in qualifiers:
        if hasattr(qualifier, "as_py"):
            qualifier = qualifier.as_py()
        if not isinstance(qualifier, dict):
            raise ContractViolationError("seq_annot__features qualifier entries must be dictionaries")
        key = str(qualifier.get("key") or "").strip().lower()
        value = normalize_text(qualifier.get("value"))
        if value is None:
            continue
        if key == "feature_sequence":
            return value
        if key != "note":
            continue
        for token in value.replace(";", " ").split():
            if token.startswith("feature_sequence="):
                return token.split("=", 1)[1]
    return None


def _feature_sequence_from_bounds(feature: dict[str, object], *, sequence: str) -> str | None:
    start = feature.get("start_0")
    end = feature.get("end_0")
    if start is None or end is None:
        return None
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return None
    if start_i < 0 or end_i <= start_i or end_i > len(sequence):
        return None
    return sequence[start_i:end_i]


def spacer_length(row: dict[str, object]) -> int | None:
    if is_control_row(row):
        return None
    detail_entries = used_tfbs_detail_entries(row.get("densegen__used_tfbs_detail"))
    if not detail_entries:
        return None
    spacer_values: set[int] = set()
    for entry in detail_entries:
        if str(entry.get("part_kind") or "").strip().lower() != "fixed_element":
            continue
        spacer_raw = entry.get("spacer_length")
        if spacer_raw is None:
            continue
        try:
            spacer_values.add(int(spacer_raw))
        except (TypeError, ValueError) as exc:
            raise ContractViolationError("spacer_length metadata must be integer-valued") from exc
    if not spacer_values:
        raise ContractViolationError(
            "spacer_length could not be derived for a synthetic promoter row; expected densegen__used_tfbs_detail"
        )
    if len(spacer_values) != 1:
        raise ContractViolationError(
            f"spacer_length derivation expected one realized spacer length, found {sorted(spacer_values)}"
        )
    return next(iter(spacer_values))
