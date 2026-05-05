"""
Promoter-family metadata derivations for LatentDNA view rows.
"""

from __future__ import annotations

import json
import re

from ..contracts.errors import ContractViolationError
from ..contracts.promoter_metadata import REGULONDB_NATIVE_PROMOTER_METADATA_COLUMNS

_SIG35_PATTERN = re.compile(r"__sig35[=_]([A-Za-z0-9]+)")
_CONTROL_LABELS = {"spyp", "sulap", "soxsp", "j23105", "spy_p", "sul_ap", "sox_sp"}


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_regulator_name(value: object) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    token = text.split("_", 1)[0].strip()
    if not token:
        return None
    return {
        "baer": "baeR",
        "background": "background",
        "background_only": "background",
        "cpxr": "cpxR",
        "control": "control",
        "lexa": "lexA",
    }.get(token.lower(), token)


def _normalized_regulators(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        values = value
    else:
        values = [value]
    normalized = sorted(
        {_canonical_regulator_name(item) for item in values if _canonical_regulator_name(item) is not None},
        key=str.casefold,
    )
    return [str(item) for item in normalized]


def _is_control_row(row: dict[str, object]) -> bool:
    label = (_normalize_text(row.get("usr_label__primary")) or "").lower()
    template_id = (_construct_template_id(row) or "").lower()
    plan = _normalize_text(row.get("densegen__plan"))
    if template_id in {"wt", "wildtype", "manual"}:
        return True
    if label in _CONTROL_LABELS:
        return True
    return plan is None


def _construct_template_id(row: dict[str, object]) -> str | None:
    return (
        _normalize_text(row.get("construct_template_id"))
        or _normalize_text(row.get("template_id"))
        or _normalize_text(row.get("construct__template_id"))
    )


def _design_family(row: dict[str, object]) -> str:
    plan = _normalize_text(row.get("densegen__plan"))
    if plan is not None:
        if plan.startswith("background_only"):
            return "background_only"
        if plan.startswith("ethanol_ciprofloxacin"):
            return "ethanol_ciprofloxacin"
        if plan.startswith("ethanol"):
            return "ethanol"
        if plan.startswith("ciprofloxacin"):
            return "ciprofloxacin"
    if _is_control_row(row):
        return "control"
    return "control"


def _design_regulator_composition(row: dict[str, object]) -> str:
    if _is_control_row(row):
        return "control"
    family = _design_family(row)
    regulators = _normalized_regulators(row.get("densegen__required_regulators"))
    if family == "background_only" and not regulators:
        return "background"
    if regulators:
        return regulators[0] if len(regulators) == 1 else "+".join(regulators)

    plan = _normalize_text(row.get("densegen__plan")) or ""
    tokens = [token for token in plan.split("__") if token]
    if len(tokens) >= 2 and not tokens[1].startswith("sigma70_"):
        composition_parts = [_canonical_regulator_name(token) for token in tokens[1].replace("_", "+").split("+")]
        composition_parts = sorted(
            {
                part
                for part in composition_parts
                if part not in {None, "control"} and not str(part).startswith("sig35=")
            },
            key=str.casefold,
        )
        if composition_parts:
            return composition_parts[0] if len(composition_parts) == 1 else "+".join(composition_parts)
    if family == "background_only":
        return "background"
    return "unknown"


def _sig35_variant(row: dict[str, object]) -> str:
    plan = _normalize_text(row.get("densegen__plan")) or ""
    match = _SIG35_PATTERN.search(plan)
    if match is not None:
        return match.group(1).lower()
    annotated = (
        _sig35_variant_from_feature_detail(row)
        or _sig35_variant_from_seq_annot_features(row)
        or _sig35_variant_from_derived_retention(row)
    )
    if annotated is not None:
        return annotated
    if _is_control_row(row):
        return "control"
    raise ContractViolationError(
        "sig35_variant could not be derived for a synthetic promoter row; expected densegen__plan to contain "
        "__sig35=, densegen__used_tfbs_detail to contain an upstream sigma70_core fixed element, "
        "or seq_annot__features to contain a Sigma-35 feature sequence"
    )


def _sig35_variant_from_feature_detail(row: dict[str, object]) -> str | None:
    for entry in _used_tfbs_detail_entries(row.get("densegen__used_tfbs_detail")):
        if str(entry.get("part_kind") or "").strip().lower() != "fixed_element":
            continue
        if str(entry.get("role") or "").strip().lower() != "upstream":
            continue
        if str(entry.get("constraint_name") or "").strip().lower() != "sigma70_core":
            continue
        variant = _normalize_text(entry.get("variant_id")) or _normalize_text(entry.get("core_sequence"))
        if variant is None:
            variant = _normalize_text(entry.get("sequence"))
        if variant is not None:
            return variant.lower() if len(variant) == 1 else variant.upper()
    return None


def _used_tfbs_detail_entries(value: object) -> list[dict[str, object]]:
    return _coerce_list_of_dict_entries(value, field_name="densegen__used_tfbs_detail")


def _coerce_list_of_dict_entries(value: object, *, field_name: str) -> list[dict[str, object]]:
    if value is None:
        return []
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - malformed payloads are caught by callers
            raise ContractViolationError(f"{field_name} must be valid JSON when encoded as text") from exc
    if not isinstance(value, list) and hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            value = converted
    if not isinstance(value, list):
        raise ContractViolationError(f"{field_name} must decode to a list of dict entries")
    entries: list[dict[str, object]] = []
    for item in value:
        if hasattr(item, "as_py"):
            item = item.as_py()
        if not isinstance(item, dict):
            raise ContractViolationError(f"{field_name} entries must be dictionaries")
        entries.append(dict(item))
    return entries


def _sig35_variant_from_seq_annot_features(row: dict[str, object]) -> str | None:
    sequence = _normalize_text(row.get("sequence"))
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
    sequence = _normalize_text(row.get("sequence"))
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
    return _coerce_list_of_dict_entries(value, field_name=field_name)


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
        value = _normalize_text(qualifier.get("value"))
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


def _spacer_length(row: dict[str, object]) -> int | None:
    if _is_control_row(row):
        return None
    detail_entries = _used_tfbs_detail_entries(row.get("densegen__used_tfbs_detail"))
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


def _campaign_prior(row: dict[str, object]) -> str:
    family = _design_family(row)
    return {
        "background_only": "background",
        "ethanol": "ethanol",
        "ciprofloxacin": "cipro",
        "ethanol_ciprofloxacin": "and",
        "control": "control",
    }.get(family, "control")


def _source_class(row: dict[str, object]) -> str:
    if _normalize_text(row.get("densegen__plan")) is not None:
        return "densegen"
    source_family = _normalize_text(row.get("source_family"))
    if source_family is not None:
        normalized = source_family.lower()
        if "densegen" in normalized:
            return "densegen"
        if "reference" in normalized or "genbank" in normalized or "standard" in normalized:
            return "reference_control"
        return normalized
    if _normalize_text(row.get("promoter_standard__collection_id")) is not None:
        return "synthetic_reference_standard"
    return "manual_or_wildtype" if _is_control_row(row) else "densegen"


def derive_promoter_metadata_value(row: dict[str, object], *, derive: str) -> object:
    if derive in REGULONDB_NATIVE_PROMOTER_METADATA_COLUMNS:
        if derive not in row:
            raise ContractViolationError(f"native RegulonDB promoter metadata column is missing: {derive}")
        return row[derive]
    if derive == "design_family":
        return _design_family(row)
    if derive == "design_regulator_composition":
        return _design_regulator_composition(row)
    if derive == "sig35_variant":
        return _sig35_variant(row)
    if derive == "spacer_length":
        return _spacer_length(row)
    if derive == "campaign_prior":
        return _campaign_prior(row)
    if derive == "is_control":
        return _is_control_row(row)
    if derive == "source_class":
        return _source_class(row)
    raise ContractViolationError(f"unsupported promoter metadata derivation: {derive}")
