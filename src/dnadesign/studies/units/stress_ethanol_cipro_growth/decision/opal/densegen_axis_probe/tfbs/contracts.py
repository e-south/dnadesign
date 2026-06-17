"""DenseGen TFBS learnability parser and row-level contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .schema import TFBS_LEARNABILITY_LABEL_RECIPE_HASH, TFBS_LEARNABILITY_ORACLE_VERSION


@dataclass(frozen=True)
class ParsedTfbsEntry:
    """One active DenseGen TFBS entry in final-sequence coordinates."""

    family: Literal["LexA", "CpxR", "BaeR", "background"]
    offset_raw: int
    length: int
    end_raw: int


@dataclass(frozen=True)
class ParsedFixedElement:
    """One passive sigma-core fixed element in final-sequence coordinates."""

    role: Literal["sigma35", "sigma10"]
    offset_raw: int
    length: int
    end_raw: int
    variant_id: str | None
    consensus_identity: str | None
    spacer_length: int


@dataclass(frozen=True)
class ParsedDenseGenTfbsRow:
    """Validated DenseGen construction metadata for one OPAL candidate."""

    candidate_id: str
    sequence: str
    tfbs_entries: tuple[ParsedTfbsEntry, ParsedTfbsEntry, ParsedTfbsEntry]
    sigma35: ParsedFixedElement
    sigma10: ParsedFixedElement

    @property
    def slot_families(self) -> tuple[str, str, str]:
        return tuple(entry.family for entry in self.tfbs_entries)

    def to_label_row(self) -> dict[str, Any]:
        counts = _family_counts(self.tfbs_entries)
        cpxr_or_baer = counts["CpxR"] + counts["BaeR"]
        slot0, slot1, slot2 = self.slot_families
        return {
            "id": self.candidate_id,
            "quality_flag": "ok",
            "lexA_count": counts["LexA"],
            "cpxR_count": counts["CpxR"],
            "baeR_count": counts["BaeR"],
            "cpxR_or_baeR_count": cpxr_or_baer,
            "lexA_present": int(counts["LexA"] > 0),
            "cpxR_present": int(counts["CpxR"] > 0),
            "baeR_present": int(counts["BaeR"] > 0),
            "cpxR_or_baeR_present": int(cpxr_or_baer > 0),
            "lexA_count_fraction": counts["LexA"] / 3.0,
            "cpxR_count_fraction": counts["CpxR"] / 3.0,
            "baeR_count_fraction": counts["BaeR"] / 3.0,
            "cpxR_or_baeR_count_fraction": cpxr_or_baer / 3.0,
            "lexA_in_slot0": int(slot0 == "LexA"),
            "lexA_in_slot1": int(slot1 == "LexA"),
            "lexA_in_slot2": int(slot2 == "LexA"),
            "baeR_in_slot1": int(slot1 == "BaeR"),
            "cpxR_or_baeR_in_slot0": int(slot0 in {"CpxR", "BaeR"}),
            "cpxR_or_baeR_in_slot1": int(slot1 in {"CpxR", "BaeR"}),
            "cpxR_or_baeR_in_slot2": int(slot2 in {"CpxR", "BaeR"}),
            "slot0_family": slot0,
            "slot1_family": slot1,
            "slot2_family": slot2,
            "sigma35_variant": self.sigma35.variant_id,
            "sigma10_consensus_identity": self.sigma10.consensus_identity,
            "spacer_length": self.sigma35.spacer_length,
            "sigma35_offset_raw": self.sigma35.offset_raw,
            "sigma10_offset_raw": self.sigma10.offset_raw,
            "sigma35_end_raw": self.sigma35.end_raw,
            "sigma10_end_raw": self.sigma10.end_raw,
            "oracle_version": TFBS_LEARNABILITY_ORACLE_VERSION,
            "label_recipe_hash": TFBS_LEARNABILITY_LABEL_RECIPE_HASH,
        }


def parse_densegen_tfbs_row(row: Mapping[str, Any]) -> ParsedDenseGenTfbsRow:
    """Parse and validate one DenseGen TFBS learnability source row."""

    candidate_id = _required_text(row.get("id"), field="id")
    sequence = _required_text(row.get("sequence"), field="sequence")
    if len(sequence) != 60:
        raise ValueError(f"{candidate_id}: candidate sequence length must be exactly 60 bp, got {len(sequence)}")

    entries = _normalize_detail_entries(row.get("densegen__used_tfbs_detail"))
    tfbs_entries: list[ParsedTfbsEntry] = []
    fixed_elements: list[ParsedFixedElement] = []
    for entry in entries:
        part_kind = _required_text(entry.get("part_kind"), field="part_kind").lower()
        if part_kind == "tfbs":
            tfbs_entries.append(_parse_tfbs_entry(candidate_id, entry))
        elif part_kind == "fixed_element":
            fixed_elements.append(_parse_fixed_element(candidate_id, entry))
        else:
            raise ValueError(f"{candidate_id}: unsupported DenseGen part_kind: {part_kind}")

    if len(tfbs_entries) != 3:
        raise ValueError(f"{candidate_id}: expected exactly 3 tfbs entries, got {len(tfbs_entries)}")
    if len(fixed_elements) != 2:
        raise ValueError(f"{candidate_id}: expected exactly 2 fixed_element entries, got {len(fixed_elements)}")

    offsets = [entry.offset_raw for entry in tfbs_entries]
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"{candidate_id}: ambiguous slot order from tied offset_raw values: {offsets}")
    slots = tuple(sorted(tfbs_entries, key=lambda entry: entry.offset_raw))
    fixed_by_role = _fixed_elements_by_role(candidate_id, fixed_elements)
    sigma35 = fixed_by_role["sigma35"]
    sigma10 = fixed_by_role["sigma10"]
    _validate_sigma_core(candidate_id, sigma35=sigma35, sigma10=sigma10)
    return ParsedDenseGenTfbsRow(
        candidate_id=candidate_id,
        sequence=sequence,
        tfbs_entries=(slots[0], slots[1], slots[2]),
        sigma35=sigma35,
        sigma10=sigma10,
    )


def normalize_tf_family(value: Any) -> Literal["LexA", "CpxR", "BaeR", "background"]:
    """Normalize configured DenseGen TFBS regulator text into the v1 family ontology."""

    text = _clean_text(value).lower()
    if "lexa" in text:
        return "LexA"
    if "cpxr" in text:
        return "CpxR"
    if "baer" in text:
        return "BaeR"
    if text in {"", "background", "bg", "none", "null", "non_target", "nontarget"} or "background" in text:
        return "background"
    raise ValueError(f"unknown TFBS regulator family: {value!r}")


def _parse_tfbs_entry(candidate_id: str, entry: Mapping[str, Any]) -> ParsedTfbsEntry:
    if "offset_raw" not in entry and "offset" in entry:
        raise ValueError(f"{candidate_id}: offset_raw is required for active TFBS slots; offset must not be used")
    offset_raw = _required_int(entry.get("offset_raw"), field="offset_raw", candidate_id=candidate_id)
    length = _required_int(entry.get("length"), field="length", candidate_id=candidate_id)
    end_raw = _validated_end_raw(candidate_id, entry=entry, offset_raw=offset_raw, length=length)
    _validate_coordinate(candidate_id, offset_raw=offset_raw, end_raw=end_raw)
    family = normalize_tf_family(_first_present(entry, ("regulator", "family", "tf_family", "regulator_family")))
    return ParsedTfbsEntry(family=family, offset_raw=offset_raw, length=length, end_raw=end_raw)


def _parse_fixed_element(candidate_id: str, entry: Mapping[str, Any]) -> ParsedFixedElement:
    offset_raw = _required_int(entry.get("offset_raw"), field="offset_raw", candidate_id=candidate_id)
    length = _required_int(entry.get("length"), field="length", candidate_id=candidate_id)
    end_raw = _validated_end_raw(candidate_id, entry=entry, offset_raw=offset_raw, length=length)
    _validate_coordinate(candidate_id, offset_raw=offset_raw, end_raw=end_raw)
    role = _fixed_role(candidate_id, entry)
    spacer_length = _required_int(
        _first_present(entry, ("spacer_length", "spacer")),
        field="spacer_length",
        candidate_id=candidate_id,
    )
    if spacer_length not in {16, 17, 18, 19, 20}:
        raise ValueError(f"{candidate_id}: spacer_length must be one of 16, 17, 18, 19, 20; got {spacer_length}")
    variant_id = _optional_text(_first_present(entry, ("variant_id", "variant", "sigma35_variant")))
    consensus_identity = _optional_text(
        _first_present(entry, ("consensus_identity", "sigma10_consensus_identity", "consensus", "sequence"))
    )
    return ParsedFixedElement(
        role=role,
        offset_raw=offset_raw,
        length=length,
        end_raw=end_raw,
        variant_id=variant_id,
        consensus_identity=consensus_identity,
        spacer_length=spacer_length,
    )


def _fixed_role(candidate_id: str, entry: Mapping[str, Any]) -> Literal["sigma35", "sigma10"]:
    role_text = " ".join(
        _clean_text(_first_present(entry, keys))
        for keys in (("role",), ("fixed_role",), ("element_role",), ("name", "label", "part_id"))
    ).lower()
    if any(token in role_text for token in ("sigma35", "sig35", "-35", "upstream")):
        return "sigma35"
    if any(token in role_text for token in ("sigma10", "sig10", "-10", "downstream")):
        return "sigma10"
    raise ValueError(f"{candidate_id}: fixed_element role must map to exactly one sigma35 or sigma10 element")


def _fixed_elements_by_role(
    candidate_id: str, fixed_elements: Sequence[ParsedFixedElement]
) -> dict[Literal["sigma35", "sigma10"], ParsedFixedElement]:
    by_role: dict[Literal["sigma35", "sigma10"], ParsedFixedElement] = {}
    for element in fixed_elements:
        if element.role in by_role:
            raise ValueError(f"{candidate_id}: duplicate fixed_element role: {element.role}")
        by_role[element.role] = element
    missing = sorted({"sigma35", "sigma10"} - set(by_role))
    if missing:
        raise ValueError(f"{candidate_id}: missing fixed_element role(s): {missing}")
    return by_role


def _validate_sigma_core(candidate_id: str, *, sigma35: ParsedFixedElement, sigma10: ParsedFixedElement) -> None:
    if sigma35.variant_id is None:
        raise ValueError(f"{candidate_id}: sigma35 fixed element missing variant_id")
    if sigma10.consensus_identity is None:
        raise ValueError(f"{candidate_id}: sigma10 fixed element missing consensus identity")
    if sigma35.spacer_length != sigma10.spacer_length:
        raise ValueError(f"{candidate_id}: sigma35/sigma10 spacer_length values disagree")
    if not 0 <= sigma35.offset_raw <= 32:
        raise ValueError(f"{candidate_id}: sigma35 offset_raw outside expected 0-32 span")
    if not 22 <= sigma10.offset_raw <= 54:
        raise ValueError(f"{candidate_id}: sigma10 offset_raw outside expected 22-54 span")
    expected_delta = 6 + sigma35.spacer_length
    observed_delta = sigma10.offset_raw - sigma35.offset_raw
    if observed_delta != expected_delta:
        raise ValueError(
            f"{candidate_id}: invalid sigma-core spacer relationship; "
            f"sigma10_offset_raw - sigma35_offset_raw = {observed_delta}, expected {expected_delta}"
        )


def _normalize_detail_entries(value: Any) -> list[Mapping[str, Any]]:
    if _is_missing(value):
        raise ValueError("missing densegen__used_tfbs_detail")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"densegen__used_tfbs_detail is not valid JSON: {exc}") from exc
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"densegen__used_tfbs_detail must be a list, got {type(value).__name__}")
    entries: list[Mapping[str, Any]] = []
    for item in value:
        if hasattr(item, "as_py"):
            item = item.as_py()
        if not isinstance(item, Mapping):
            raise ValueError("densegen__used_tfbs_detail entries must be mappings")
        entries.append(item)
    return entries


def _family_counts(entries: Sequence[ParsedTfbsEntry]) -> dict[str, int]:
    counts = {"LexA": 0, "CpxR": 0, "BaeR": 0, "background": 0}
    for entry in entries:
        counts[entry.family] += 1
    return counts


def _validated_end_raw(candidate_id: str, *, entry: Mapping[str, Any], offset_raw: int, length: int) -> int:
    if length <= 0:
        raise ValueError(f"{candidate_id}: length must be > 0")
    computed = offset_raw + length
    if "end_raw" in entry and not _is_missing(entry.get("end_raw")):
        observed = _required_int(entry.get("end_raw"), field="end_raw", candidate_id=candidate_id)
        if observed != computed:
            raise ValueError(f"{candidate_id}: end_raw {observed} does not equal offset_raw + length {computed}")
    return computed


def _validate_coordinate(candidate_id: str, *, offset_raw: int, end_raw: int) -> None:
    if not 0 <= offset_raw < 60:
        raise ValueError(f"{candidate_id}: offset_raw outside final 60 bp sequence: {offset_raw}")
    if not 0 < end_raw <= 60:
        raise ValueError(f"{candidate_id}: end_raw outside final 60 bp sequence: {end_raw}")


def _required_int(value: Any, *, field: str, candidate_id: str) -> int:
    if _is_missing(value):
        raise ValueError(f"{candidate_id}: missing required field {field}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{candidate_id}: field {field} must be an integer, got {value!r}") from exc
    if float(number) != float(value):
        raise ValueError(f"{candidate_id}: field {field} must be an integer, got {value!r}")
    return number


def _first_present(entry: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in entry and not _is_missing(entry.get(key)):
            return entry.get(key)
    return None


def _required_text(value: Any, *, field: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"missing required field {field}")
    return text


def _optional_text(value: Any) -> str | None:
    text = _clean_text(value)
    return text or None


def _clean_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if value is pd.NA:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False
