"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_learnability_parser.py

Regression tests for TFBS learnability parser studies units stress ethanol cipro.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from .probe_modules import probe_module

_contracts = probe_module("tfbs.contracts")
normalize_tf_family = _contracts.normalize_tf_family
parse_densegen_tfbs_row = _contracts.parse_densegen_tfbs_row

SEQ60 = "A" * 60


def test_parse_densegen_tfbs_row_uses_offset_raw_slots_and_sigma_core_controls() -> None:
    parsed = parse_densegen_tfbs_row({"id": "cand-1", "sequence": SEQ60, "densegen__used_tfbs_detail": _detail()})

    assert parsed.slot_families == ("LexA", "BaeR", "background")
    assert parsed.sigma35.variant_id == "f"
    assert parsed.sigma10.consensus_identity == "TATAAT"

    labels = parsed.to_label_row()
    assert labels["lexA_count"] == 1
    assert labels["cpxR_count"] == 0
    assert labels["baeR_count"] == 1
    assert labels["cpxR_or_baeR_count"] == 1
    assert labels["lexA_present"] == 1
    assert labels["cpxR_present"] == 0
    assert labels["lexA_count_fraction"] == pytest.approx(1 / 3)
    assert labels["lexA_in_slot0"] == 1
    assert labels["baeR_in_slot1"] == 1
    assert labels["cpxR_or_baeR_in_slot1"] == 1
    assert labels["cpxR_or_baeR_in_slot2"] == 0
    assert labels["sigma35_offset_raw"] == 0
    assert labels["sigma10_offset_raw"] == 22


def test_normalize_tf_family_rejects_unknown_nonempty_regulator() -> None:
    assert normalize_tf_family("lexA_CTGTATAWAWWHACA") == "LexA"
    assert normalize_tf_family("background") == "background"

    with pytest.raises(ValueError, match="unknown TFBS regulator family"):
        normalize_tf_family("surpriseRegulator")


def test_parse_densegen_tfbs_row_rejects_offset_without_offset_raw() -> None:
    detail = _detail()
    detail[0].pop("offset_raw")
    detail[0]["offset"] = 10

    with pytest.raises(ValueError, match="offset_raw is required.*offset must not be used"):
        parse_densegen_tfbs_row({"id": "cand-1", "sequence": SEQ60, "densegen__used_tfbs_detail": detail})


def test_parse_densegen_tfbs_row_rejects_wrong_sequence_length() -> None:
    with pytest.raises(ValueError, match="sequence length must be exactly 60"):
        parse_densegen_tfbs_row({"id": "cand-1", "sequence": "A" * 59, "densegen__used_tfbs_detail": _detail()})


def test_parse_densegen_tfbs_row_rejects_wrong_tfbs_count() -> None:
    detail = [entry for entry in _detail() if entry.get("part_kind") != "tfbs"]

    with pytest.raises(ValueError, match="expected exactly 3 tfbs entries"):
        parse_densegen_tfbs_row({"id": "cand-1", "sequence": SEQ60, "densegen__used_tfbs_detail": detail})


def test_parse_densegen_tfbs_row_rejects_ambiguous_slot_order() -> None:
    detail = _detail()
    detail[1]["offset_raw"] = detail[0]["offset_raw"]
    detail[1]["end_raw"] = int(detail[1]["offset_raw"]) + int(detail[1]["length"])

    with pytest.raises(ValueError, match="ambiguous slot order"):
        parse_densegen_tfbs_row({"id": "cand-1", "sequence": SEQ60, "densegen__used_tfbs_detail": detail})


def test_parse_densegen_tfbs_row_rejects_invalid_coordinate_range() -> None:
    detail = _detail()
    detail[0]["offset_raw"] = 59
    detail[0]["length"] = 2
    detail[0]["end_raw"] = 61

    with pytest.raises(ValueError, match="end_raw outside final 60 bp"):
        parse_densegen_tfbs_row({"id": "cand-1", "sequence": SEQ60, "densegen__used_tfbs_detail": detail})


def test_parse_densegen_tfbs_row_rejects_fixed_element_role_and_spacer_errors() -> None:
    detail = _detail()
    detail[3]["role"] = "ambiguous_fixed"

    with pytest.raises(ValueError, match="fixed_element role"):
        parse_densegen_tfbs_row({"id": "cand-1", "sequence": SEQ60, "densegen__used_tfbs_detail": detail})

    detail = _detail()
    detail[4]["offset_raw"] = 23
    detail[4]["end_raw"] = 29

    with pytest.raises(ValueError, match="invalid sigma-core spacer relationship"):
        parse_densegen_tfbs_row({"id": "cand-1", "sequence": SEQ60, "densegen__used_tfbs_detail": detail})


def _detail() -> list[dict[str, object]]:
    return [
        _tfbs("LexA", 10),
        _tfbs("background", 32),
        _tfbs("BaeR", 21),
        _fixed("upstream_sigma70_core", 0, variant_id="f"),
        _fixed("downstream_sigma70_core", 22, sequence="TATAAT"),
    ]


def _tfbs(regulator: str, offset_raw: int) -> dict[str, object]:
    return {
        "part_kind": "tfbs",
        "regulator": regulator,
        "offset_raw": offset_raw,
        "length": 6,
        "end_raw": offset_raw + 6,
    }


def _fixed(
    role: str,
    offset_raw: int,
    *,
    variant_id: str | None = None,
    sequence: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "part_kind": "fixed_element",
        "role": role,
        "offset_raw": offset_raw,
        "length": 6,
        "end_raw": offset_raw + 6,
        "spacer_length": 16,
    }
    if variant_id is not None:
        row["variant_id"] = variant_id
    if sequence is not None:
        row["sequence"] = sequence
    return row
