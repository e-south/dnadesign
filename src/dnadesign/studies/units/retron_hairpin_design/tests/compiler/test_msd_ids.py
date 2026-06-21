"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_msd_ids.py

MSD identifier parser tests for the Retron MSD compiler.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.retron_hairpin_design.catalog.msd_ids import (
    MsdDesignPartInput,
    MsdIdError,
    compute_scar_nick_profile,
    parse_msd_construct_label,
    parse_msd_design_parts,
)


def test_compute_scar_nick_profile_uses_s3_to_s0_convention() -> None:
    assert compute_scar_nick_profile(left_base="CGGT", right_base="ACAG") == "MXMM"
    assert compute_scar_nick_profile(left_base="CTCT", right_base="AGTG") == "MXMM"
    assert compute_scar_nick_profile(left_base="AGTG", right_base="CAAG") == "XXMM"


def test_parse_msd_construct_label_infers_profile_when_missing() -> None:
    parsed = parse_msd_construct_label("pES-retron-177-msd[TetR]; C172-LCGGT-RACAG")

    assert parsed.construct_id == "pES-retron-177"
    assert parsed.payload_id == "TetR"
    assert parsed.cap_id == "C172"
    assert parsed.left_base == "CGGT"
    assert parsed.right_base == "ACAG"
    assert parsed.profile_s3s2s1s0 == "MXMM"
    assert parsed.msd_design_id == "msd-tetr-C172-LCGGT-RACAG-MXMM"


def test_parse_msd_design_parts_uses_same_static_lint_without_manual_label_syntax() -> None:
    parsed = parse_msd_design_parts(
        MsdDesignPartInput(
            construct_id="pES-retron-177",
            payload_id="TetR",
            cap_id="C172",
            left_base="CGGT",
            right_base="ACAG",
        )
    )

    assert parsed.construct_label == "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM"
    assert parsed.profile_s3s2s1s0 == "MXMM"


def test_parse_msd_construct_label_rejects_wrong_profile() -> None:
    with pytest.raises(MsdIdError, match="provided profile"):
        parse_msd_construct_label("pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MMMM")


def test_parse_msd_construct_label_rejects_non_ligatable_s0() -> None:
    with pytest.raises(MsdIdError, match="S0"):
        parse_msd_construct_label("pES-retron-177-msd[TetR]; C172-LCGGT-RCCAA")


def test_parse_msd_construct_label_allows_non_ligatable_s0_with_explicit_opt_in() -> None:
    parsed = parse_msd_construct_label(
        "pES-retron-177-msd[TetR]; C172-LCGGG-RACAG-MXMX",
        allow_non_ligatable_s0=True,
    )

    assert parsed.left_base == "CGGG"
    assert parsed.right_base == "ACAG"
    assert parsed.profile_s3s2s1s0 == "MXMX"
    assert parsed.s0_match_required is False
    assert parsed.msd_design_id == "msd-tetr-C172-LCGGG-RACAG-MXMX"
