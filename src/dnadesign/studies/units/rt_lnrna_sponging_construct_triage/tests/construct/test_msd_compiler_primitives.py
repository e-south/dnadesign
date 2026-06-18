"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/construct/test_msd_compiler_primitives.py

Primitive-backed MSD compiler combinatorics tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.source_promotions import (
    resolve_msd_compiler_promotions,
)

from .helpers import (
    _WT_RT_CDS_SEQUENCE,
    _repo_root,
    _source_window_policy,
    _write_msd_compiler_pool_spec,
)


def test_rt_lnrna_msd_compiler_expands_snapback_and_scar_nick_primitive_combinatorics(
    tmp_path: Path,
) -> None:
    spec_path = _write_msd_compiler_pool_spec(tmp_path / "msd-pool.yaml")

    promotions = resolve_msd_compiler_promotions(
        repo_root=_repo_root(),
        pool_spec_path=spec_path,
        wt_rt_cds_sequence=_WT_RT_CDS_SEQUENCE,
        window_policy=_source_window_policy(),
    )

    assert len(promotions) == 80
    assert {promotion.overlay_fields["construct_subject__msd_cap_id"] for promotion in promotions} == {
        "CDE033R01",
        "CDE033R02",
        "CDE033R03",
        "CDE033R04",
        "CDE033R05",
    }
    assert {
        (
            promotion.overlay_fields["construct_subject__msd_stem_base_left"],
            promotion.overlay_fields["construct_subject__msd_stem_base_right"],
            promotion.overlay_fields["construct_subject__msd_profile_s3s2s1s0"],
        )
        for promotion in promotions
    } == {
        ("AGTG", "CAAT", "MXMM"),
        ("AGTG", "CATG", "XWMM"),
        ("AGTG", "CTTT", "MWXM"),
        ("AGTG", "CGAT", "MXWM"),
        ("AATG", "CGTG", "XMWM"),
        ("AGTG", "CATT", "MWMM"),
        ("AATG", "CGTT", "MMWM"),
        ("AGTG", "CGTT", "MWWM"),
        ("AGTG", "CAAG", "XXMM"),
        ("AATG", "CTTG", "XMXM"),
        ("AATG", "CAGT", "MXMM"),
        ("AGTG", "CAGT", "MXMM"),
        ("AATG", "CAAT", "MXMM"),
        ("AATG", "CACT", "MXMM"),
        ("CTCT", "AGTG", "MXMM"),
        ("CTCA", "TGTG", "MXMM"),
    }
    assert len({promotion.source_record_id for promotion in promotions}) == 80
    assert len({promotion.lnrna_sequence for promotion in promotions}) == 80
