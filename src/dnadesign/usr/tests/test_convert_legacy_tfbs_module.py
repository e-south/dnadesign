"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_convert_legacy_tfbs_module.py

Contract tests for TFBS helper extraction used by legacy conversion repair flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib


def test_parse_tfbs_parts_filters_invalid_or_short_tokens() -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_tfbs")
    parsed = module._parse_tfbs_parts(
        [
            "cpxr:A",
            "lexa:ATGC",
            "broken",
            ":ATGC",
            " cpxr : atgc ",
            None,
        ],
        min_len=4,
    )
    assert parsed == [("lexa", "ATGC"), ("cpxr", "ATGC")]


def test_scan_used_tfbs_picks_earliest_orientation_and_counts_tf() -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_tfbs")
    used_simple, used_detail, used_counts = module._scan_used_tfbs(
        "TTTGCATAAAAAATGC",
        [("cpxr", "ATGC"), ("lexa", "AAAA")],
    )
    assert used_simple == ["cpxr:ATGC", "lexa:AAAA"]
    assert used_detail == [
        {"offset": 3, "orientation": "rev", "tf": "cpxr", "tfbs": "ATGC"},
        {"offset": 7, "orientation": "fwd", "tf": "lexa", "tfbs": "AAAA"},
    ]
    assert used_counts == {"cpxr": 1, "lexa": 1}


def test_detect_promoter_forward_defaults_to_profile_plan_when_unspecified() -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_tfbs")
    hits = module._detect_promoter_forward("GGACCGCGTTTTTATAATCC", "")
    assert hits == [
        {"offset": 2, "orientation": "fwd", "tf": "sigma70_mid_upstream", "tfbs": "ACCGCG"},
        {"offset": 12, "orientation": "fwd", "tf": "sigma70_mid_downstream", "tfbs": "TATAAT"},
    ]
