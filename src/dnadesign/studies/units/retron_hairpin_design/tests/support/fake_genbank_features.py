"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/support/fake_genbank_features.py

Fake GenBank feature blocks for Retron review-output tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


def fake_reverse_complement_msd_features(payload_trim_id: str) -> list[str]:
    return [
        "     misc_feature    complement(1..17)",
        '                     /label="3\' Flanking"',
        '                     /dnadesign_role="flank_3p"',
        '                     /strand="-1"',
        "     misc_feature    complement(14..17)",
        '                     /label="Right Base"',
        '                     /dnadesign_role="stem_base_right"',
        '                     /strand="-1"',
        "     misc_feature    complement(18..32)",
        f'                     /label="msd[{payload_trim_id}] complement"',
        '                     /dnadesign_role="payload_complement"',
        '                     /strand="-1"',
        "     misc_feature    complement(33..36)",
        '                     /label="Foldback"',
        '                     /dnadesign_feature_id="snapback_foldback_geometry"',
        '                     /dnadesign_role="snapback_foldback_geometry"',
        '                     /strand="-1"',
        "     misc_feature    complement(37..51)",
        f'                     /label="msd[{payload_trim_id}]"',
        '                     /dnadesign_role="payload_primary"',
        '                     /strand="-1"',
        "     misc_feature    complement(52..55)",
        '                     /label="Left Base"',
        '                     /dnadesign_role="stem_base_left"',
        '                     /strand="-1"',
        "     misc_feature    complement(52..66)",
        '                     /label="5\' Flanking"',
        '                     /dnadesign_role="flank_5p"',
        '                     /strand="-1"',
    ]


__all__ = ["fake_reverse_complement_msd_features"]
