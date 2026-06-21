"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/support/pwm_fixtures.py

PWM fixture data for Retron hairpin review-output tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def write_test_tetr_meme_pwm(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEST_TETR_MEME_PWM, encoding="utf-8")
    return path


_TEST_TETR_MEME_PWM = """\
MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies:
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF tetR
letter-probability matrix: alength= 4 w= 17
0.258275 0.368716 0.23121 0.141799
0.166812 0.358693 0.100411 0.374084
0.110052 0.486613 0.153344 0.249991
0.154816 0.0198925 0.232402 0.59289
0.750836 0.0478954 0.0703561 0.130913
0.0871603 0.0544072 0.0756621 0.78277
0.167185 0.647598 0.0461668 0.139051
0.338091 0.124663 0.346859 0.190386
0.239939 0.278202 0.27095 0.210909
0.235833 0.380002 0.0926193 0.291546
0.0536714 0.0204268 0.814817 0.111085
0.79187 0.0669451 0.0842271 0.0569581
0.200266 0.0967125 0.0353436 0.667678
0.427505 0.318375 0.0292238 0.224896
0.260031 0.103466 0.490511 0.145992
0.440424 0.142895 0.22872 0.18796
0.142758 0.183362 0.40392 0.26996
"""


__all__ = ["write_test_tetr_meme_pwm"]
