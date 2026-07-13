"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_svg_assertions.py

SVG geometry assertions for Eco1 RT selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re


def assert_svg_aspect_ratio_at_most(svg_text: str, *, maximum: float) -> None:
    match = re.search(r'viewBox="[0-9.]+ [0-9.]+ ([0-9.]+) ([0-9.]+)"', svg_text)
    assert match is not None
    width, height = (float(value) for value in match.groups())
    assert width / height <= maximum


def assert_svg_aspect_ratio_at_least(svg_text: str, *, minimum: float) -> None:
    match = re.search(r'viewBox="[0-9.]+ [0-9.]+ ([0-9.]+) ([0-9.]+)"', svg_text)
    assert match is not None
    width, height = (float(value) for value in match.groups())
    assert width / height >= minimum
