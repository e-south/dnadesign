"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/config/test_stage_a_namespace_layout.py

Tests that Stage-A APIs are exposed from the canonical core namespace only.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util


def test_stage_a_adapter_namespace_is_not_present() -> None:
    spec = importlib.util.find_spec("dnadesign.densegen.src.adapters.sources.stage_a")
    assert spec is None or spec.loader is None
