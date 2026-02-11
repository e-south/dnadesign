"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_convert_legacy_imports.py

Import-time contract tests for legacy conversion helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import importlib
import sys

import pytest


def test_convert_legacy_import_is_lazy() -> None:
    if "torch" in sys.modules:
        pytest.skip("torch already imported; cannot assert lazy import")
    sys.modules.pop("dnadesign.usr.src.convert_legacy", None)
    importlib.import_module("dnadesign.usr.src.convert_legacy")
    assert "torch" not in sys.modules
