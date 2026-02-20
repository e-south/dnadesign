"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_registry_strict_loading.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from dnadesign.opal.src.core.utils import OpalError


def test_models_registry_raises_on_bad_builtin_module(tmp_path: Path) -> None:
    reg = importlib.import_module("dnadesign.opal.src.registries.models")
    pkg = importlib.import_module("dnadesign.opal.src.models")

    bad_dir = tmp_path / "bad_models"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "bad_model.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    pkg.__path__.append(str(bad_dir))
    reg._BUILTINS_LOADED = False
    try:
        with pytest.raises(OpalError, match="Failed to import built-in models modules"):
            reg.list_models()
    finally:
        pkg.__path__.remove(str(bad_dir))
