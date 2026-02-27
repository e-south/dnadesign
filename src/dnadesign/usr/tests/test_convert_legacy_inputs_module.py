"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_convert_legacy_inputs_module.py

Contract tests for convert-legacy profile and input helper extraction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from dnadesign.usr.src.errors import SchemaError, ValidationError


def test_convert_legacy_inputs_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_inputs")
    assert hasattr(module, "Profile")
    assert hasattr(module, "profile_60bp_dual_promoter")
    assert hasattr(module, "_coerce_logits")
    assert hasattr(module, "_tf_from_parts")
    assert hasattr(module, "_count_tf")
    assert hasattr(module, "_ensure_pt_list_of_dicts")
    assert hasattr(module, "_gather_pt_files")


def test_profile_60bp_dual_promoter_contract() -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_inputs")
    profile = module.profile_60bp_dual_promoter()
    assert profile.name == "60bp_dual_promoter_cpxR_LexA"
    assert profile.expected_length == 60
    assert profile.logits_expected_dim == 512
    assert profile.densegen_plan == "sigma70_mid"


def test_coerce_logits_list_and_flatten_contract() -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_inputs")
    out = module._coerce_logits([[1.0, 2.0]], want_dim=2)
    assert out == [1.0, 2.0]


def test_coerce_logits_list_length_mismatch_raises() -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_inputs")
    with pytest.raises(ValidationError, match="logits length mismatch"):
        module._coerce_logits([1.0], want_dim=2)


def test_tf_parsing_helpers_contract() -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_inputs")
    parts = ["cpxr:AAA", "lexa:TTT", "lexa:GGG", "other"]
    assert module._tf_from_parts(parts) == ["cpxr", "lexa"]
    assert module._count_tf(parts) == {"cpxr": 1, "lexa": 2}


def test_gather_pt_files_rejects_non_pt_non_dir(tmp_path: Path) -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_inputs")
    other = tmp_path / "bad.txt"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(SchemaError, match="Not a .pt file or directory"):
        module._gather_pt_files([other])
