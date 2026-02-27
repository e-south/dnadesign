"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_convert_legacy_dedupe_module.py

Contract tests for case-insensitive dedupe policy helpers used by legacy repair.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib

import pyarrow as pa
import pytest

from dnadesign.usr.src.errors import ValidationError


def _make_casefold_dupe_table() -> pa.Table:
    return pa.table(
        {
            "id": ["r1", "r2", "r3"],
            "bio_type": ["dna", "DNA", "dna"],
            "sequence": ["acgt", "ACGT", "TTTT"],
            "created_at": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
        }
    )


def test_convert_legacy_dedupe_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_dedupe")
    assert hasattr(module, "apply_casefold_sequence_dedupe")


def test_apply_casefold_sequence_dedupe_keep_last_drops_older_duplicate() -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_dedupe")
    table = _make_casefold_dupe_table()
    out = module.apply_casefold_sequence_dedupe(
        table,
        dedupe_policy="keep-last",
        dry_run=False,
        assume_yes=True,
    )
    assert out.num_rows == 2
    assert set(out.column("id").to_pylist()) == {"r2", "r3"}


def test_apply_casefold_sequence_dedupe_dry_run_only_plans_changes() -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_dedupe")
    table = _make_casefold_dupe_table()
    out = module.apply_casefold_sequence_dedupe(
        table,
        dedupe_policy="keep-first",
        dry_run=True,
        assume_yes=True,
    )
    assert out.num_rows == 3
    assert out.column("id").to_pylist() == ["r1", "r2", "r3"]


def test_apply_casefold_sequence_dedupe_ask_rejects_invalid_selection() -> None:
    module = importlib.import_module("dnadesign.usr.src.convert_legacy_dedupe")
    table = _make_casefold_dupe_table()

    prompts: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return "x"

    with pytest.raises(ValidationError, match="Invalid selection"):
        module.apply_casefold_sequence_dedupe(
            table,
            dedupe_policy="ask",
            dry_run=False,
            assume_yes=True,
            input_fn=_input,
        )
    assert prompts
