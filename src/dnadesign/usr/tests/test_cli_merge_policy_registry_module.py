"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_merge_policy_registry_module.py

Tests for CLI merge duplicate-policy registry helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.usr.src.cli_merge_policy import register_merge_policy, resolve_merge_policy
from dnadesign.usr.src.errors import SequencesError
from dnadesign.usr.src.merge_datasets import MergePolicy


def test_resolve_merge_policy_supports_builtin_values() -> None:
    assert resolve_merge_policy("error") is MergePolicy.ERROR
    assert resolve_merge_policy("skip") is MergePolicy.SKIP


def test_resolve_merge_policy_rejects_unknown_value() -> None:
    with pytest.raises(SequencesError, match="Unsupported duplicate policy"):
        resolve_merge_policy("unknown")


def test_register_merge_policy_rejects_duplicate_name() -> None:
    register_merge_policy("unit_custom_merge", MergePolicy.ERROR)
    with pytest.raises(SequencesError, match="duplicate policy name"):
        register_merge_policy("unit_custom_merge", MergePolicy.SKIP)
