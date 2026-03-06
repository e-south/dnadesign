"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/test_batch_policy.py

Contract tests for infer runtime batch-size policy helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.infer.batch_policy import (
    resolve_default_extract_batch_size,
    resolve_extract_batch_policy,
    resolve_micro_batch_size,
)


def test_resolve_micro_batch_size_prefers_model_value(monkeypatch) -> None:
    monkeypatch.setenv("DNADESIGN_INFER_BATCH", "128")
    assert resolve_micro_batch_size(model_batch_size=16) == 16


def test_resolve_micro_batch_size_uses_env_when_model_is_none(monkeypatch) -> None:
    monkeypatch.setenv("DNADESIGN_INFER_BATCH", "128")
    assert resolve_micro_batch_size(model_batch_size=None) == 128


def test_resolve_micro_batch_size_defaults_to_zero(monkeypatch) -> None:
    monkeypatch.delenv("DNADESIGN_INFER_BATCH", raising=False)
    assert resolve_micro_batch_size(model_batch_size=None) == 0


def test_resolve_default_extract_batch_size_defaults_to_64(monkeypatch) -> None:
    monkeypatch.delenv("DNADESIGN_INFER_DEFAULT_BS", raising=False)
    assert resolve_default_extract_batch_size() == 64


def test_resolve_extract_batch_policy_reads_both_values(monkeypatch) -> None:
    monkeypatch.setenv("DNADESIGN_INFER_BATCH", "32")
    monkeypatch.setenv("DNADESIGN_INFER_DEFAULT_BS", "96")
    assert resolve_extract_batch_policy(model_batch_size=None) == (32, 96)


def test_batch_policy_fails_fast_on_invalid_integer_env(monkeypatch) -> None:
    monkeypatch.setenv("DNADESIGN_INFER_BATCH", "nope")
    with pytest.raises(ValueError):
        resolve_micro_batch_size(model_batch_size=None)
