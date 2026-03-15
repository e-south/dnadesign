"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_resume_policy.py

Contract tests for resume filter chunk-size policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.infer.src.runtime.resume_policy import resolve_resume_filter_chunk_size


def test_resume_filter_chunk_size_defaults_to_10000(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DNADESIGN_INFER_RESUME_FILTER_CHUNK", raising=False)
    assert resolve_resume_filter_chunk_size() == 10_000


def test_resume_filter_chunk_size_reads_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DNADESIGN_INFER_RESUME_FILTER_CHUNK", "2048")
    assert resolve_resume_filter_chunk_size() == 2048


def test_resume_filter_chunk_size_fails_fast_on_non_integer_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DNADESIGN_INFER_RESUME_FILTER_CHUNK", "nope")
    with pytest.raises(ValueError, match="DNADESIGN_INFER_RESUME_FILTER_CHUNK"):
        resolve_resume_filter_chunk_size()


def test_resume_filter_chunk_size_fails_fast_on_non_positive_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DNADESIGN_INFER_RESUME_FILTER_CHUNK", "0")
    with pytest.raises(ValueError, match="must be >= 1"):
        resolve_resume_filter_chunk_size()


def test_resume_filter_chunk_size_fails_fast_on_oversized_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DNADESIGN_INFER_RESUME_FILTER_CHUNK", "100001")
    with pytest.raises(ValueError, match="must be <= 100000"):
        resolve_resume_filter_chunk_size()
