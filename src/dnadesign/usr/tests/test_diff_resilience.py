"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_diff_resilience.py

Pressure tests for USR diff helpers under local parquet corruption.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.usr.src.diff import file_stats, parquet_stats
from dnadesign.usr.src.errors import VerificationError


def test_parquet_stats_fails_with_verification_error_on_corrupt_primary(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    records_path.write_text("not a parquet file", encoding="utf-8")

    with pytest.raises(VerificationError, match="Failed to read local parquet file"):
        parquet_stats(records_path, include_sha=False, include_parquet=True)


def test_file_stats_fails_with_verification_error_on_corrupt_parquet(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.parquet"
    payload_path.write_text("broken parquet payload", encoding="utf-8")

    with pytest.raises(VerificationError, match="Failed to read local parquet file"):
        file_stats(payload_path, include_sha=False, include_parquet=True)
