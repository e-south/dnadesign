"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_feature_shard_ledger.py

Infer feature shard checkpoint ledger tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.infer.src.features.shard_ledger import (
    SHARD_LEDGER_SCHEMA_VERSION,
    SHARD_STATUS_COMMITTED,
    SHARD_STATUS_FAILED,
    build_initial_shard_ledger,
    committed_shard_indices,
    load_shard_ledger,
    mark_shard_status,
    resume_shard_indices,
    write_shard_ledger,
)


def test_shard_ledger_resume_skips_committed_and_retries_failed(tmp_path: Path) -> None:
    ledger = build_initial_shard_ledger(
        bundle_id="context_forward",
        runtime_fingerprint_key="fingerprint-test",
        shard_size_views=50,
        pending_view_estimate=125,
        pending_vector_keys=250,
        pending_scalar_keys=125,
    )
    ledger = mark_shard_status(
        ledger,
        shard_index=0,
        status=SHARD_STATUS_COMMITTED,
        committed_views=50,
        committed_vector_keys=84,
        committed_scalar_keys=42,
        checksum="sha256:first",
    )
    ledger = mark_shard_status(
        ledger,
        shard_index=1,
        status=SHARD_STATUS_FAILED,
        error="oom",
    )

    path = write_shard_ledger(tmp_path / "ledger.json", ledger)
    loaded = load_shard_ledger(path)

    assert loaded.schema_version == SHARD_LEDGER_SCHEMA_VERSION
    assert loaded.shard_count == 3
    assert committed_shard_indices(loaded) == (0,)
    assert resume_shard_indices(loaded) == (1, 2)


def test_shard_ledger_rejects_unknown_status() -> None:
    ledger = build_initial_shard_ledger(
        bundle_id="context_forward",
        runtime_fingerprint_key="fingerprint-test",
        shard_size_views=10,
        pending_view_estimate=10,
        pending_vector_keys=20,
        pending_scalar_keys=10,
    )

    with pytest.raises(ValueError, match="Unsupported Infer shard status"):
        mark_shard_status(ledger, shard_index=0, status="unknown")
