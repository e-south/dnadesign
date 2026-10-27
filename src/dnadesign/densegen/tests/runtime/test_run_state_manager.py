"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_run_state_manager.py

Unit tests for run-state reconciliation against durable output counts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

from dnadesign.densegen.src.core.pipeline.run_state_manager import reconcile_run_state_with_outputs
from dnadesign.densegen.src.core.run_state import RunState, load_run_state


def test_reconcile_run_state_with_outputs_rewrites_when_checkpoint_ahead(tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "outputs" / "meta" / "run_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    config_sha = hashlib.sha256(b"demo").hexdigest()
    state = RunState.from_counts(
        run_id="run-1",
        schema_version="2.9",
        config_sha256=config_sha,
        accepted_config_sha256=[config_sha],
        run_root=str(run_root),
        counts={("demo_input", "demo_plan"): 5},
        created_at="2026-02-01T00:00:00Z",
        updated_at="2026-02-01T00:00:00Z",
    )
    state.write_json(state_path)

    result = reconcile_run_state_with_outputs(
        path=state_path,
        run_id="run-1",
        schema_version="2.9",
        config_sha256=config_sha,
        accepted_config_sha256=[config_sha],
        run_root=str(run_root),
        existing_counts={("demo_input", "demo_plan"): 2},
        created_at="2026-02-01T00:00:00Z",
    )

    assert result.updated is True
    assert result.state_total == 5
    assert result.durable_total == 2
    reconciled = load_run_state(state_path)
    assert reconciled.items[0].generated == 2


def test_reconcile_run_state_with_outputs_rewrites_to_zero_when_durable_outputs_missing(tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "outputs" / "meta" / "run_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    config_sha = hashlib.sha256(b"demo").hexdigest()
    state = RunState.from_counts(
        run_id="run-1",
        schema_version="2.9",
        config_sha256=config_sha,
        accepted_config_sha256=[config_sha],
        run_root=str(run_root),
        counts={("demo_input", "demo_plan"): 3},
        created_at="2026-02-01T00:00:00Z",
        updated_at="2026-02-01T00:00:00Z",
    )
    state.write_json(state_path)

    result = reconcile_run_state_with_outputs(
        path=state_path,
        run_id="run-1",
        schema_version="2.9",
        config_sha256=config_sha,
        accepted_config_sha256=[config_sha],
        run_root=str(run_root),
        existing_counts={},
        created_at="2026-02-01T00:00:00Z",
    )

    assert result.updated is True
    assert result.state_total == 3
    assert result.durable_total == 0
    reconciled = load_run_state(state_path)
    assert reconciled.items == []
