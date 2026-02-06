"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_run_layout_optimizer_stats.py

Validates manifest/sidecar handling for optimizer stats payloads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.cruncher.app.sample.run_layout import _materialize_optimizer_stats


def test_materialize_optimizer_stats_moves_move_stats_to_sidecar(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "sample" / "run_a"
    run_dir.mkdir(parents=True, exist_ok=True)
    move_stats = [{"sweep_idx": 0, "attempted": 10, "accepted": 5, "phase": "draw"}]
    raw_stats = {
        "acceptance_rate_all": 0.5,
        "swap_acceptance_rate": 0.2,
        "move_stats": move_stats,
    }

    manifest_stats, artifact_entries, stats_path = _materialize_optimizer_stats(
        run_dir,
        raw_stats,
        stage="sample",
    )

    assert "move_stats" not in manifest_stats
    assert manifest_stats["move_stats_rows"] == 1
    assert stats_path is not None
    assert len(artifact_entries) == 1
    assert artifact_entries[0]["path"] == stats_path

    payload = json.loads((run_dir / stats_path).read_text())
    assert payload["move_stats"] == move_stats


def test_materialize_optimizer_stats_skips_sidecar_without_move_stats(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "sample" / "run_b"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_stats = {"acceptance_rate_all": 0.5}

    manifest_stats, artifact_entries, stats_path = _materialize_optimizer_stats(
        run_dir,
        raw_stats,
        stage="sample",
    )

    assert manifest_stats == raw_stats
    assert artifact_entries == []
    assert stats_path is None
