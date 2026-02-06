"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_optimizer_stats_sidecar.py

Validates strict loading of optimizer move-stats sidecar data during analysis.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.cruncher.app.analyze_workflow import _resolve_optimizer_stats


def test_resolve_optimizer_stats_loads_move_stats_sidecar(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    sidecar = run_dir / "artifacts" / "optimizer_move_stats.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"move_stats": [{"sweep_idx": 0, "attempted": 10, "accepted": 3}]}))

    manifest = {
        "optimizer_stats": {
            "acceptance_rate_all": 0.3,
            "move_stats_path": "artifacts/optimizer_move_stats.json",
            "move_stats_rows": 1,
        }
    }

    resolved = _resolve_optimizer_stats(manifest, run_dir)
    assert isinstance(resolved, dict)
    assert isinstance(resolved.get("move_stats"), list)
    assert len(resolved["move_stats"]) == 1


def test_resolve_optimizer_stats_requires_sidecar_when_declared(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "optimizer_stats": {
            "move_stats_path": "artifacts/optimizer_move_stats.json",
        }
    }

    with pytest.raises(FileNotFoundError, match="Missing optimizer stats sidecar"):
        _resolve_optimizer_stats(manifest, run_dir)
