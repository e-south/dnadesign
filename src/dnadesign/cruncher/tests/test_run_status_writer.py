"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_run_status_writer.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.cruncher.artifacts.layout import live_metrics_path, status_path
from dnadesign.cruncher.artifacts.status import RunStatusWriter


def test_run_status_writer_emits_live_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    status_file = status_path(run_dir)
    metrics_path = live_metrics_path(run_dir)

    writer = RunStatusWriter(
        path=status_file,
        stage="sample",
        run_dir=run_dir,
        metrics_path=metrics_path,
        payload={"status_message": "init"},
    )
    writer.update(progress_pct=10.0, current_score=1.0, best_score=1.5)
    writer.finish(status="completed")

    lines = metrics_path.read_text().strip().splitlines()
    assert len(lines) >= 3
    events = [json.loads(line).get("event") for line in lines]
    assert events[0] == "start"
    assert "update" in events
    assert events[-1] == "finish"
