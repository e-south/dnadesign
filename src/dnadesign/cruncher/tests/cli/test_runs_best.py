"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_runs_best.py

Validates runs best ignores failed runs by default.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.app.run_service import save_run_index
from dnadesign.cruncher.artifacts.layout import status_path
from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def test_runs_best_ignores_failed_runs_by_default(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    stage_dir = tmp_path / "runs" / "sample"
    failed_dir = stage_dir / "run_failed"
    passed_dir = stage_dir / "run_passed"
    failed_dir.mkdir(parents=True, exist_ok=True)
    passed_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    failed_status = {"stage": "sample", "status": "failed", "run_dir": str(failed_dir.resolve()), "best_score": 9.0}
    passed_status = {"stage": "sample", "status": "completed", "run_dir": str(passed_dir.resolve()), "best_score": 1.0}
    status_path(failed_dir).parent.mkdir(parents=True, exist_ok=True)
    status_path(passed_dir).parent.mkdir(parents=True, exist_ok=True)
    status_path(failed_dir).write_text(json.dumps(failed_status))
    status_path(passed_dir).write_text(json.dumps(passed_status))

    save_run_index(
        config_path,
        {
            "run_failed": {
                "stage": "sample",
                "status": "failed",
                "created_at": now,
                "run_dir": str(failed_dir.resolve()),
                "best_score": 9.0,
            },
            "run_passed": {
                "stage": "sample",
                "status": "completed",
                "created_at": now,
                "run_dir": str(passed_dir.resolve()),
                "best_score": 1.0,
            },
        },
    )

    result = runner.invoke(app, ["runs", "best", "-c", str(config_path)], color=False)
    assert result.exit_code == 0
    assert result.stdout.strip() == "run_passed"
