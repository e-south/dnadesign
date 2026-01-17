"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_runs_clean.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.app.run_service import load_run_index, save_run_index
from dnadesign.cruncher.artifacts.layout import status_path
from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def test_runs_clean_marks_stale_running(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [["lexA"]],
            "motif_store": {
                "catalog_root": str(catalog_root),
                "pwm_source": "matrix",
                "source_preference": ["regulondb"],
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = tmp_path / "runs" / "sample" / "run_stale"
    run_dir.mkdir(parents=True, exist_ok=True)
    old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    payload = {
        "stage": "sample",
        "status": "running",
        "run_dir": str(run_dir.resolve()),
        "started_at": old_time,
        "updated_at": old_time,
    }
    status_file = status_path(run_dir)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps(payload, indent=2))

    save_run_index(
        config_path,
        {
            "run_stale": {
                "stage": "sample",
                "status": "running",
                "run_dir": str(run_dir.resolve()),
                "created_at": old_time,
            }
        },
    )

    result = runner.invoke(
        app,
        ["runs", "clean", "--stale", "--older-than-hours", "0", str(config_path)],
        color=False,
    )
    assert result.exit_code == 0

    run_index = load_run_index(config_path)
    assert run_index["run_stale"]["status"] == "aborted"
    status_payload = json.loads(status_file.read_text())
    assert status_payload["status"] == "aborted"
