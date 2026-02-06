"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_run_index.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dnadesign.cruncher.app.run_service import list_runs, rebuild_run_index
from dnadesign.cruncher.artifacts.layout import manifest_path, status_path
from dnadesign.cruncher.config.load import load_config


def test_run_index_rebuild(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA"]]},
            "catalog": {"root": "cache_root", "pwm_source": "matrix"},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cfg = load_config(config_path)

    run_name = "20250101_000000_abcd12"
    run_dir = tmp_path / "results" / "sample" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "stage": "sample",
        "created_at": created_at,
        "run_dir": str(run_dir.resolve()),
        "motifs": [{"tf_name": "lexA"}],
        "motif_store": {"pwm_source": "matrix"},
        "regulator_set": {"index": 1, "tfs": ["lexA"]},
        "artifacts": ["meta/config_used.yaml"],
    }
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest))
    status_file = status_path(run_dir)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps({"stage": "sample", "status": "completed", "started_at": created_at}))

    index_path = rebuild_run_index(cfg, config_path)
    assert index_path == tmp_path / ".cruncher" / "run_index.json"
    payload = json.loads(index_path.read_text())
    assert run_name in payload

    runs = list_runs(cfg, config_path)
    assert runs
    assert runs[0].name == run_name
    assert runs[0].status == "completed"
