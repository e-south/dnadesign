"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_run_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dnadesign.cruncher.app.run_service import (
    drop_run_index_entries,
    get_run,
    load_run_index,
    save_run_index,
    update_run_index_from_status,
)
from dnadesign.cruncher.artifacts.layout import manifest_path
from dnadesign.cruncher.config.load import load_config


def test_get_run_accepts_path(tmp_path: Path) -> None:
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
    run_dir = tmp_path / "results" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "stage": "sample",
        "created_at": created_at,
        "run_dir": str(run_dir.resolve()),
        "motifs": [{"tf_name": "lexA"}],
        "motif_store": {"pwm_source": "matrix"},
        "regulator_set": {"index": 1, "tfs": ["lexA"]},
        "artifacts": ["config_used.yaml"],
    }
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest))

    run = get_run(cfg, config_path, str(run_dir))
    assert run.name == run_name
    assert run.run_dir == run_dir

    rel_path = run_dir.relative_to(config_path.parent)
    run_rel = get_run(cfg, config_path, str(rel_path))
    assert run_rel.run_dir == run_dir


def test_drop_run_index_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher: {}")
    payload = {
        "run_a": {"stage": "sample", "run_dir": str(tmp_path / "run_a")},
        "run_b": {"stage": "sample", "run_dir": str(tmp_path / "run_b")},
    }
    save_run_index(config_path, payload, catalog_root=".cruncher")
    removed = drop_run_index_entries(config_path, ["run_b"], catalog_root=".cruncher")
    assert removed == 1
    index = load_run_index(config_path, catalog_root=".cruncher")
    assert "run_b" not in index
    assert "run_a" in index


def test_update_run_index_uses_semantic_keys_without_slots(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher: {}")

    sample_dir = tmp_path / "results"
    parse_dir = tmp_path / ".cruncher" / "parse"
    sample_dir.mkdir(parents=True, exist_ok=True)
    parse_dir.mkdir(parents=True, exist_ok=True)

    update_run_index_from_status(
        config_path,
        sample_dir,
        {
            "stage": "sample",
            "status": "running",
            "run_dir": str(sample_dir.resolve()),
            "run_group": "lexA",
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    update_run_index_from_status(
        config_path,
        parse_dir,
        {
            "stage": "parse",
            "status": "completed",
            "run_dir": str(parse_dir.resolve()),
            "run_group": "lexA",
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    index = load_run_index(config_path)
    assert "sample/lexA" in index
    assert "parse/lexA" in index
