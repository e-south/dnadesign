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

    run_name = "sample/results"
    run_dir = tmp_path / "results"
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


def test_run_index_rebuild_includes_set_scoped_runs(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA"], ["cpxR"]]},
            "catalog": {"root": "cache_root", "pwm_source": "matrix"},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cfg = load_config(config_path)

    run_dir = tmp_path / "results" / "set1_lexA"
    run_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "stage": "sample",
        "created_at": created_at,
        "run_dir": str(run_dir.resolve()),
        "motifs": [{"tf_name": "lexA"}],
        "motif_store": {"pwm_source": "matrix"},
        "regulator_set": {"index": 1, "tfs": ["lexA"]},
        "run_group": "set1_lexA",
        "artifacts": [],
    }
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest))

    index_path = rebuild_run_index(cfg, config_path)
    payload = json.loads(index_path.read_text())
    assert "sample/set1_lexA" in payload


def test_run_index_rebuild_ignores_study_and_portfolio_subtrees(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "outputs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": "cache_root", "pwm_source": "matrix"},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cfg = load_config(config_path)

    primary_run_dir = tmp_path / "outputs"
    primary_run_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    primary_manifest = {
        "stage": "sample",
        "created_at": created_at,
        "run_dir": str(primary_run_dir.resolve()),
        "motifs": [{"tf_name": "lexA"}],
        "motif_store": {"pwm_source": "matrix"},
        "regulator_set": {"index": 1, "tfs": ["lexA"]},
        "artifacts": [],
    }
    primary_manifest_file = manifest_path(primary_run_dir)
    primary_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    primary_manifest_file.write_text(json.dumps(primary_manifest))

    study_trial_run_dir = tmp_path / "outputs" / "studies" / "length_vs_score" / "abc123" / "trials" / "L6" / "seed_1"
    study_trial_manifest = {
        "stage": "sample",
        "created_at": created_at,
        "run_dir": str(study_trial_run_dir.resolve()),
        "motifs": [{"tf_name": "lexA"}],
        "motif_store": {"pwm_source": "matrix"},
        "regulator_set": {"index": 1, "tfs": ["lexA"]},
        "artifacts": [],
    }
    study_trial_manifest_file = manifest_path(study_trial_run_dir)
    study_trial_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    study_trial_manifest_file.write_text(json.dumps(study_trial_manifest))

    portfolio_run_dir = tmp_path / "outputs" / "portfolios" / "handoff" / "xyz987"
    portfolio_manifest = {
        "stage": "sample",
        "created_at": created_at,
        "run_dir": str(portfolio_run_dir.resolve()),
        "motifs": [{"tf_name": "lexA"}],
        "motif_store": {"pwm_source": "matrix"},
        "regulator_set": {"index": 1, "tfs": ["lexA"]},
        "artifacts": [],
    }
    portfolio_manifest_file = manifest_path(portfolio_run_dir)
    portfolio_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    portfolio_manifest_file.write_text(json.dumps(portfolio_manifest))

    index_path = rebuild_run_index(cfg, config_path)
    payload = json.loads(index_path.read_text())
    assert list(payload.keys()) == ["sample/outputs"]
