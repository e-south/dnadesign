"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_runs_clean.py

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
from dnadesign.cruncher.artifacts.layout import manifest_path, status_path
from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def test_runs_clean_marks_stale_running(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {
                "root": str(catalog_root),
                "pwm_source": "matrix",
                "source_preference": ["regulondb"],
            },
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


def _write_sample_manifest(run_dir: Path, *, created_at: str) -> None:
    payload = {
        "stage": "sample",
        "created_at": created_at,
        "run_dir": str(run_dir.resolve()),
        "motif_store": {"pwm_source": "matrix"},
        "artifacts": [],
    }
    mf = manifest_path(run_dir)
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(payload))


def test_runs_prune_dry_run_keeps_files(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(catalog_root), "pwm_source": "matrix", "source_preference": ["regulondb"]},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    now = datetime.now(timezone.utc)
    new_time = now.isoformat()
    old_time = (now - timedelta(days=40)).isoformat()
    run_new = tmp_path / "runs" / "sample" / "run_new"
    run_old = tmp_path / "runs" / "sample" / "run_old"
    run_new.mkdir(parents=True, exist_ok=True)
    run_old.mkdir(parents=True, exist_ok=True)
    _write_sample_manifest(run_new, created_at=new_time)
    _write_sample_manifest(run_old, created_at=old_time)

    save_run_index(
        config_path,
        {
            "run_new": {
                "stage": "sample",
                "status": "completed",
                "run_dir": str(run_new.resolve()),
                "created_at": new_time,
            },
            "run_old": {
                "stage": "sample",
                "status": "completed",
                "run_dir": str(run_old.resolve()),
                "created_at": old_time,
            },
        },
    )

    result = runner.invoke(
        app,
        ["runs", "prune", "--stage", "sample", "--keep-latest", "1", "--older-than-days", "0", str(config_path)],
        color=False,
    )
    assert result.exit_code == 0
    assert "Dry-run only" in result.output
    assert run_old.exists()
    assert run_new.exists()
    run_index = load_run_index(config_path)
    assert "run_old" in run_index


def test_runs_prune_apply_archives_old_runs_and_updates_index(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(catalog_root), "pwm_source": "matrix", "source_preference": ["regulondb"]},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    now = datetime.now(timezone.utc)
    new_time = now.isoformat()
    old_time = (now - timedelta(days=40)).isoformat()
    run_new = tmp_path / "runs" / "sample" / "run_new"
    run_old = tmp_path / "runs" / "sample" / "run_old"
    run_new.mkdir(parents=True, exist_ok=True)
    run_old.mkdir(parents=True, exist_ok=True)
    _write_sample_manifest(run_new, created_at=new_time)
    _write_sample_manifest(run_old, created_at=old_time)

    save_run_index(
        config_path,
        {
            "run_new": {
                "stage": "sample",
                "status": "completed",
                "run_dir": str(run_new.resolve()),
                "created_at": new_time,
            },
            "run_old": {
                "stage": "sample",
                "status": "completed",
                "run_dir": str(run_old.resolve()),
                "created_at": old_time,
            },
        },
    )

    result = runner.invoke(
        app,
        [
            "runs",
            "prune",
            "--stage",
            "sample",
            "--keep-latest",
            "1",
            "--older-than-days",
            "0",
            "--apply",
            str(config_path),
        ],
        color=False,
    )
    assert result.exit_code == 0
    archive_bucket = datetime.fromisoformat(old_time).strftime("%Y-%m")
    archive_dir = tmp_path / "runs" / "_archive" / "sample" / archive_bucket
    assert not run_old.exists()
    assert run_new.exists()
    assert (archive_dir / "run_old").exists()
    run_index = load_run_index(config_path)
    assert "run_old" not in run_index
    assert "run_new" in run_index


def test_runs_repair_index_drops_entries_missing_manifest(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(catalog_root), "pwm_source": "matrix", "source_preference": ["regulondb"]},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    now = datetime.now(timezone.utc).isoformat()
    valid_run = tmp_path / "runs" / "sample" / "run_valid"
    stale_run = tmp_path / "runs" / "sample" / "run_stale"
    valid_run.mkdir(parents=True, exist_ok=True)
    stale_run.mkdir(parents=True, exist_ok=True)
    _write_sample_manifest(valid_run, created_at=now)

    save_run_index(
        config_path,
        {
            "run_valid": {
                "stage": "sample",
                "status": "completed",
                "run_dir": str(valid_run.resolve()),
                "created_at": now,
            },
            "run_stale": {
                "stage": "sample",
                "status": "aborted",
                "run_dir": str(stale_run.resolve()),
                "created_at": now,
            },
        },
    )

    result = runner.invoke(
        app,
        ["runs", "repair-index", "--apply", str(config_path)],
        color=False,
    )
    assert result.exit_code == 0
    assert "Removed 1 invalid run index entry" in result.output
    run_index = load_run_index(config_path)
    assert "run_valid" in run_index
    assert "run_stale" not in run_index


def test_runs_prune_fails_with_actionable_error_on_invalid_index_entries(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(catalog_root), "pwm_source": "matrix", "source_preference": ["regulondb"]},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(days=40)).isoformat()
    run_old = tmp_path / "runs" / "sample" / "run_old"
    run_invalid = tmp_path / "runs" / "sample" / "run_invalid"
    run_old.mkdir(parents=True, exist_ok=True)
    run_invalid.mkdir(parents=True, exist_ok=True)
    _write_sample_manifest(run_old, created_at=old_time)

    save_run_index(
        config_path,
        {
            "run_old": {
                "stage": "sample",
                "status": "completed",
                "run_dir": str(run_old.resolve()),
                "created_at": old_time,
            },
            "run_invalid": {
                "stage": "sample",
                "status": "aborted",
                "run_dir": str(run_invalid.resolve()),
                "created_at": old_time,
            },
        },
    )

    result = runner.invoke(
        app,
        ["runs", "prune", "--stage", "sample", "--keep-latest", "0", "--older-than-days", "0", str(config_path)],
        color=False,
    )
    assert result.exit_code != 0
    assert "invalid run index entries" in result.output.lower()
    assert "runs repair-index --apply" in result.output
