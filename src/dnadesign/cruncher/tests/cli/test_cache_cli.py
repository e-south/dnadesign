"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_cache_cli.py

Validates cache maintenance CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def _write_config(tmp_path):
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_cache_clean_dry_run_lists_matches_without_deleting(tmp_path):
    config_path = _write_config(tmp_path)
    pycache_dir = tmp_path / "src" / "pkg" / "__pycache__"
    pytest_cache_dir = tmp_path / "tests" / ".pytest_cache"
    pycache_dir.mkdir(parents=True, exist_ok=True)
    pytest_cache_dir.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(app, ["cache", "clean", "--scope", "workspace", str(config_path)], color=False)

    assert result.exit_code == 0
    assert "Dry-run only" in result.output
    assert pycache_dir.exists()
    assert pytest_cache_dir.exists()


def test_cache_clean_apply_deletes_matches(tmp_path):
    config_path = _write_config(tmp_path)
    pycache_dir = tmp_path / "src" / "pkg" / "__pycache__"
    pycache_dir.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(
        app,
        ["cache", "clean", "--scope", "workspace", "--apply", str(config_path)],
        color=False,
    )

    assert result.exit_code == 0
    assert "Deleted" in result.output
    assert not pycache_dir.exists()


def test_cache_clean_root_override_scans_outside_workspace(tmp_path):
    config_path = _write_config(tmp_path)
    outside_pycache = tmp_path / "outside" / "__pycache__"
    outside_pycache.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(
        app,
        ["cache", "clean", "--scope", "workspace", "--root", str(tmp_path), str(config_path)],
        color=False,
    )

    assert result.exit_code == 0
    assert "outside/__pycache__" in result.output


def test_cache_clean_accepts_package_scope(tmp_path):
    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["cache", "clean", "--scope", "package", str(config_path)], color=False)
    assert result.exit_code == 0
