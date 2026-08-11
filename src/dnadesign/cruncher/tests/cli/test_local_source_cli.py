"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/cli/test_local_source_cli.py

Regression tests for local source CLI Cruncher CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
from pathlib import Path
from types import SimpleNamespace

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def test_sources_list_includes_local_sources(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": ".cruncher"},
            "ingest": {
                "local_sources": [
                    {
                        "source_id": "local_omalle",
                        "root": "motifs",
                        "patterns": ["*.txt"],
                        "format_map": {".txt": "MEME"},
                    }
                ]
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = runner.invoke(app, ["sources", "list", str(config_path)])
    assert result.exit_code == 0
    assert "local_omalle" in result.output


def test_sources_materialize_promoters_publishes_complete_export(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "source-data"
    data_root.mkdir()
    destination = tmp_path / "promoter-export"

    def _materialize(staging: Path, **kwargs):
        assert kwargs == {
            "data_root": data_root,
            "require_association_sources": True,
        }
        (staging / "manifest.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(complete=True, record_count=12)

    monkeypatch.setattr(
        "dnadesign.cruncher.cli.commands.sources.export_dnadesign_data_promoter_superset",
        _materialize,
    )

    result = runner.invoke(
        app,
        [
            "sources",
            "materialize-promoters",
            str(destination),
            "--data-root",
            str(data_root),
            "--require-association-sources",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "complete": True,
        "destination": str(destination),
        "record_count": 12,
        "schema": "cruncher.promoter-export/v1",
        "status": "ok",
    }
    assert (destination / "manifest.json").is_file()


def test_sources_materialize_promoters_rejects_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "promoter-export"
    destination.mkdir()

    result = runner.invoke(app, ["sources", "materialize-promoters", str(destination)])

    assert result.exit_code == 1
    assert "Destination already exists" in result.output


def test_sources_materialize_promoters_removes_failed_staging_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "promoter-export"

    def _fail(_staging: Path, **_kwargs):
        raise ValueError("invalid source inventory")

    monkeypatch.setattr(
        "dnadesign.cruncher.cli.commands.sources.export_dnadesign_data_promoter_superset",
        _fail,
    )

    result = runner.invoke(app, ["sources", "materialize-promoters", str(destination)])

    assert result.exit_code == 1
    assert "invalid source inventory" in result.output
    assert not destination.exists()
    assert list(tmp_path.glob(".promoter-export.*")) == []


def test_sources_materialize_promoters_rejects_incomplete_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "promoter-export"

    def _materialize(staging: Path, **_kwargs):
        (staging / "manifest.json").write_text("{}\n", encoding="utf-8")
        inventory = SimpleNamespace(route_failure_count=2, conflict_count=1)
        return SimpleNamespace(complete=False, record_count=12, source_inventory=inventory)

    monkeypatch.setattr(
        "dnadesign.cruncher.cli.commands.sources.export_dnadesign_data_promoter_superset",
        _materialize,
    )

    result = runner.invoke(app, ["sources", "materialize-promoters", str(destination)])

    assert result.exit_code == 1
    assert "promoter export is incomplete (2 route failures; 1 sequence conflicts)" in result.output
    assert not destination.exists()
    assert list(tmp_path.glob(".promoter-export.*")) == []


def test_sources_materialize_promoters_never_replaces_destination_created_during_export(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "promoter-export"

    def _materialize(staging: Path, **_kwargs):
        (staging / "manifest.json").write_text("{}\n", encoding="utf-8")
        destination.mkdir()
        (destination / "keep.txt").write_text("keep\n", encoding="utf-8")
        inventory = SimpleNamespace(route_failure_count=0, conflict_count=0)
        return SimpleNamespace(complete=True, record_count=12, source_inventory=inventory)

    monkeypatch.setattr(
        "dnadesign.cruncher.cli.commands.sources.export_dnadesign_data_promoter_superset",
        _materialize,
    )

    result = runner.invoke(app, ["sources", "materialize-promoters", str(destination)])

    assert result.exit_code == 1
    assert "publication is create-only" in result.output
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep\n"


def test_sources_materialize_promoters_reports_staging_preparation_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "promoter-export"

    def _fail(_destination: Path):
        raise OSError("staging is unavailable")

    monkeypatch.setattr(
        "dnadesign.cruncher.cli.commands.sources.CreateOnlyDirectoryPublication.prepare",
        _fail,
    )

    result = runner.invoke(app, ["sources", "materialize-promoters", str(destination)])

    assert result.exit_code == 1
    assert "staging is unavailable" in result.output
    assert not destination.exists()
