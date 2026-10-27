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


def _write_promoter_source_fixture(data_root: Path) -> None:
    release = data_root / "sources/databases/regulondb/13.0"
    promoter_set = release / "promoters/PromoterSet.tsv"
    promoter_set.parent.mkdir(parents=True)
    promoter_set.write_text(
        "\n".join(
            [
                "1)pmId\t2)pmName\t3)strand\t4)posTSS\t5)sigmaFactor\t6)pmSequence\t"
                "7)firstGeneName\t8)pmEvidence\t9)confidenceLevel",
                "PM1\trecAp\tforward\t100\tsigma70\tAACCGGTTAACC\trecA\t[EXP]\tS",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    association_set = release / "binding_sites/TF-RISet.tsv"
    association_set.parent.mkdir()
    association_set.write_text(
        "\n".join(
            [
                "1)riId\t2)riType\t3)regulatorId\t4)regulatorName\t5)cnfName\t6)tfrsID\t"
                "7)tfrsLeft\t8)tfrsRight\t9)strand\t10)tfrsSeq\t11)riFunction\t"
                "12)promoterID\t13)promoterName\t14)tss\t15)sigmaF\t16)tfrsDistToPm\t"
                "17)firstGene\t18)tfrsDistTo1Gene\t19)targetTuOrGene\t20)confidenceLevel\t"
                "21)tfrsEvidence\t22)riEvidence\t23)addEvidence\t24)riEvTech\t25)riEvCategory\t"
                "26)tfrsPMIDS\t27)riPMIDS",
                "RI0001\ttf-promoter\tREG0001\tLexA\tLexA\tBS0001\t10\t20\tforward\t"
                "ACGT\trepressor\tPM1\trecAp\t100\tsigma70\t-50\trecA\t-80\tTU0001:recA\tS\t"
                "EXP:S\tEXP:S\t\tbinding\texpression\t12345\t12345",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


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


def test_sources_materialize_promoters_reads_an_explicit_root_without_sibling_package(tmp_path: Path) -> None:
    data_root = tmp_path / "source-data"
    destination = tmp_path / "promoter-export"
    _write_promoter_source_fixture(data_root)

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
    payload = json.loads(result.output)
    assert payload["complete"] is True
    assert payload["record_count"] == 1
    assert json.loads((destination / "source_files.json").read_text(encoding="utf-8"))[0]["source_id"] == (
        "regulondb_13_promoter_set"
    )
    assert (
        json.loads((destination / "association_source_files.json").read_text(encoding="utf-8"))[0]["source_id"]
        == "regulondb_13_tf_riset"
    )


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
