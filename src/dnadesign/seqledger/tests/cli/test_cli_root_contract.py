"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/cli/test_cli_root_contract.py

Tests for SeqLedger root path contract enforcement.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

from typer.testing import CliRunner

import dnadesign.seqledger.src.cli as cli_module
from dnadesign.devtools.tests.support.usr import ensure_registry
from dnadesign.seqledger import Dataset


def test_cli_rejects_historical_package_archive_root_but_allows_canonical_archive_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pkg_root = tmp_path / "pkg_usr"
    (pkg_root / "datasets").mkdir(parents=True)
    (pkg_root / "archived").mkdir(parents=True)
    (pkg_root / "datasets" / "archived").mkdir(parents=True)
    monkeypatch.setattr(cli_module, "_pkg_usr_root", lambda: pkg_root)

    runner = CliRunner()
    result_archived = runner.invoke(cli_module.app, ["--root", str(pkg_root / "archived"), "ls"])
    assert result_archived.exit_code != 0
    assert "legacy" in result_archived.output.lower()

    result_datasets_archived = runner.invoke(
        cli_module.app,
        ["--root", str(pkg_root / "datasets" / "archived"), "ls"],
    )
    assert result_datasets_archived.exit_code == 0
    assert "legacy" not in result_datasets_archived.output.lower()


def test_cli_normalizes_usr_package_root_to_datasets_root(tmp_path: Path, monkeypatch) -> None:
    pkg_root = tmp_path / "pkg_usr"
    pkg_root.mkdir(parents=True, exist_ok=True)
    (pkg_root / "__init__.py").write_text("# stub\n", encoding="utf-8")
    dataset_root = pkg_root / "datasets"
    ensure_registry(dataset_root)
    ds = Dataset(dataset_root, "demo")
    ds.init(source="test")
    ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")
    monkeypatch.setattr(cli_module, "_pkg_usr_root", lambda: pkg_root)

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["--root", str(pkg_root), "ls", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert '"dataset":"demo"' in result.output
    assert '"dataset":"datasets/demo"' not in result.output


def test_cli_uses_external_usr_root_from_environment(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "external-usr"
    ensure_registry(dataset_root)
    ds = Dataset(dataset_root, "external_demo")
    ds.init(source="test")
    ds.add_sequences(["ACGT"], bio_type="dna", alphabet="dna_4", source="test")
    monkeypatch.setenv("DNADESIGN_SEQLEDGER_ROOT", str(dataset_root))

    result = CliRunner().invoke(cli_module.app, ["ls", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert '"dataset":"external_demo"' in result.output


def test_cli_defaults_to_project_datasets_without_creating_directories(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DNADESIGN_SEQLEDGER_ROOT", raising=False)
    result = CliRunner().invoke(cli_module.app, ["ls"])
    assert result.exit_code != 0
    assert "seqledger_datasets" in result.output
    assert "Create" in result.output
    assert not (tmp_path / "seqledger_datasets").exists()

    ensure_registry(tmp_path / "seqledger_datasets")
    ds = Dataset(tmp_path / "seqledger_datasets", "local_demo")
    ds.init(source="test")
    result = CliRunner().invoke(cli_module.app, ["ls", "--format", "json"])
    assert result.exit_code == 0, result.output
    assert '"dataset":"local_demo"' in result.output


def test_cli_rejects_missing_external_usr_root_from_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    missing_root = tmp_path / "missing-usr"
    monkeypatch.setenv("DNADESIGN_SEQLEDGER_ROOT", str(missing_root))

    result = CliRunner().invoke(cli_module.app, ["ls", "--format", "json"])

    assert result.exit_code != 0
    assert "SeqLedger root is not an existing directory" in result.output
    assert not missing_root.exists()
