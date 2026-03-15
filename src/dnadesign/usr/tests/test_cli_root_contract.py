"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_root_contract.py

Tests for USR root path contract enforcement.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

from typer.testing import CliRunner

import dnadesign.usr.src.cli as cli_module
from dnadesign.usr import Dataset
from dnadesign.usr.tests.registry_helpers import ensure_registry


def test_cli_rejects_legacy_package_archive_roots(tmp_path: Path, monkeypatch) -> None:
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
    assert result_datasets_archived.exit_code != 0
    assert "legacy" in result_datasets_archived.output.lower()


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
