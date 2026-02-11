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
