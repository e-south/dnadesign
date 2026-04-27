"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/cli/commands/genbank/test_cli_genbank_import.py

CLI tests for USR GenBank import commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typer.testing import CliRunner

from dnadesign.usr.src.cli import app
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.sequence_views import load_sequence_views

from ....datasets.core.test_genbank_import import _write_genbank_fixture, _write_manifest


def test_cli_genbank_import_manifest_flow(tmp_path) -> None:
    root = tmp_path / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    gb_path = tmp_path / "sulap.gb"
    manifest_path = tmp_path / "import.yaml"
    _write_genbank_fixture(gb_path)
    _write_manifest(manifest_path, output_dataset="usr_reference_genbank_native", source_file=gb_path)

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "genbank", "import", "--manifest", str(manifest_path)])

    dataset = Dataset(root, "usr_reference_genbank_native")
    assert result.exit_code == 0, result.stdout
    assert "Imported 1 native record" in result.stdout
    assert dataset.head(10).shape[0] == 1
    assert len(load_sequence_views(dataset)) == 1
