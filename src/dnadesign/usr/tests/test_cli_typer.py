"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_cli_typer.py

Typer CLI integration tests for USR.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.usr.src.cli import app
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.merge_datasets import MergePolicy, MergePreview
from dnadesign.usr.tests.registry_helpers import ensure_registry


def _make_dataset(root: Path) -> None:
    ensure_registry(root)
    ds = Dataset(root, "demo")
    ds.init(source="test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )


def test_cols_accepts_dataset_name(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)
    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "cols", "demo"])
    assert result.exit_code == 0
    assert "sequence" in result.stdout


def test_head_prefers_dataset_mode_for_plain_dataset_id(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)
    (tmp_path / "demo").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["--no-rich", "--root", str(root), "head", "demo", "-n", "1"])
    assert result.exit_code == 0
    assert "ACGT" in result.stdout


def test_head_requires_existing_path_for_explicit_path_target(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "head", "./missing"])
    assert result.exit_code == 4
    assert "Path target not found" in result.stdout


def test_head_accepts_existing_relative_directory_path_with_separator(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)
    monkeypatch.chdir(tmp_path)

    relative_dataset_dir = Path("datasets") / "demo"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--no-rich",
            "--root",
            str(root),
            "head",
            str(relative_dataset_dir),
            "-n",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "ACGT" in result.stdout


def test_cols_prefers_dataset_mode_for_plain_dataset_id(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)
    (tmp_path / "demo").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "cols", "demo"])
    assert result.exit_code == 0
    assert "sequence" in result.stdout


def test_cell_requires_existing_path_for_explicit_path_target(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "cell", "./missing", "--row", "0", "--col", "sequence"])
    assert result.exit_code == 4
    assert "Path target not found" in result.stdout


def test_merge_defaults_are_strict(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    def _fake_merge_usr_to_usr(**kwargs):
        captured["duplicate_policy"] = kwargs["duplicate_policy"]
        captured["overlap_coercion"] = kwargs["overlap_coercion"]
        return MergePreview(
            dest_rows_before=0,
            src_rows=0,
            duplicates_total=0,
            duplicates_skipped=0,
            duplicates_replaced=0,
            duplicate_policy=kwargs["duplicate_policy"],
            new_rows=0,
            dest_rows_after=0,
            columns_total=0,
            overlapping_columns=0,
        )

    monkeypatch.setattr("dnadesign.usr.src.cli.merge_usr_to_usr", _fake_merge_usr_to_usr)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "maintenance",
            "merge",
            "--dest",
            "demo_dest",
            "--src",
            "demo_src",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert captured["duplicate_policy"] == MergePolicy.ERROR
    assert captured["overlap_coercion"] == "none"


def test_pull_help_mentions_verify_sidecars_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["pull", "--help"])
    assert result.exit_code == 0
    assert "--verify-sidecars" in result.stdout


def test_pull_help_mentions_no_verify_sidecars_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["pull", "--help"])
    assert result.exit_code == 0
    assert "--no-verify-sidecars" in result.stdout


def test_pull_help_defaults_verify_to_hash() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["pull", "--help"])
    assert result.exit_code == 0
    assert "Verification mode:" in result.stdout
    assert "hash|auto|size|parquet" in result.stdout
    assert "[default: hash]" in result.stdout


def test_pull_help_mentions_verify_derived_hashes_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["pull", "--help"])
    assert result.exit_code == 0
    assert "--verify-derived-hashes" in result.stdout


def test_pull_help_mentions_no_verify_derived_hashes_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["pull", "--help"])
    assert result.exit_code == 0
    assert "--no-verify-derived-hashes" in result.stdout


def test_pull_help_mentions_audit_json_output_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["pull", "--help"])
    assert result.exit_code == 0
    assert "--audit-json-out" in result.stdout


def test_push_help_mentions_audit_json_output_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["push", "--help"])
    assert result.exit_code == 0
    assert "--audit-json-out" in result.stdout


def test_root_help_mentions_workflow_map_and_default_sync_contract() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "workflow-map.md" in result.stdout
    assert "verify=hash" in result.stdout
