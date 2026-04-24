"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/cli/test_cli_typer.py

Typer CLI integration tests for USR.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.testsupport.usr import ensure_registry
from dnadesign.usr import pkg_usr_root
from dnadesign.usr.src.cli import app
from dnadesign.usr.src.contracts import SequencesError
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.datasets.merge import MergePolicy, MergePreview

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain_output(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


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


def test_head_rejects_negative_row_count(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "head", "demo", "-n", "-1"])

    assert result.exit_code != 0
    assert isinstance(result.exception, SequencesError)
    assert "head row count" in str(result.exception)


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


def test_public_cli_quickstart_sequence_is_hermetic(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    out_dir = tmp_path / "exports"
    root.mkdir()
    out_dir.mkdir()
    assets = pkg_usr_root() / "assets" / "demo_material"
    dataset = "quickstart_demo"
    runner = CliRunner()

    def invoke_ok(args: list[str]) -> None:
        result = runner.invoke(app, ["--root", str(root), *args])
        assert result.exit_code == 0, result.stdout

    invoke_ok(
        [
            "namespace",
            "register",
            "quickstart",
            "--columns",
            "quickstart__X_value:list<float64>,quickstart__intensity_log2_offset_delta:float64",
        ]
    )
    invoke_ok(["init", dataset, "--source", "test quickstart"])
    invoke_ok(
        [
            "import",
            dataset,
            "--from",
            "csv",
            "--path",
            str(assets / "demo_sequences.csv"),
            "--bio-type",
            "dna",
            "--alphabet",
            "dna_4",
        ]
    )
    invoke_ok(
        [
            "attach",
            dataset,
            "--path",
            str(assets / "demo_attachment_one.csv"),
            "--namespace",
            "quickstart",
            "--key",
            "sequence",
            "--key-col",
            "sequence",
            "--columns",
            "X_value",
        ]
    )
    invoke_ok(
        [
            "attach",
            dataset,
            "--path",
            str(assets / "demo_y_sfxi.csv"),
            "--namespace",
            "quickstart",
            "--key",
            "sequence",
            "--key-col",
            "sequence",
            "--columns",
            "intensity_log2_offset_delta",
            "--allow-missing",
        ]
    )
    invoke_ok(["materialize", dataset, "--yes", "--snapshot-before"])
    invoke_ok(["validate", dataset, "--strict"])
    invoke_ok(["export", dataset, "--fmt", "parquet", "--out", str(out_dir)])

    assert (out_dir / f"{dataset}.parquet").exists()


def test_merge_defaults_are_strict(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    def _fake_merge_usr_to_usr(**kwargs):
        captured["duplicate_policy"] = kwargs["duplicate_policy"]
        captured["overlap_coercion"] = kwargs["overlap_coercion"]
        captured["carry_namespaces"] = kwargs["carry_namespaces"]
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
            "--union-columns",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert captured["duplicate_policy"] == MergePolicy.ERROR
    assert captured["overlap_coercion"] == "none"
    assert captured["carry_namespaces"] == []


def test_merge_passes_explicit_carry_namespaces(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    def _fake_merge_usr_to_usr(**kwargs):
        captured["carry_namespaces"] = kwargs["carry_namespaces"]
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
            "--union-columns",
            "--carry-namespace",
            "usr_label",
            "--carry-namespace",
            "infer",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert captured["carry_namespaces"] == ["usr_label", "infer"]


def test_merge_rejects_conflicting_column_mode_flags(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir(parents=True, exist_ok=True)

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
            "--require-same-columns",
            "--union-columns",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, SequencesError)
    assert "Choose exactly one" in str(result.exception)


def test_merge_requires_explicit_column_mode(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir(parents=True, exist_ok=True)

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

    assert result.exit_code != 0
    assert isinstance(result.exception, SequencesError)
    assert "Choose exactly one" in str(result.exception)


def test_pull_help_mentions_verify_sidecars_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--no-rich", "pull", "--help"])
    assert result.exit_code == 0
    assert "--verify-sidecars" in _plain_output(result.stdout)


def test_pull_help_mentions_no_verify_sidecars_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--no-rich", "pull", "--help"])
    assert result.exit_code == 0
    assert "--no-verify-sidecars" in _plain_output(result.stdout)


def test_pull_help_defaults_verify_to_hash() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--no-rich", "pull", "--help"])
    assert result.exit_code == 0
    stdout = _plain_output(result.stdout)
    assert "Verification mode:" in stdout
    assert "hash|auto|size|parquet" in stdout
    assert "[default: hash]" in stdout


def test_pull_help_mentions_verify_derived_hashes_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--no-rich", "pull", "--help"])
    assert result.exit_code == 0
    assert "--verify-derived-hashes" in _plain_output(result.stdout)


def test_pull_help_mentions_no_verify_derived_hashes_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--no-rich", "pull", "--help"])
    assert result.exit_code == 0
    assert "--no-verify-derived-hashes" in _plain_output(result.stdout)


def test_pull_help_mentions_audit_json_output_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--no-rich", "pull", "--help"])
    assert result.exit_code == 0
    assert "--audit-json-out" in _plain_output(result.stdout)


def test_push_help_mentions_audit_json_output_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--no-rich", "push", "--help"])
    assert result.exit_code == 0
    assert "--audit-json-out" in _plain_output(result.stdout)


def test_diff_help_mentions_audit_json_output_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--no-rich", "diff", "--help"])
    assert result.exit_code == 0
    assert "--audit-json-out" in _plain_output(result.stdout)


def test_root_help_mentions_workflow_map_and_default_sync_contract() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--no-rich", "--help"])
    assert result.exit_code == 0
    stdout = _plain_output(result.stdout)
    assert "workflow-map.md" in stdout
    assert "verify=hash" in stdout
