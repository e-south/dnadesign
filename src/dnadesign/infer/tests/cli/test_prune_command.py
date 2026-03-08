"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/cli/test_prune_command.py

CLI contract tests for infer namespace prune workflows against USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.infer.cli import app
from dnadesign.infer.src.writers.usr import write_back_usr
from dnadesign.usr import Dataset
from dnadesign.usr.tests.registry_helpers import register_test_namespace

_RUNNER = CliRunner()


def test_prune_usr_archives_infer_overlay_by_default(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"
    register_test_namespace(
        root,
        namespace="infer",
        columns_spec="infer__evo2_7b__job_a__ll_mean:float64",
        overwrite=True,
    )
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [{"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"}],
        source="unit",
    )
    row_id = ds.head(1, columns=["id"])["id"].tolist()
    write_back_usr(
        ds,
        ids=row_id,
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.0]},
        overwrite=False,
    )

    result = _RUNNER.invoke(app, ["prune", "--usr", "demo", "--usr-root", root.as_posix()])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "dataset: demo" in output
    assert "namespace: infer" in output
    assert "mode: archive" in output
    assert "removed: True" in output
    assert "archived_path:" in output
    assert [overlay.namespace for overlay in ds.list_overlays()] == []


def test_prune_usr_fails_fast_when_infer_overlay_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"
    register_test_namespace(
        root,
        namespace="infer",
        columns_spec="infer__evo2_7b__job_a__ll_mean:float64",
        overwrite=True,
    )
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")

    result = _RUNNER.invoke(app, ["prune", "--usr", "demo", "--usr-root", root.as_posix()])

    assert result.exit_code == 2
    assert "Overlay 'infer' not found." in (result.stdout or "")
