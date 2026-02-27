"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_notebook_export_paths.py

Tests for DenseGen notebook export destination normalization helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.densegen.src.cli.notebook_export_paths import (
    resolve_baserender_export_destination,
    resolve_records_export_destination,
)


def test_resolve_records_export_destination_normalizes_format_suffix_and_scope(tmp_path: Path) -> None:
    run_root = tmp_path
    destination = resolve_records_export_destination(
        raw_path="exports/records_window.csv",
        selected_format="parquet",
        run_root=run_root,
    )
    assert destination == run_root / "exports" / "records_window.parquet"


def test_resolve_records_export_destination_uses_default_stem_when_path_is_empty(tmp_path: Path) -> None:
    destination = resolve_records_export_destination(
        raw_path="",
        selected_format="csv",
        run_root=tmp_path,
    )
    assert destination == tmp_path / "outputs" / "notebooks" / "records_preview.csv"


def test_resolve_records_export_destination_uses_repo_root_for_relative_paths_when_provided(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "src" / "dnadesign" / "densegen" / "workspaces" / "demo"
    destination = resolve_records_export_destination(
        raw_path="exports/records_window.csv",
        selected_format="parquet",
        run_root=run_root,
        repo_root=repo_root,
    )
    assert destination == repo_root / "exports" / "records_window.parquet"


def test_resolve_records_export_destination_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="records export format must be parquet or csv"):
        resolve_records_export_destination(
            raw_path="outputs/notebooks/records_preview.parquet",
            selected_format="json",
            run_root=tmp_path,
        )


def test_resolve_baserender_export_destination_normalizes_format_suffix(tmp_path: Path) -> None:
    destination = resolve_baserender_export_destination(
        raw_path="outputs/notebooks/baserender_preview.png",
        selected_format="pdf",
        run_root=tmp_path,
    )
    assert destination == tmp_path / "outputs" / "notebooks" / "baserender_preview.pdf"


def test_resolve_baserender_export_destination_uses_repo_root_for_relative_paths_when_provided(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "src" / "dnadesign" / "densegen" / "workspaces" / "demo"
    destination = resolve_baserender_export_destination(
        raw_path="exports/baserender_preview.png",
        selected_format="pdf",
        run_root=run_root,
        repo_root=repo_root,
    )
    assert destination == repo_root / "exports" / "baserender_preview.pdf"


def test_resolve_baserender_export_destination_rejects_empty_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BaseRender export path cannot be empty"):
        resolve_baserender_export_destination(
            raw_path="",
            selected_format="png",
            run_root=tmp_path,
        )


def test_resolve_baserender_export_destination_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BaseRender export format must be png or pdf"):
        resolve_baserender_export_destination(
            raw_path=str(Path("/tmp") / "baserender_preview.png"),
            selected_format="svg",
            run_root=tmp_path,
        )
