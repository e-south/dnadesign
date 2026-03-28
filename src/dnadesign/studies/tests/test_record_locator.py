"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_record_locator.py

Focused tests for flat checked-in study index loading and active-study
resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.core.record_locator import discover_active_study_selection
from dnadesign.studies.core.registry import load_study_index


def _write_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path
    (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
    study_root = repo_root / "docs" / "studies" / "demo_study"
    study_root.mkdir(parents=True, exist_ok=True)
    (study_root / "ops.study.yaml").write_text("version: 2\nstudy_id: demo_study\nfamily: promoter\n", encoding="utf-8")
    (repo_root / "docs" / "studies" / "index.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "active_study_id": "demo_study",
                "studies": [
                    {
                        "study_id": "demo_study",
                        "family": "promoter",
                        "title": "Demo study",
                        "record_root": "docs/studies/demo_study",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return repo_root


def test_load_study_index_reads_flat_study_first_layout(tmp_path: Path) -> None:
    repo_root = _write_repo(tmp_path)

    index = load_study_index(repo_root)

    assert index.active_study_id == "demo_study"
    assert index.studies[0].family == "promoter"
    assert index.studies[0].record_root == (repo_root / "docs" / "studies" / "demo_study").resolve()


def test_discover_active_study_selection_uses_top_level_studies_index(tmp_path: Path) -> None:
    repo_root = _write_repo(tmp_path)

    selection = discover_active_study_selection(
        repo_root=repo_root,
        status_kind="promoter-study-status",
    )

    assert selection.active_study_id == "demo_study"
    assert selection.family == "promoter"
    assert selection.index_path == (repo_root / "docs" / "studies" / "index.yaml").resolve()
    assert selection.study_root == (repo_root / "docs" / "studies" / "demo_study").resolve()
