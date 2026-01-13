"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_notebook_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json

import pytest

from dnadesign.cruncher.services import notebook_service
from dnadesign.cruncher.utils.run_layout import manifest_path


def test_generate_notebook_writes_template(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    analysis_id = "20250101T000000Z_test"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    meta_dir = analysis_dir / "meta"
    meta_dir.mkdir(parents=True)
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps({"artifacts": [], "config_path": ""}))
    (meta_dir / "summary.json").write_text(json.dumps({"tf_names": ["LexA"], "analysis_id": analysis_id}))
    (meta_dir / "plot_manifest.json").write_text(json.dumps({"plots": []}))

    monkeypatch.setattr(notebook_service, "ensure_marimo", lambda: None)

    notebook_path = notebook_service.generate_notebook(run_dir, latest=True)
    assert notebook_path.exists()
    content = notebook_path.read_text()
    assert f"default_id_hint = {analysis_id!r}" in content
    assert "Path(__file__).resolve()" in content
    assert "analysis_root = notebook_path.parent.parent" in content
    assert "Refresh analysis list" in content
    assert "plot_options" in content
    assert "Missing JSON at" in content
    assert "scatter controls disabled" in content
    assert "Text output" in content
    assert "mo.ui.pyplot" not in content
    assert str(run_dir) not in content


def test_generate_notebook_strict_requires_summary(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    analysis_id = "20250101T000000Z_missing"
    analysis_dir = run_dir / "analysis" / analysis_id
    analysis_dir.mkdir(parents=True)
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps({"artifacts": [], "config_path": ""}))

    monkeypatch.setattr(notebook_service, "ensure_marimo", lambda: None)

    try:
        notebook_service.generate_notebook(run_dir, analysis_id=analysis_id, strict=True)
    except FileNotFoundError as exc:
        assert "Missing analysis artifacts" in str(exc)
    else:
        raise AssertionError("Expected strict notebook generation to fail when summary is missing.")


def test_generate_notebook_lenient_allows_missing_summary(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    meta_dir = analysis_dir / "meta"
    meta_dir.mkdir(parents=True)
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps({"artifacts": [], "config_path": ""}))
    (meta_dir / "plot_manifest.json").write_text(json.dumps({"plots": []}))

    monkeypatch.setattr(notebook_service, "ensure_marimo", lambda: None)

    notebook_path = notebook_service.generate_notebook(run_dir, latest=True, strict=False)
    assert notebook_path.exists()
    content = notebook_path.read_text()
    assert "list_analysis_entries_verbose" in content
    assert "analysis_root = notebook_path.parent.parent" in content


def test_generate_notebook_rejects_latest_and_analysis_id(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(notebook_service, "ensure_marimo", lambda: None)

    with pytest.raises(ValueError, match="analysis-id|latest"):
        notebook_service.generate_notebook(run_dir, analysis_id="20250101T000000Z", latest=True)
