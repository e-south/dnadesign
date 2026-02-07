"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_notebook_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json

import pytest

from dnadesign.cruncher.app import notebook_service
from dnadesign.cruncher.artifacts.layout import manifest_path


def test_generate_notebook_writes_template(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    analysis_id = "20250101T000000Z_test"
    analysis_dir = run_dir
    analysis_dir.mkdir(parents=True)
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps({"artifacts": [], "config_path": ""}))
    (analysis_dir / "summary.json").write_text(json.dumps({"tf_names": ["LexA"], "analysis_id": analysis_id}))
    (analysis_dir / "plot_manifest.json").write_text(json.dumps({"plots": []}))
    (analysis_dir / "table_manifest.json").write_text(
        json.dumps(
            {
                "tables": [
                    {"key": "scores_summary", "path": "tables/scores_summary.parquet", "exists": True},
                    {"key": "metrics_joint", "path": "tables/metrics_joint.parquet", "exists": True},
                    {"key": "elites_topk", "path": "tables/elites_topk.parquet", "exists": True},
                ]
            }
        )
    )
    (analysis_dir / "tables").mkdir(parents=True, exist_ok=True)
    (analysis_dir / "tables" / "scores_summary.parquet").write_text("placeholder")
    (analysis_dir / "tables" / "metrics_joint.parquet").write_text("placeholder")
    (analysis_dir / "tables" / "elites_topk.parquet").write_text("placeholder")

    monkeypatch.setattr(notebook_service, "ensure_marimo", lambda: None)

    notebook_path = notebook_service.generate_notebook(run_dir, latest=True)
    assert notebook_path.exists()
    content = notebook_path.read_text()
    assert f"default_id_hint = {analysis_id!r}" in content
    assert "Path(__file__).resolve()" in content
    assert "analysis_dir = notebook_path.parent" in content
    assert 'if analysis_dir.parent.name == "_archive"' in content
    assert "run_dir = analysis_dir" in content
    assert "Refresh analysis list" in content
    assert "plot_options" in content
    assert "table_manifest.json" in content
    assert "scores_summary" in content
    assert "metrics_joint" in content
    assert "elites_topk" in content
    assert "Missing JSON at" in content
    assert "per_pwm_scores" not in content
    assert "joint_metrics." not in content
    assert "score_summary." not in content
    assert "elite_topk." not in content
    assert "Text output" in content
    assert "mo.ui.pyplot" not in content
    assert str(run_dir) not in content


def test_generate_notebook_strict_requires_summary(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    analysis_dir = run_dir
    analysis_dir.mkdir(parents=True)
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps({"artifacts": [], "config_path": ""}))

    monkeypatch.setattr(notebook_service, "ensure_marimo", lambda: None)

    with pytest.raises(FileNotFoundError, match="Missing analysis summary"):
        notebook_service.generate_notebook(run_dir, latest=True)


def test_generate_notebook_rejects_lenient_mode(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    analysis_dir = run_dir
    analysis_dir.mkdir(parents=True)
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps({"artifacts": [], "config_path": ""}))
    (analysis_dir / "plot_manifest.json").write_text(json.dumps({"plots": []}))

    monkeypatch.setattr(notebook_service, "ensure_marimo", lambda: None)

    with pytest.raises((TypeError, ValueError), match="strict|lenient"):
        notebook_service.generate_notebook(run_dir, latest=True, strict=False)  # type: ignore[call-arg]


def test_generate_notebook_rejects_latest_and_analysis_id(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(notebook_service, "ensure_marimo", lambda: None)

    with pytest.raises(ValueError, match="analysis-id|latest"):
        notebook_service.generate_notebook(run_dir, analysis_id="20250101T000000Z", latest=True)


def test_generate_notebook_requires_table_manifest_contract_keys(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    analysis_id = "20250101T000000Z_test"
    analysis_dir = run_dir
    analysis_dir.mkdir(parents=True)
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps({"artifacts": [], "config_path": ""}))
    (analysis_dir / "summary.json").write_text(json.dumps({"tf_names": ["LexA"], "analysis_id": analysis_id}))
    (analysis_dir / "plot_manifest.json").write_text(json.dumps({"plots": []}))
    (analysis_dir / "table_manifest.json").write_text(json.dumps({"tables": []}))

    monkeypatch.setattr(notebook_service, "ensure_marimo", lambda: None)

    with pytest.raises(FileNotFoundError, match="analysis table manifest missing required table keys"):
        notebook_service.generate_notebook(run_dir, latest=True)
