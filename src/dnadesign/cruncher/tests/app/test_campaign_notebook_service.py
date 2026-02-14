"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_campaign_notebook_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json

import pytest

from dnadesign.cruncher.app import campaign_notebook_service


def test_generate_campaign_notebook_reads_output_and_plots_dirs(tmp_path, monkeypatch) -> None:
    summary_dir = tmp_path / "campaign" / "demo"
    output_dir = summary_dir / "analysis"
    plots_dir = summary_dir / "plots"
    output_dir.mkdir(parents=True)
    plots_dir.mkdir(parents=True)
    (output_dir / "campaign_summary.csv").write_text("set_index,joint_hmean\n1,0.9\n")
    (output_dir / "campaign_best.csv").write_text("set_index,joint_hmean\n1,0.95\n")
    (output_dir / "campaign_manifest.json").write_text(json.dumps({"campaign_name": "demo"}))
    (plots_dir / "best_jointscore_bar.png").write_text("placeholder")

    monkeypatch.setattr(campaign_notebook_service, "ensure_marimo", lambda: None)

    notebook_path = campaign_notebook_service.generate_campaign_notebook(summary_dir)
    assert notebook_path.exists()
    content = notebook_path.read_text()
    assert 'data_dir = summary_dir / "analysis"' in content
    assert 'plots_dir = summary_dir / "plots"' in content
    assert 'summary_path = data_dir / "campaign_summary.csv"' in content
    assert 'best_path = data_dir / "campaign_best.csv"' in content
    assert 'manifest_path = data_dir / "campaign_manifest.json"' in content


def test_generate_campaign_notebook_strict_requires_output_contract(tmp_path, monkeypatch) -> None:
    summary_dir = tmp_path / "campaign" / "demo"
    output_dir = summary_dir / "analysis"
    output_dir.mkdir(parents=True)
    (output_dir / "campaign_best.csv").write_text("set_index,joint_hmean\n1,0.95\n")
    (output_dir / "campaign_manifest.json").write_text(json.dumps({"campaign_name": "demo"}))

    monkeypatch.setattr(campaign_notebook_service, "ensure_marimo", lambda: None)

    with pytest.raises(FileNotFoundError, match="campaign_summary\\.csv"):
        campaign_notebook_service.generate_campaign_notebook(summary_dir, strict=True)
