"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/study/test_study_run_smoke.py

Smoke test for end-to-end Study execution with aggregate outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dnadesign.cruncher.app.study_workflow import run_study
from dnadesign.cruncher.study.layout import (
    study_log_path,
    study_manifest_path,
    study_plot_path,
    study_status_path,
    study_table_path,
)
from dnadesign.cruncher.study.manifest import load_study_manifest, load_study_status
from dnadesign.cruncher.tests.study._helpers import write_study_spec, write_workspace_config
from dnadesign.cruncher.utils.paths import resolve_run_index_path


def test_study_run_smoke(tmp_path: Path) -> None:
    config_path = write_workspace_config(tmp_path)
    spec_path = tmp_path / "smoke.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=True,
        seeds=[11, 12],
        trials=[
            {"id": "L6", "factors": {"sample.sequence_length": 6}},
            {"id": "L7", "factors": {"sample.sequence_length": 7}},
        ],
    )

    run_dir = run_study(spec_path)
    assert run_dir.exists()
    assert study_manifest_path(run_dir).exists()
    assert study_status_path(run_dir).exists()

    manifest = load_study_manifest(study_manifest_path(run_dir))
    status = load_study_status(study_status_path(run_dir))
    assert len(manifest.trial_runs) == 4
    assert status.success_runs == 4
    assert status.error_runs == 0
    assert status.status == "completed"
    assert all(item.status == "success" for item in manifest.trial_runs)

    expected_tables = [
        study_table_path(run_dir, "trial_runs"),
        study_table_path(run_dir, "trial_metrics"),
        study_table_path(run_dir, "trial_metrics_agg"),
        study_table_path(run_dir, "mmr_tradeoff_agg"),
        study_table_path(run_dir, "length_tradeoff_agg"),
    ]
    for table_path in expected_tables:
        assert table_path.exists(), f"missing table: {table_path}"

    expected_plots = [
        study_plot_path(run_dir, "mmr_diversity_tradeoff"),
        study_plot_path(run_dir, "sequence_length_tradeoff"),
    ]
    for plot_path in expected_plots:
        assert plot_path.exists(), f"missing plot: {plot_path}"

    mmr_agg = pd.read_parquet(study_table_path(run_dir, "mmr_tradeoff_agg"))
    assert "target_set_index" in mmr_agg.columns
    length_agg = pd.read_parquet(study_table_path(run_dir, "length_tradeoff_agg"))
    assert sorted(int(value) for value in length_agg["sequence_length"].dropna().unique().tolist()) == [6, 7]

    # Study trials should not pollute the workspace run index.
    assert not resolve_run_index_path(config_path).exists()

    log_text = study_log_path(run_dir).read_text(encoding="utf-8")
    assert "REPLAY_START" in log_text
    assert "REPLAY_DONE" in log_text
    assert "SUMMARIZE_START" in log_text
    assert "SUMMARIZE_DONE" in log_text

    # Keep this assert to ensure the run is tied to the workspace under test.
    assert str(config_path.parent.resolve()) in str(run_dir.resolve())


def test_study_run_smoke_allows_empty_trial_factors(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "empty_factors.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=True,
        seeds=[21],
        trials=[{"id": "BASE", "factors": {}}],
    )

    run_dir = run_study(spec_path)
    trial_runs_df = pd.read_parquet(study_table_path(run_dir, "trial_runs"))
    trial_metrics_df = pd.read_parquet(study_table_path(run_dir, "trial_metrics"))

    assert len(trial_runs_df) == 1
    assert trial_runs_df["trial_id"].tolist() == ["BASE"]
    assert len(trial_metrics_df) == 1


def test_study_run_smoke_parallel_trials(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "parallel.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=False,
        seeds=[11, 12],
        trials=[
            {"id": "L6", "factors": {"sample.sequence_length": 6}},
            {"id": "L7", "factors": {"sample.sequence_length": 7}},
        ],
        parallelism=2,
    )

    run_dir = run_study(spec_path, progress_bar=False, quiet_logs=True)
    manifest = load_study_manifest(study_manifest_path(run_dir))
    status = load_study_status(study_status_path(run_dir))

    assert len(manifest.trial_runs) == 4
    assert status.success_runs == 4
    assert status.error_runs == 0
    assert all(item.status == "success" for item in manifest.trial_runs)
    assert all(item.run_dir for item in manifest.trial_runs)


def test_study_run_parallel_logs_progress(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "parallel_progress.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=False,
        seeds=[11],
        trials=[
            {"id": "L6", "factors": {"sample.sequence_length": 6}},
            {"id": "L7", "factors": {"sample.sequence_length": 7}},
        ],
        parallelism=2,
    )

    caplog.set_level("INFO", logger="dnadesign.cruncher.app.study_workflow")
    run_study(spec_path, progress_bar=False, quiet_logs=True)
    assert any("Study trial progress:" in record.message for record in caplog.records)
