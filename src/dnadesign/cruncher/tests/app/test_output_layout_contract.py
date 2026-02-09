"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_output_layout_contract.py

Asserts the run output layout remains flat, canonical, and easy to navigate.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.analysis.layout import (
    analysis_plot_path,
    analysis_root,
    analysis_table_path,
)
from dnadesign.cruncher.artifacts.layout import (
    build_run_dir,
    config_used_path,
    elites_hits_path,
    elites_json_path,
    elites_mmr_meta_path,
    elites_path,
    elites_yaml_path,
    live_metrics_path,
    lockfile_snapshot_path,
    manifest_path,
    parse_manifest_path,
    pwm_summary_path,
    random_baseline_hits_path,
    random_baseline_path,
    run_input_dir,
    run_optimize_dir,
    sequences_path,
    status_path,
    trace_path,
)


def test_build_run_dir_uses_canonical_root_without_opaque_hash(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher: {}\n")

    run_dir = build_run_dir(
        config_path=config_path,
        out_dir="outputs",
        stage="sample",
        tfs=["lexA", "cpxR"],
        set_index=1,
        include_set_index=False,
    )

    assert run_dir == tmp_path / "outputs"


def test_build_run_dir_parse_stage_uses_workspace_state_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher: {}\n")

    parse_dir = build_run_dir(
        config_path=config_path,
        out_dir="outputs",
        stage="parse",
        tfs=["lexA", "cpxR"],
        set_index=1,
        include_set_index=False,
    )
    sample_dir = build_run_dir(
        config_path=config_path,
        out_dir="outputs",
        stage="sample",
        tfs=["lexA", "cpxR"],
        set_index=1,
        include_set_index=False,
    )

    assert parse_dir == tmp_path / ".cruncher" / "parse"
    assert sample_dir == tmp_path / "outputs"
    assert parse_dir != sample_dir


def test_build_run_dir_with_multiple_sets_uses_set_folder(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher: {}\n")

    run_dir = build_run_dir(
        config_path=config_path,
        out_dir="outputs",
        stage="sample",
        tfs=["lexA", "cpxR"],
        set_index=2,
        include_set_index=True,
    )

    assert run_dir == tmp_path / "outputs" / "set2_lexA-cpxR"


def test_run_artifacts_live_in_lifecycle_subdirs(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs"

    assert run_input_dir(run_dir) == run_dir / "input"
    assert run_optimize_dir(run_dir) == run_dir / "optimize"

    assert manifest_path(run_dir) == run_dir / "run_manifest.json"
    assert status_path(run_dir) == run_dir / "run_status.json"
    assert config_used_path(run_dir) == run_dir / "config_used.yaml"
    assert lockfile_snapshot_path(run_dir) == run_dir / "input" / "lockfile.json"
    assert parse_manifest_path(run_dir) == run_dir / "input" / "parse_manifest.json"
    assert pwm_summary_path(run_dir) == run_dir / "input" / "pwm_summary.json"
    assert live_metrics_path(run_dir) == run_dir / "optimize" / "metrics.jsonl"
    assert trace_path(run_dir) == run_dir / "optimize" / "trace.nc"
    assert sequences_path(run_dir) == run_dir / "optimize" / "sequences.parquet"
    assert random_baseline_path(run_dir) == run_dir / "optimize" / "random_baseline.parquet"
    assert random_baseline_hits_path(run_dir) == run_dir / "optimize" / "random_baseline_hits.parquet"
    assert elites_path(run_dir) == run_dir / "optimize" / "elites.parquet"
    assert elites_hits_path(run_dir) == run_dir / "optimize" / "elites_hits.parquet"
    assert elites_mmr_meta_path(run_dir) == run_dir / "optimize" / "elites_mmr_meta.parquet"
    assert elites_json_path(run_dir) == run_dir / "optimize" / "elites.json"
    assert elites_yaml_path(run_dir) == run_dir / "optimize" / "elites.yaml"


def test_analysis_root_is_run_root_for_flat_access(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs"
    assert analysis_root(run_dir) == run_dir


def test_analysis_tables_and_plots_use_flat_semantic_filenames(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs"
    analysis_dir = analysis_root(run_dir)

    assert analysis_table_path(analysis_dir, "scores_summary", "parquet") == (
        analysis_dir / "analysis" / "table__scores_summary.parquet"
    )
    assert analysis_plot_path(analysis_dir, "opt_trajectory", "png") == (
        analysis_dir / "plots" / "plot__opt_trajectory.png"
    )
    assert analysis_plot_path(analysis_dir, "opt_trajectory_sweep", "png") == (
        analysis_dir / "plots" / "plot__opt_trajectory_sweep.png"
    )
