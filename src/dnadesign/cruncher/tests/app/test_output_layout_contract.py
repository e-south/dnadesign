"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_output_layout_contract.py

Asserts the run output layout remains canonical, structured, and easy to navigate.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.analysis.layout import analysis_plot_path, analysis_root, analysis_table_path
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
    run_meta_dir,
    run_optimize_dir,
    run_optimize_meta_dir,
    run_optimize_tables_dir,
    run_provenance_dir,
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

    assert run_meta_dir(run_dir) == run_dir / "meta"
    assert run_provenance_dir(run_dir) == run_dir / "provenance"
    assert run_optimize_dir(run_dir) == run_dir / "optimize"
    assert run_optimize_tables_dir(run_dir) == run_dir / "optimize" / "tables"
    assert run_optimize_meta_dir(run_dir) == run_dir / "optimize" / "state"

    assert manifest_path(run_dir) == run_dir / "meta" / "run_manifest.json"
    assert status_path(run_dir) == run_dir / "meta" / "run_status.json"
    assert config_used_path(run_dir) == run_dir / "meta" / "config_used.yaml"
    assert lockfile_snapshot_path(run_dir) == run_dir / "provenance" / "lockfile.json"
    assert parse_manifest_path(run_dir) == run_dir / "provenance" / "parse_manifest.json"
    assert pwm_summary_path(run_dir) == run_dir / "provenance" / "pwm_summary.json"
    assert live_metrics_path(run_dir) == run_dir / "optimize" / "state" / "metrics.jsonl"
    assert trace_path(run_dir) == run_dir / "optimize" / "state" / "trace.nc"
    assert sequences_path(run_dir) == run_dir / "optimize" / "tables" / "sequences.parquet"
    assert random_baseline_path(run_dir) == run_dir / "optimize" / "tables" / "random_baseline.parquet"
    assert random_baseline_hits_path(run_dir) == run_dir / "optimize" / "tables" / "random_baseline_hits.parquet"
    assert elites_path(run_dir) == run_dir / "optimize" / "tables" / "elites.parquet"
    assert elites_hits_path(run_dir) == run_dir / "optimize" / "tables" / "elites_hits.parquet"
    assert elites_mmr_meta_path(run_dir) == run_dir / "optimize" / "tables" / "elites_mmr_meta.parquet"
    assert elites_json_path(run_dir) == run_dir / "optimize" / "state" / "elites.json"
    assert elites_yaml_path(run_dir) == run_dir / "optimize" / "state" / "elites.yaml"


def test_analysis_root_is_run_analysis_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs"
    assert analysis_root(run_dir) == run_dir / "analysis"


def test_analysis_tables_and_plots_use_structured_semantic_filenames(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs"
    analysis_dir = analysis_root(run_dir)

    assert analysis_table_path(analysis_dir, "scores_summary", "parquet") == (
        analysis_dir / "tables" / "table__scores_summary.parquet"
    )
    assert analysis_plot_path(analysis_dir, "opt_trajectory", "png") == (run_dir / "plots" / "opt_trajectory.png")
    assert analysis_plot_path(analysis_dir, "opt_trajectory_sweep", "png") == (
        run_dir / "plots" / "opt_trajectory_sweep.png"
    )
