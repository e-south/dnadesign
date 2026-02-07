"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_output_layout_contract.py

Asserts the run output layout remains flat, slot-based, and easy to navigate.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.analysis.layout import analysis_root
from dnadesign.cruncher.artifacts.layout import (
    build_run_dir,
    config_used_path,
    elites_hits_path,
    elites_json_path,
    elites_mmr_meta_path,
    elites_path,
    elites_yaml_path,
    live_metrics_path,
    manifest_path,
    random_baseline_hits_path,
    random_baseline_path,
    sequences_path,
    status_path,
    trace_path,
)


def test_build_run_dir_uses_latest_slot_without_opaque_hash(tmp_path: Path) -> None:
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

    assert run_dir == tmp_path / "outputs" / "sample" / "latest"


def test_build_run_dir_with_multiple_sets_uses_set_folder_and_slot(tmp_path: Path) -> None:
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

    assert run_dir == tmp_path / "outputs" / "sample" / "set2_lexA-cpxR" / "latest"


def test_run_artifacts_live_at_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "sample" / "latest"

    assert manifest_path(run_dir) == run_dir / "run_manifest.json"
    assert status_path(run_dir) == run_dir / "run_status.json"
    assert config_used_path(run_dir) == run_dir / "config_used.yaml"
    assert live_metrics_path(run_dir) == run_dir / "metrics.jsonl"
    assert trace_path(run_dir) == run_dir / "trace.nc"
    assert sequences_path(run_dir) == run_dir / "sequences.parquet"
    assert random_baseline_path(run_dir) == run_dir / "random_baseline.parquet"
    assert random_baseline_hits_path(run_dir) == run_dir / "random_baseline_hits.parquet"
    assert elites_path(run_dir) == run_dir / "elites.parquet"
    assert elites_hits_path(run_dir) == run_dir / "elites_hits.parquet"
    assert elites_mmr_meta_path(run_dir) == run_dir / "elites_mmr_meta.parquet"
    assert elites_json_path(run_dir) == run_dir / "elites.json"
    assert elites_yaml_path(run_dir) == run_dir / "elites.yaml"


def test_analysis_root_is_run_root_for_flat_access(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "sample" / "latest"
    assert analysis_root(run_dir) == run_dir
