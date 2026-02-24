"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_analyze_outputs.py

Tests for analysis output payload and artifact helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.analyze.outputs import (
    build_summary_payload,
    build_table_artifacts,
    build_table_entries,
)


def _table_paths(tmp_root: Path) -> dict[str, Path]:
    tables_root = tmp_root / "tables"
    return {
        "scores_summary": tables_root / "table__scores_summary.parquet",
        "elites_topk": tables_root / "table__elites_topk.parquet",
        "metrics_joint": tables_root / "table__metrics_joint.parquet",
        "chain_trajectory_points": tables_root / "table__chain_trajectory_points.parquet",
        "chain_trajectory_lines": tables_root / "table__chain_trajectory_lines.parquet",
        "overlap_pair_summary": tables_root / "table__overlap_pair_summary.parquet",
        "overlap_per_elite": tables_root / "table__overlap_per_elite.parquet",
        "diagnostics_summary": tables_root / "table__diagnostics_summary.parquet",
        "objective_components": tables_root / "table__objective_components.parquet",
        "elites_mmr_summary": tables_root / "table__elites_mmr_summary.parquet",
        "elites_mmr_sweep": tables_root / "table__elites_mmr_sweep.parquet",
        "elites_nn_distance": tables_root / "table__elites_nn_distance.parquet",
    }


def test_build_table_entries_marks_disabled_mmr_sweep() -> None:
    entries = build_table_entries(table_paths=_table_paths(Path("/tmp/analysis")), mmr_sweep_enabled=False)
    mmr_entry = next(entry for entry in entries if entry["key"] == "elites_mmr_sweep")
    assert mmr_entry["exists"] is False
    assert mmr_entry["skip_reason"] == "analysis.mmr_sweep.enabled=false"


def test_build_table_artifacts_skips_disabled_mmr_sweep() -> None:
    run_dir = Path("/tmp/run")
    analysis_root = run_dir / "analysis"
    tmp_root = analysis_root / "state" / "tmp"
    artifacts = build_table_artifacts(
        analysis_root_path=analysis_root,
        tmp_root=tmp_root,
        run_dir=run_dir,
        table_paths=_table_paths(tmp_root),
        mmr_sweep_enabled=False,
    )
    artifact_paths = {entry["path"] for entry in artifacts}
    assert "analysis/tables/table__elites_mmr_sweep.parquet" not in artifact_paths
    assert "analysis/tables/table__chain_trajectory_lines.parquet" in artifact_paths


def test_build_summary_payload_paths_are_run_relative(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    analysis_root = run_dir / "analysis"
    payload = build_summary_payload(
        analysis_id="a1",
        run_name="outputs",
        created_at="2026-02-19T00:00:00+00:00",
        analysis_root_path=analysis_root,
        run_dir=run_dir,
        tf_names=["lexA", "cpxR"],
        diagnostics_payload={"status": "ok"},
        objective_components={"contribution": 1.0},
        overlap_summary={"overlap_total_bp_median": 0.0},
        artifacts=[],
        version="test",
    )
    assert payload["analysis_used"] == "analysis/reports/analysis_used.yaml"
    assert payload["plot_manifest"] == "analysis/manifests/plot_manifest.json"
    assert payload["table_manifest"] == "analysis/manifests/table_manifest.json"
    assert payload["report_json"] == "analysis/reports/report.json"
    assert payload["report_md"] == "analysis/reports/report.md"
