"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/publish.py

Publish analysis manifests, reports, and summary payloads for a completed run.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign.cruncher.analysis.layout import (
    analysis_manifest_path,
    plot_manifest_path,
    report_json_path,
    report_md_path,
    summary_path,
    table_manifest_path,
)
from dnadesign.cruncher.analysis.report import build_report_payload, write_report_json, write_report_md
from dnadesign.cruncher.app.analyze.manifests import build_analysis_manifests
from dnadesign.cruncher.app.analyze.outputs import (
    build_summary_payload,
    build_table_artifacts,
    build_table_entries,
)
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.artifacts.entries import artifact_entry

__all__ = ["publish_analysis_outputs"]


def publish_analysis_outputs(
    *,
    analysis_id: str,
    created_at: str,
    run_name: str,
    analysis_root_path: Path,
    tmp_root: Path,
    run_dir: Path,
    analysis_used_file: Path,
    analysis_cfg_payload: dict[str, object],
    tf_names: list[str],
    diagnostics_payload: dict[str, object],
    objective_components: dict[str, object],
    overlap_summary: dict[str, object],
    table_paths: dict[str, Path],
    mmr_sweep_enabled: bool,
    plot_entries: list[dict[str, object]],
    plot_artifacts: list[dict[str, object]],
    version: str,
) -> list[dict[str, Any]]:
    table_entries = build_table_entries(
        table_paths=table_paths,
        mmr_sweep_enabled=mmr_sweep_enabled,
    )
    analysis_artifacts = build_table_artifacts(
        analysis_root_path=analysis_root_path,
        tmp_root=tmp_root,
        run_dir=run_dir,
        table_paths=table_paths,
        mmr_sweep_enabled=mmr_sweep_enabled,
    )
    analysis_artifacts += plot_artifacts

    build_analysis_manifests(
        analysis_id=analysis_id,
        created_at=created_at,
        analysis_root=tmp_root,
        analysis_used_file=analysis_used_file,
        plot_entries=plot_entries,
        table_entries=table_entries,
        analysis_artifacts=analysis_artifacts,
    )
    report_json = report_json_path(tmp_root)
    report_md = report_md_path(tmp_root)
    report_payload = build_report_payload(
        analysis_root=tmp_root,
        summary_payload={
            "analysis_id": analysis_id,
            "run": run_name,
            "tf_names": tf_names,
            "diagnostics": diagnostics_payload,
            "objective_components": objective_components,
            "overlap_summary": overlap_summary,
        },
        diagnostics_payload=diagnostics_payload,
        objective_components=objective_components,
        overlap_summary=overlap_summary,
        analysis_used_payload={"analysis": analysis_cfg_payload},
    )
    write_report_json(report_json, report_payload)
    write_report_md(report_md, report_payload, analysis_root=tmp_root)
    analysis_artifacts.append(
        artifact_entry(report_json_path(analysis_root_path), run_dir, kind="meta", stage="analysis")
    )
    analysis_artifacts.append(
        artifact_entry(report_md_path(analysis_root_path), run_dir, kind="meta", stage="analysis")
    )

    build_analysis_manifests(
        analysis_id=analysis_id,
        created_at=created_at,
        analysis_root=tmp_root,
        analysis_used_file=analysis_used_file,
        plot_entries=plot_entries,
        table_entries=table_entries,
        analysis_artifacts=analysis_artifacts,
    )
    analysis_artifacts.append(
        artifact_entry(plot_manifest_path(analysis_root_path), run_dir, kind="meta", stage="analysis")
    )
    analysis_artifacts.append(
        artifact_entry(table_manifest_path(analysis_root_path), run_dir, kind="meta", stage="analysis")
    )
    analysis_artifacts.append(
        artifact_entry(analysis_manifest_path(analysis_root_path), run_dir, kind="meta", stage="analysis")
    )

    summary_payload = build_summary_payload(
        analysis_id=analysis_id,
        run_name=run_name,
        created_at=created_at,
        analysis_root_path=analysis_root_path,
        run_dir=run_dir,
        tf_names=tf_names,
        diagnostics_payload=diagnostics_payload,
        objective_components=objective_components,
        overlap_summary=overlap_summary,
        artifacts=analysis_artifacts,
        version=version,
    )

    summary_file = summary_path(tmp_root)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_file, summary_payload)
    return analysis_artifacts
