"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/outputs.py

Build analysis table entries, artifact entries, and summary payloads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign.cruncher.analysis.layout import (
    analysis_used_path,
    plot_manifest_path,
    report_json_path,
    report_md_path,
    table_manifest_path,
)
from dnadesign.cruncher.artifacts.entries import artifact_entry

_TABLE_ORDER = [
    "scores_summary",
    "elites_topk",
    "metrics_joint",
    "chain_trajectory_points",
    "chain_trajectory_lines",
    "overlap_pair_summary",
    "overlap_per_elite",
    "diagnostics_summary",
    "objective_components",
    "elites_mmr_summary",
    "elites_mmr_sweep",
    "elites_nn_distance",
]

_TABLE_LABELS = {
    "scores_summary": "Per-TF summary",
    "elites_topk": "Elite top-K",
    "metrics_joint": "Joint score metrics",
    "chain_trajectory_points": "Chain trajectory points",
    "chain_trajectory_lines": "Chain trajectory lines",
    "overlap_pair_summary": "Overlap pair summary",
    "overlap_per_elite": "Overlap per elite",
    "diagnostics_summary": "Diagnostics summary",
    "objective_components": "Objective components",
    "elites_mmr_summary": "Elites MMR summary",
    "elites_mmr_sweep": "Elites MMR sweep",
    "elites_nn_distance": "Elites NN distance",
}

_TABLE_PURPOSE = {
    "chain_trajectory_points": "plot_support",
    "chain_trajectory_lines": "plot_support",
}


def _require_table_paths(table_paths: dict[str, Path]) -> None:
    missing = [key for key in _TABLE_ORDER if key not in table_paths]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing analysis table paths: {missing_text}")


def build_table_entries(*, table_paths: dict[str, Path], mmr_sweep_enabled: bool) -> list[dict[str, object]]:
    _require_table_paths(table_paths)
    entries: list[dict[str, object]] = []
    for key in _TABLE_ORDER:
        path = table_paths[key]
        entry: dict[str, object] = {
            "key": key,
            "label": _TABLE_LABELS[key],
            "path": path.name,
            "exists": True,
        }
        purpose = _TABLE_PURPOSE.get(key)
        if purpose is not None:
            entry["purpose"] = purpose
        if key == "elites_mmr_sweep" and not mmr_sweep_enabled:
            entry["exists"] = False
            entry["skip_reason"] = "analysis.mmr_sweep.enabled=false"
        entries.append(entry)
    return entries


def _analysis_output_path(analysis_root_path: Path, tmp_root: Path, output_path: Path) -> Path:
    try:
        rel_path = output_path.relative_to(tmp_root)
    except ValueError as exc:
        raise ValueError(f"analysis output path must be under tmp root: {output_path}") from exc
    return analysis_root_path / rel_path


def build_table_artifacts(
    *,
    analysis_root_path: Path,
    tmp_root: Path,
    run_dir: Path,
    table_paths: dict[str, Path],
    mmr_sweep_enabled: bool,
) -> list[dict[str, Any]]:
    _require_table_paths(table_paths)
    keys = [key for key in _TABLE_ORDER if key != "elites_mmr_sweep" or mmr_sweep_enabled]
    artifacts: list[dict[str, Any]] = []
    for key in keys:
        output_path = _analysis_output_path(analysis_root_path, tmp_root, table_paths[key])
        artifacts.append(artifact_entry(output_path, run_dir, kind="table", stage="analysis"))
    return artifacts


def build_summary_payload(
    *,
    analysis_id: str,
    run_name: str,
    created_at: str,
    analysis_root_path: Path,
    run_dir: Path,
    tf_names: list[str],
    diagnostics_payload: dict[str, object],
    objective_components: dict[str, object],
    overlap_summary: dict[str, object],
    artifacts: list[dict[str, Any]],
    version: str,
) -> dict[str, object]:
    return {
        "analysis_id": analysis_id,
        "run": run_name,
        "created_at": created_at,
        "analysis_dir": str(analysis_root_path.resolve()),
        "analysis_used": str(analysis_used_path(analysis_root_path).relative_to(run_dir)),
        "plot_manifest": str(plot_manifest_path(analysis_root_path).relative_to(run_dir)),
        "table_manifest": str(table_manifest_path(analysis_root_path).relative_to(run_dir)),
        "report_json": str(report_json_path(analysis_root_path).relative_to(run_dir)),
        "report_md": str(report_md_path(analysis_root_path).relative_to(run_dir)),
        "tf_names": tf_names,
        "diagnostics": diagnostics_payload,
        "objective_components": objective_components,
        "overlap_summary": overlap_summary,
        "artifacts": artifacts,
        "version": version,
    }
