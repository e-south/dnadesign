"""Configured OPAL plot review and quality checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from dnadesign.opal import load_plot_artifact_manifest, load_plot_manifest_index

from ..artifacts import ProbeArtifactLayout


def _build_configured_plot_reviews(
    layout: ProbeArtifactLayout,
    *,
    metrics_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reviewed: list[dict[str, Any]] = []
    expected_rounds = {
        str(row.get("run_key")): int(row.get("as_of_round"))
        for row in metrics_payload.get("runs") or []
        if isinstance(row, Mapping) and row.get("run_key") and row.get("as_of_round") is not None
    }
    seen: set[str] = set()
    for row in metrics_payload.get("runs") or []:
        if not isinstance(row, Mapping):
            continue
        run_key = str(row.get("run_key") or "").strip()
        if not run_key or run_key in seen:
            continue
        seen.add(run_key)
        plots_dir = layout.campaign_workdir(run_key) / "outputs" / "plots"
        index_path = plots_dir / "plot_manifest.json"
        entry: dict[str, Any] = {
            "run_key": run_key,
            "plots_dir": str(plots_dir),
            "index_path": str(index_path),
            "expected_final_round": expected_rounds.get(run_key),
            "status": "missing_index",
            "plot_count": 0,
            "plots": [],
            "quality": {"status": "missing", "problems": [f"configured_plot_index_missing:{run_key}"]},
        }
        if not index_path.exists():
            reviewed.append(entry)
            continue
        try:
            index = load_plot_manifest_index(index_path)
            plots = [_configured_plot_entry(plot_row) for plot_row in index.get("manifests") or []]
            entry.update(
                {
                    "status": "loaded",
                    "plot_count": len(plots),
                    "generated_at": index.get("generated_at"),
                    "plots": plots,
                }
            )
            entry["quality"] = _quality_for_configured_plot_entry(entry)
        except Exception as exc:
            entry.update(
                {
                    "status": "error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "quality": {
                        "status": "error",
                        "problems": [f"plot_manifest_error:{type(exc).__name__}:{exc}"],
                    },
                }
            )
        reviewed.append(entry)
    return reviewed


def _configured_plot_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = row.get("manifest_path")
    if manifest_path and Path(str(manifest_path)).exists():
        manifest = load_plot_artifact_manifest(str(manifest_path))
    else:
        manifest = dict(row)
    outputs = [dict(output) for output in manifest.get("outputs") or [] if isinstance(output, Mapping)]
    media = [output for output in outputs if output.get("role") == "media"]
    tidy = [output for output in outputs if output.get("role") == "tidy_csv"]
    return {
        "name": manifest.get("name"),
        "kind": manifest.get("kind"),
        "status": manifest.get("status"),
        "generated_at": manifest.get("generated_at"),
        "run_id": manifest.get("run_id"),
        "rounds": manifest.get("rounds"),
        "manifest_path": manifest.get("manifest_path") or manifest_path,
        "media_paths": [str(output.get("path")) for output in media if output.get("path")],
        "tidy_csv_paths": [str(output.get("path")) for output in tidy if output.get("path")],
        "params": manifest.get("params") or {},
        "warnings": manifest.get("warnings") or [],
        "error": manifest.get("error"),
    }


def _quality_for_configured_plot_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    expected_final_round = entry.get("expected_final_round")
    expected_rounds = set(range(int(expected_final_round) + 1)) if expected_final_round is not None else set()
    for plot in entry.get("plots") or []:
        if not isinstance(plot, Mapping):
            problems.append("plot_entry_not_mapping")
            continue
        name = str(plot.get("name") or "unknown")
        if plot.get("status") != "written":
            problems.append(f"{name}:status_not_written")
        media_paths = [Path(str(path)) for path in plot.get("media_paths") or []]
        if not media_paths:
            problems.append(f"{name}:media_missing")
        for media_path in media_paths:
            problems.extend(_image_quality_problems(media_path, label=name))
        tidy_paths = [Path(str(path)) for path in plot.get("tidy_csv_paths") or []]
        if not tidy_paths:
            problems.append(f"{name}:tidy_csv_missing")
        for tidy_path in tidy_paths:
            problems.extend(
                _tidy_csv_quality_problems(
                    tidy_path,
                    label=name,
                    kind=str(plot.get("kind") or ""),
                    expected_rounds=expected_rounds,
                )
            )
    return {
        "status": "ok" if not problems else "attention",
        "problems": problems,
    }


def _plot_quality_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    problems = [
        {"run_key": entry.get("run_key"), "problem": problem}
        for entry in entries
        for problem in ((entry.get("quality") or {}).get("problems") or [])
    ]
    loaded = [entry for entry in entries if entry.get("status") == "loaded"]
    return {
        "status": "ok" if not problems else "attention",
        "campaigns_with_plot_index": len(loaded),
        "campaigns_expected": len(entries),
        "plot_count": sum(int(entry.get("plot_count") or 0) for entry in loaded),
        "problem_count": len(problems),
        "problems": problems,
    }


def _review_next_steps(*, layout: ProbeArtifactLayout, plot_quality: Mapping[str, Any]) -> dict[str, str]:
    if plot_quality.get("status") == "ok":
        return {}
    module = "dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe"
    run_root = str(layout.run_root)
    return {
        "configured_plot_refresh_command": (f"uv run python -m {module} plot --run-root {run_root} --round all --json"),
        "rerun_report_command": f"uv run python -m {module} report --run-root {run_root} --plots --json",
    }


def _image_quality_problems(path: Path, *, label: str) -> list[str]:
    if not path.exists():
        return [f"{label}:media_file_missing:{path.name}"]
    if path.stat().st_size <= 0:
        return [f"{label}:media_file_empty:{path.name}"]
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            extrema = image.convert("RGB").getextrema()
    except Exception as exc:
        return [f"{label}:media_unreadable:{type(exc).__name__}:{path.name}"]
    problems = []
    if width < 200 or height < 160:
        problems.append(f"{label}:media_too_small:{width}x{height}")
    if all(low == high for low, high in extrema):
        problems.append(f"{label}:media_blank:{path.name}")
    return problems


def _tidy_csv_quality_problems(
    path: Path,
    *,
    label: str,
    kind: str,
    expected_rounds: set[int],
) -> list[str]:
    if not path.exists():
        return [f"{label}:tidy_csv_file_missing:{path.name}"]
    if path.stat().st_size <= 0:
        return [f"{label}:tidy_csv_file_empty:{path.name}"]
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return [f"{label}:tidy_csv_unreadable:{type(exc).__name__}:{path.name}"]
    problems = []
    if frame.empty:
        problems.append(f"{label}:tidy_csv_empty")
        return problems
    if expected_rounds and "round" in frame.columns:
        rounds = {int(value) for value in pd.to_numeric(frame["round"], errors="coerce").dropna().astype(int).tolist()}
        missing = sorted(expected_rounds - rounds)
        if missing:
            problems.append(f"{label}:tidy_csv_missing_rounds:{','.join(map(str, missing))}")
    if kind == "vector_summary_heatmap" and "row_type" in frame.columns:
        row_types = set(frame["row_type"].astype(str))
        if not ({"reference_vector", "setpoint"} & row_types):
            problems.append(f"{label}:tidy_csv_missing_reference_vector")
    if kind == "feature_importance_heatmap" and "feature_id" in frame.columns:
        if frame["feature_id"].nunique(dropna=True) <= 0:
            problems.append(f"{label}:tidy_csv_no_features")
    return problems
