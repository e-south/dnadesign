"""Configured OPAL plot file-level quality checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _plot_requires_tidy_csv(plot: Mapping[str, Any]) -> bool:
    metadata = plot.get("metadata") if isinstance(plot.get("metadata"), Mapping) else {}
    quality = plot.get("quality") if isinstance(plot.get("quality"), Mapping) else {}
    capability = metadata.get("capability") if isinstance(metadata.get("capability"), Mapping) else {}
    return bool(capability.get("tidy_available") or metadata.get("tidy_schema") or quality.get("tidy_schema_declared"))


def _plot_requires_reference_vector(plot: Mapping[str, Any]) -> bool:
    params = plot.get("params") if isinstance(plot.get("params"), Mapping) else {}
    include_reference_vector = params.get("include_reference_vector")
    if isinstance(include_reference_vector, bool):
        return include_reference_vector
    if isinstance(include_reference_vector, str):
        return include_reference_vector.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(include_reference_vector)


def _expected_tidy_rounds_for_plot(rounds: Any, *, expected_final_round: Any) -> set[int] | None:
    if expected_final_round is None:
        return None
    final_round = int(expected_final_round)
    if rounds == "all":
        return set(range(final_round + 1))
    if rounds == "latest":
        return {final_round}
    if isinstance(rounds, list):
        resolved: set[int] = set()
        for value in rounds:
            if value is None:
                continue
            resolved.add(int(value))
        return resolved
    return None


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
    reference_vector_required: bool = False,
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
    if kind == "vector_summary_heatmap" and reference_vector_required and "row_type" in frame.columns:
        row_types = set(frame["row_type"].astype(str))
        if not ({"reference_vector", "setpoint"} & row_types):
            problems.append(f"{label}:tidy_csv_missing_reference_vector")
    if kind == "feature_importance_heatmap" and "feature_id" in frame.columns:
        if frame["feature_id"].nunique(dropna=True) <= 0:
            problems.append(f"{label}:tidy_csv_no_features")
    return problems
