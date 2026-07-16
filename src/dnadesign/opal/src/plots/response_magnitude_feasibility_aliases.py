"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/response_magnitude_feasibility_aliases.py

Candidate display aliases and collision-safe labels for RMF plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.utils import ExitCodes, OpalError

_ALIAS_COLUMNS = (
    "id",
    "usr_label__primary",
    "usr_label__aliases",
    "densegen__plan",
)
_PLAN_PATTERN = re.compile(r"^(background_only|ethanol|ciprofloxacin|ethanol_ciprofloxacin)__sig35=([A-Za-z0-9.-]+)$")
_PLAN_FAMILY_LABELS = {
    "background_only": "Background",
    "ethanol": "EtOH",
    "ciprofloxacin": "Cipro",
    "ethanol_ciprofloxacin": "EtOH + Cipro",
}


def resolve_candidate_display_aliases(
    records_path: str | Path,
    candidate_ids: Sequence[str],
) -> dict[str, str]:
    """Resolve compact, unique display labels without loading the model feature column."""

    path = Path(records_path)
    ids = [_required_id(value) for value in candidate_ids]
    if not ids:
        raise OpalError("Candidate display aliases require at least one candidate ID.", ExitCodes.CONTRACT_VIOLATION)
    if len(ids) != len(set(ids)):
        raise OpalError("Candidate display alias IDs must be unique.", ExitCodes.CONTRACT_VIOLATION)
    if not path.is_file():
        raise OpalError(f"Candidate alias records.parquet not found: {path}", ExitCodes.CONTRACT_VIOLATION)
    try:
        frame = pd.read_parquet(path, columns=list(_ALIAS_COLUMNS))
    except Exception as exc:
        try:
            available = set(pd.read_parquet(path, columns=[]).columns)
        except Exception:
            available = set()
        missing = sorted(set(_ALIAS_COLUMNS) - available)
        if missing:
            raise OpalError(
                f"records.parquet is missing required alias columns: {missing}.",
                ExitCodes.CONTRACT_VIOLATION,
            ) from exc
        raise OpalError(f"Failed to read candidate aliases from records.parquet: {exc}") from exc

    frame["id"] = frame["id"].map(_required_id)
    frame = frame.loc[frame["id"].isin(ids)].copy()
    duplicate_ids = sorted(frame.loc[frame["id"].duplicated(keep=False), "id"].unique().tolist())
    if duplicate_ids:
        raise OpalError(
            f"records.parquet contains duplicate requested candidate IDs: {duplicate_ids[:5]}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    missing_ids = sorted(set(ids) - set(frame["id"]))
    if missing_ids:
        raise OpalError(
            f"records.parquet is missing requested candidate IDs: {missing_ids[:5]}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    by_id = frame.set_index("id", drop=False)
    alias_lists = {candidate_id: _aliases(by_id.at[candidate_id, "usr_label__aliases"]) for candidate_id in ids}
    alias_owners = Counter(alias for values in alias_lists.values() for alias in set(values))

    labels: dict[str, str] = {}
    for candidate_id in ids:
        row = by_id.loc[candidate_id]
        primary = _clean_text(row["usr_label__primary"])
        safe_alias = next((alias for alias in alias_lists[candidate_id] if alias_owners[alias] == 1), None)
        plan_label = _plan_label(row["densegen__plan"])
        if primary:
            label = primary
        elif safe_alias:
            label = safe_alias
        elif plan_label:
            label = f"{plan_label} · {candidate_id[:6]}"
        else:
            label = short_candidate_id(candidate_id)
        labels[candidate_id] = label

    duplicate_labels = {label for label, count in Counter(labels.values()).items() if count > 1}
    return {
        candidate_id: (f"{label} · {candidate_id[:6]}" if label in duplicate_labels else label)
        for candidate_id, label in labels.items()
    }


def annotate_candidate_aliases(
    ax: Any,
    frame: pd.DataFrame,
    aliases: Mapping[str, str],
    *,
    x_column: str,
    y_column: str,
    font_size: float = 7.0,
) -> list[Any]:
    """Place selected aliases in one collision-free lane inside the plot axes."""

    required = {"id", x_column, y_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise OpalError(f"RMF annotation frame is missing columns: {missing}.", ExitCodes.CONTRACT_VIOLATION)
    if frame.empty:
        raise OpalError("RMF alias annotation requires selected candidates.", ExitCodes.CONTRACT_VIOLATION)
    ids = frame["id"].astype(str).tolist()
    missing_aliases = sorted(set(ids) - set(aliases))
    if missing_aliases:
        raise OpalError(
            f"RMF alias annotation is missing candidate labels: {missing_aliases[:5]}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    coordinates = frame[[x_column, y_column]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise OpalError("RMF alias annotation coordinates must be finite.", ExitCodes.CONTRACT_VIOLATION)

    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer=renderer)
    point_pixels = ax.transData.transform(coordinates)
    use_right_lane = float(np.median(point_pixels[:, 0])) <= float(axes_box.x0 + axes_box.x1) / 2.0
    horizontal_alignment = "left" if use_right_lane else "right"
    annotations: list[Any] = []
    for candidate_id, (x_value, y_value) in zip(ids, coordinates, strict=True):
        annotation = ax.annotate(
            aliases[candidate_id],
            xy=(x_value, y_value),
            xytext=(x_value, y_value),
            xycoords="data",
            textcoords="data",
            ha=horizontal_alignment,
            va="center",
            fontsize=font_size,
            color="#252525",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.55, "shrinkA": 2, "shrinkB": 4},
            annotation_clip=True,
            zorder=6,
        )
        annotation.set_clip_on(True)
        annotation.set_clip_path(ax.patch)
        if annotation.arrow_patch is not None:
            annotation.arrow_patch.set_clip_on(True)
            annotation.arrow_patch.set_clip_path(ax.patch)
        annotations.append(annotation)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    text_boxes = [annotation.get_bbox_patch().get_window_extent(renderer=renderer) for annotation in annotations]
    heights = [box.height for box in text_boxes]
    desired_y = point_pixels[:, 1].tolist()
    padding = 4.0
    centers = _spread_centers(
        desired_y,
        heights,
        lower=float(axes_box.y0 + padding),
        upper=float(axes_box.y1 - padding),
        gap=3.0,
    )
    inverse = ax.transData.inverted()
    for index, annotation in enumerate(annotations):
        width = text_boxes[index].width
        point_x = float(point_pixels[index, 0])
        if use_right_lane:
            anchor_x = min(point_x + 8.0, float(axes_box.x1 - padding - width))
            anchor_x = max(anchor_x, float(axes_box.x0 + padding))
        else:
            anchor_x = max(point_x - 8.0, float(axes_box.x0 + padding + width))
            anchor_x = min(anchor_x, float(axes_box.x1 - padding))
        annotation.set_position(tuple(inverse.transform((anchor_x, centers[index]))))
    figure.canvas.draw()
    return annotations


def short_candidate_id(candidate_id: str) -> str:
    value = _required_id(candidate_id)
    return value if len(value) <= 14 else f"{value[:6]}…{value[-5:]}"


def _spread_centers(
    desired: Sequence[float],
    heights: Sequence[float],
    *,
    lower: float,
    upper: float,
    gap: float,
) -> list[float]:
    order = sorted(range(len(desired)), key=lambda index: (float(desired[index]), index))
    required_height = sum(float(heights[index]) for index in order) + gap * max(0, len(order) - 1)
    if required_height > upper - lower:
        raise OpalError(
            "RMF selected-candidate aliases cannot fit inside the plot axes.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    centers_by_index: dict[int, float] = {}
    cursor = lower
    for index in order:
        half_height = float(heights[index]) / 2.0
        center = max(float(desired[index]), cursor + half_height)
        centers_by_index[index] = center
        cursor = center + half_height + gap
    last_index = order[-1]
    overflow = centers_by_index[last_index] + float(heights[last_index]) / 2.0 - upper
    if overflow > 0.0:
        for index in order:
            centers_by_index[index] -= overflow
    return [centers_by_index[index] for index in range(len(desired))]


def _aliases(value: object) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise OpalError("usr_label__aliases values must be lists or null.", ExitCodes.CONTRACT_VIOLATION)
    return tuple(dict.fromkeys(label for item in value if (label := _clean_text(item))))


def _plan_label(value: object) -> str | None:
    plan = _clean_text(value)
    if not plan:
        return None
    match = _PLAN_PATTERN.fullmatch(plan)
    if match is None:
        return None
    family, signature = match.groups()
    return f"{_PLAN_FAMILY_LABELS[family]} · σ35{signature}"


def _clean_text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = " ".join(str(value).split()).strip()
    if not text:
        return None
    return text if len(text) <= 36 else f"{text[:35].rstrip()}…"


def _required_id(value: object) -> str:
    candidate_id = str(value).strip()
    if not candidate_id or candidate_id.lower() in {"nan", "none"}:
        raise OpalError("Candidate IDs must be non-empty strings.", ExitCodes.CONTRACT_VIOLATION)
    return candidate_id


__all__ = ["annotate_candidate_aliases", "resolve_candidate_display_aliases", "short_candidate_id"]
