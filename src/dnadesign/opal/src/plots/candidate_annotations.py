"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/candidate_annotations.py

Candidate display aliases and collision-safe labels for OPAL plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.utils import ExitCodes, OpalError

_ID_COLUMN = "id"
_OPTIONAL_ALIAS_COLUMNS = (
    "usr_label__primary",
    "usr_label__aliases",
)


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
        import pyarrow.parquet as pq

        available = set(pq.read_schema(path).names)
    except Exception as exc:
        raise OpalError(f"Failed to read candidate-table schema from records.parquet: {exc}") from exc
    if _ID_COLUMN not in available:
        raise OpalError("records.parquet is missing required candidate ID column 'id'.", ExitCodes.CONTRACT_VIOLATION)
    projected_columns = [_ID_COLUMN, *(column for column in _OPTIONAL_ALIAS_COLUMNS if column in available)]
    try:
        frame = pd.read_parquet(path, columns=projected_columns)
    except Exception as exc:
        raise OpalError(f"Failed to read candidate aliases from records.parquet: {exc}") from exc
    for column in _OPTIONAL_ALIAS_COLUMNS:
        if column not in frame.columns:
            frame[column] = None

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
        if primary:
            label = primary
        elif safe_alias:
            label = safe_alias
        else:
            label = short_candidate_id(candidate_id)
        labels[candidate_id] = label

    resolved = _collision_safe_labels(list(labels), list(labels.values()))
    return dict(zip(labels, resolved, strict=True))


def annotate_candidate_aliases(
    ax: Any,
    frame: pd.DataFrame,
    aliases: Mapping[str, str],
    *,
    x_column: str,
    y_column: str,
    font_size: float = 7.0,
    max_lanes: int = 1,
) -> list[Any]:
    """Place candidate aliases in one or two collision-free lanes inside the plot axes."""

    required = {"id", x_column, y_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise OpalError(f"Candidate annotation frame is missing columns: {missing}.", ExitCodes.CONTRACT_VIOLATION)
    if frame.empty:
        raise OpalError("Candidate annotation requires at least one row.", ExitCodes.CONTRACT_VIOLATION)
    if isinstance(max_lanes, bool) or max_lanes not in {1, 2}:
        raise OpalError("Candidate annotations support max_lanes of 1 or 2.", ExitCodes.CONTRACT_VIOLATION)
    ids = frame["id"].astype(str).tolist()
    missing_aliases = sorted(set(ids) - set(aliases))
    if missing_aliases:
        raise OpalError(
            f"Candidate annotation is missing display labels: {missing_aliases[:5]}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    coordinates = frame[[x_column, y_column]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise OpalError("Candidate annotation coordinates must be finite.", ExitCodes.CONTRACT_VIOLATION)

    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer=renderer)
    point_pixels = ax.transData.transform(coordinates)
    annotations: list[Any] = []
    for candidate_id, (x_value, y_value) in zip(ids, coordinates, strict=True):
        annotation = ax.annotate(
            aliases[candidate_id],
            xy=(x_value, y_value),
            xytext=(x_value, y_value),
            xycoords="data",
            textcoords="data",
            ha="left",
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
        annotation.set_in_layout(False)
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
    lower = float(axes_box.y0 + padding)
    upper = float(axes_box.y1 - padding)
    try:
        centers = _spread_centers(desired_y, heights, lower=lower, upper=upper, gap=3.0)
    except OpalError:
        if max_lanes == 1:
            raise
        ordered = sorted(range(len(annotations)), key=lambda index: (float(point_pixels[index, 0]), index))
        split = (len(ordered) + 1) // 2
        lane_specs = ((ordered[:split], False), (ordered[split:], True))
    else:
        use_right_lane = float(np.median(point_pixels[:, 0])) <= float(axes_box.x0 + axes_box.x1) / 2.0
        lane_specs = ((list(range(len(annotations))), use_right_lane),)

    inverse = ax.transData.inverted()
    for lane_indexes, use_right_lane in lane_specs:
        lane_centers = (
            [centers[index] for index in lane_indexes]
            if len(lane_specs) == 1
            else _spread_centers(
                [desired_y[index] for index in lane_indexes],
                [heights[index] for index in lane_indexes],
                lower=lower,
                upper=upper,
                gap=3.0,
            )
        )
        for index, center in zip(lane_indexes, lane_centers, strict=True):
            annotation = annotations[index]
            width = text_boxes[index].width
            point_x = float(point_pixels[index, 0])
            annotation.set_ha("left" if use_right_lane else "right")
            if use_right_lane:
                anchor_x = min(point_x + 8.0, float(axes_box.x1 - padding - width))
                anchor_x = max(anchor_x, float(axes_box.x0 + padding))
            else:
                anchor_x = max(point_x - 8.0, float(axes_box.x0 + padding + width))
                anchor_x = min(anchor_x, float(axes_box.x1 - padding))
            annotation.set_position(tuple(inverse.transform((anchor_x, center))))
    figure.canvas.draw()
    return annotations


def short_candidate_id(candidate_id: str) -> str:
    value = _required_id(candidate_id)
    return value if len(value) <= 14 else f"{value[:6]}…{value[-5:]}"


def observed_candidate_display_labels(
    observed: pd.DataFrame,
    *,
    fallbacks: Mapping[str, str],
) -> pd.Series:
    """Return source labels with deterministic identity fallbacks and disambiguation."""

    required = {"id", "display_label"}
    missing = sorted(required - set(observed.columns))
    if missing:
        raise OpalError(
            f"Observed candidate labels are missing columns: {missing}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    labels = observed["display_label"].astype("string")
    missing_labels = labels.isna() | labels.str.strip().eq("")
    if missing_labels.any():
        labels = labels.where(
            ~missing_labels,
            observed["id"].astype(str).map(fallbacks).astype("string"),
        )
    unresolved = labels.isna() | labels.str.strip().eq("")
    if unresolved.any():
        ids = observed.loc[unresolved, "id"].astype(str).tolist()[:5]
        raise OpalError(
            f"Observed candidate labels lack display fallbacks: {ids}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    resolved = _collision_safe_labels(observed["id"].astype(str).tolist(), labels.astype(str).tolist())
    return pd.Series(resolved, index=observed.index, dtype="string")


def _collision_safe_labels(candidate_ids: Sequence[str], base_labels: Sequence[str]) -> list[str]:
    """Disambiguate distinct identity/label pairs without relying on fixed ID prefixes."""

    pairs = list(zip(candidate_ids, base_labels, strict=True))
    unique_pairs = list(dict.fromkeys(pairs))
    resolved = {pair: pair[1] for pair in unique_pairs}
    duplicate_bases = {label for label, count in Counter(resolved.values()).items() if count > 1}
    if duplicate_bases:
        for pair in unique_pairs:
            candidate_id, label = pair
            if label in duplicate_bases:
                resolved[pair] = f"{label} · {short_candidate_id(candidate_id)}"
    duplicate_short = {label for label, count in Counter(resolved.values()).items() if count > 1}
    if duplicate_short:
        for pair in unique_pairs:
            candidate_id, _label = pair
            if resolved[pair] in duplicate_short:
                resolved[pair] = f"{pair[1]} · {candidate_id}"
    if len(set(resolved.values())) != len(resolved):
        raise OpalError(
            "Candidate display labels could not be made unique from exact candidate IDs.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return [resolved[pair] for pair in pairs]


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
            "Candidate annotations cannot fit inside the plot axes.",
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


def _clean_text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = " ".join(str(value).split()).strip()
    if not text:
        return None
    return text if len(text) <= 36 else f"{text[:35].rstrip()}…"


def _required_id(value: object) -> str:
    if not isinstance(value, str):
        raise OpalError("Candidate IDs must be exact strings, without coercion.", ExitCodes.CONTRACT_VIOLATION)
    candidate_id = value
    if candidate_id != candidate_id.strip():
        raise OpalError(
            "Candidate IDs must not contain leading or trailing whitespace.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if not candidate_id or candidate_id.lower() in {"nan", "none"}:
        raise OpalError("Candidate IDs must be non-empty strings.", ExitCodes.CONTRACT_VIOLATION)
    return candidate_id


__all__ = [
    "annotate_candidate_aliases",
    "observed_candidate_display_labels",
    "resolve_candidate_display_aliases",
    "short_candidate_id",
]
