"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/dense_array_video_source.py

Dense-array video source validation, ordering, and sampling helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from dnadesign.densegen.src.config.plots import PlotVideoConfig
from dnadesign.densegen.src.core.record_values import normalize_used_tfbs_entries

REQUIRED_VIDEO_SOURCE_COLUMNS = (
    "id",
    "sequence",
    "densegen__plan",
    "densegen__used_tfbs_detail",
)
OPTIONAL_VIDEO_SOURCE_COLUMNS = ("densegen__promoter_detail",)


def require_video_source_columns(df: pd.DataFrame) -> None:
    missing = [name for name in REQUIRED_VIDEO_SOURCE_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"Dense-array video requires columns {list(REQUIRED_VIDEO_SOURCE_COLUMNS)}; missing={missing}")


def prepare_video_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    require_video_source_columns(df)
    selected_columns = [
        *REQUIRED_VIDEO_SOURCE_COLUMNS,
        *[name for name in OPTIONAL_VIDEO_SOURCE_COLUMNS if name in df.columns],
    ]
    frame = df.loc[:, selected_columns].copy()
    _validate_scalar_source_columns(frame)
    _validate_annotation_column(frame["densegen__used_tfbs_detail"])
    return frame


def order_video_source_rows(frame: pd.DataFrame, *, video_cfg: PlotVideoConfig) -> pd.DataFrame:
    mode = str(video_cfg.mode)
    if mode == "single_plan_single_video":
        target = str(video_cfg.single_plan_name or "").strip()
        scoped = frame.loc[frame["densegen__plan"].astype(str) == target].copy()
        if scoped.empty:
            raise ValueError(f"Dense-array video single-plan mode selected no rows for plan {target!r}.")
        return scoped.reset_index(drop=True)

    grouped_indices: dict[str, list[int]] = {}
    for index, plan_name in enumerate(frame["densegen__plan"].astype(str).tolist()):
        grouped_indices.setdefault(plan_name, []).append(int(index))
    ordered_indices: list[int] = []
    plan_names = sorted(grouped_indices.keys())
    cursor = 0
    while True:
        emitted = False
        for plan_name in plan_names:
            bucket = grouped_indices[plan_name]
            if cursor < len(bucket):
                ordered_indices.append(int(bucket[cursor]))
                emitted = True
        if not emitted:
            break
        cursor += 1
    if not ordered_indices:
        raise ValueError("Dense-array video has no rows after round-robin ordering.")
    return frame.iloc[ordered_indices].reset_index(drop=True)


def sample_video_source_rows(
    frame: pd.DataFrame,
    *,
    stride: int,
    max_source_rows: int,
    max_snapshots: int,
    plan_snapshot_counts: Mapping[str, int] | None = None,
) -> tuple[pd.DataFrame, int]:
    total_rows = len(frame)
    if total_rows < 1:
        raise ValueError("Dense-array video has no source rows to sample.")
    if plan_snapshot_counts:
        sampled = _sample_video_source_rows_by_plan(frame, plan_snapshot_counts=plan_snapshot_counts)
        if len(sampled) > int(max_snapshots):
            raise ValueError("Dense-array video plan snapshot selection exceeds max_snapshots.")
        return sampled, 1
    source_cap_stride = max(1, int(math.ceil(float(total_rows) / float(max(1, int(max_source_rows))))))
    effective_stride = max(1, int(stride), source_cap_stride)
    sampled_indices = list(range(0, total_rows, effective_stride))
    if (total_rows - 1) not in sampled_indices:
        sampled_indices.append(total_rows - 1)
    sampled_indices = sorted(set(int(i) for i in sampled_indices))
    if len(sampled_indices) > int(max_snapshots):
        if int(max_snapshots) <= 0:
            raise ValueError("Dense-array video max_snapshots must be >= 1 after runtime budgeting.")
        if int(max_snapshots) == 1:
            sampled_indices = [sampled_indices[-1]]
        else:
            first = int(sampled_indices[0])
            last = int(sampled_indices[-1])
            middle = [idx for idx in sampled_indices[1:-1] if idx not in {first, last}]
            keep_middle = _uniform_pick(middle, max(0, int(max_snapshots) - 2))
            sampled_indices = [first, *keep_middle, last]
    sampled_indices = sorted(set(int(i) for i in sampled_indices))
    sampled = frame.iloc[sampled_indices].reset_index(drop=True)
    if sampled.empty:
        raise ValueError("Dense-array video sampling produced zero snapshots.")
    return sampled, effective_stride


def _sample_video_source_rows_by_plan(
    frame: pd.DataFrame,
    *,
    plan_snapshot_counts: Mapping[str, int],
) -> pd.DataFrame:
    plan_values = frame["densegen__plan"].astype(str).tolist()
    plan_order = [str(plan_name).strip() for plan_name in plan_snapshot_counts if str(plan_name).strip()]
    if not plan_order:
        raise ValueError("Dense-array video plan snapshot selection requires at least one plan.")

    selected_by_plan: dict[str, list[int]] = {}
    for plan_name in plan_order:
        bucket = [idx for idx, value in enumerate(plan_values) if value == plan_name]
        if not bucket:
            raise ValueError(f"Dense-array video plan_snapshots references unknown or empty plan {plan_name!r}.")
        requested = int(plan_snapshot_counts[plan_name])
        selected_by_plan[plan_name] = _uniform_pick(bucket, min(requested, len(bucket)))

    selected_indices: list[int] = []
    cursor = 0
    while True:
        emitted = False
        for plan_name in plan_order:
            bucket = selected_by_plan[plan_name]
            if cursor < len(bucket):
                selected_indices.append(int(bucket[cursor]))
                emitted = True
        if not emitted:
            break
        cursor += 1
    sampled = frame.iloc[selected_indices].reset_index(drop=True)
    if sampled.empty:
        raise ValueError("Dense-array video plan snapshot selection produced zero snapshots.")
    return sampled


def encode_video_source_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    encoded_frame = frame.copy()
    encoded_frame["densegen__used_tfbs_detail"] = [
        _encode_annotation_payload(value) for value in encoded_frame["densegen__used_tfbs_detail"].tolist()
    ]
    return encoded_frame


def _validate_scalar_source_columns(frame: pd.DataFrame) -> None:
    for col in ("id", "sequence", "densegen__plan"):
        if frame[col].isna().any():
            raise ValueError(f"Dense-array video source rows include null {col} values.")
        frame[col] = frame[col].astype(str).str.strip()
    if frame["id"].eq("").any():
        raise ValueError("Dense-array video source rows include blank id values.")
    if frame["sequence"].eq("").any():
        raise ValueError("Dense-array video source rows include blank sequence values.")
    if frame["densegen__plan"].eq("").any():
        raise ValueError("Dense-array video source rows include blank densegen__plan values.")
    duplicate_ids = frame["id"].duplicated(keep=False)
    if bool(duplicate_ids.any()):
        preview = sorted(set(frame.loc[duplicate_ids, "id"].astype(str).tolist()))
        raise ValueError(f"Dense-array video source rows must use unique id values; duplicates={preview[:10]}")


def _validate_annotation_column(values: pd.Series) -> None:
    for value in values.tolist():
        _parse_annotation_payload(value)


def _encode_annotation_payload(value: object) -> str:
    try:
        normalized_items = normalize_used_tfbs_entries([dict(item) for item in _parse_annotation_payload(value)])
        return json.dumps(normalized_items, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dense-array video densegen__used_tfbs_detail includes non-JSON values.") from exc


def _parse_annotation_payload(value: object) -> list[dict]:
    if _is_null_annotation(value):
        raise ValueError("Dense-array video source rows include null densegen__used_tfbs_detail values.")
    if isinstance(value, str):
        raw = value.strip()
        if raw == "":
            raise ValueError("Dense-array video source rows include blank densegen__used_tfbs_detail values.")
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            raise ValueError(
                "Dense-array video densegen__used_tfbs_detail strings must be valid JSON list/dict values."
            ) from exc
    else:
        parsed = value
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("Dense-array video densegen__used_tfbs_detail must be a JSON string, list, or dict per row.")
    parsed_items = list(parsed)
    if any(not isinstance(item, dict) for item in parsed_items):
        raise ValueError("Dense-array video densegen__used_tfbs_detail must contain dict items only.")
    return parsed_items


def _is_null_annotation(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _uniform_pick(indices: list[int], k: int) -> list[int]:
    if k <= 0 or not indices:
        return []
    if len(indices) <= k:
        return list(indices)
    if k == 1:
        return [int(indices[-1])]
    span = len(indices) - 1
    picks: list[int] = []
    seen: set[int] = set()
    for i in range(k):
        pos = int(round((i * span) / float(k - 1)))
        idx = int(indices[pos])
        if idx in seen:
            continue
        seen.add(idx)
        picks.append(idx)
    if len(picks) >= k:
        return picks[:k]
    for idx in indices:
        if idx in seen:
            continue
        picks.append(int(idx))
        seen.add(int(idx))
        if len(picks) >= k:
            break
    return picks[:k]
