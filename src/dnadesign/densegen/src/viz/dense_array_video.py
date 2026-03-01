"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/dense_array_video.py

Builds workspace-scoped DenseGen showcase videos from sampled accepted outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import math
import re
import tempfile
from pathlib import Path

import pandas as pd

from dnadesign.baserender import run_job
from dnadesign.densegen.src.config.plots import PlotVideoConfig
from dnadesign.densegen.src.integrations.baserender.notebook_contract import densegen_notebook_render_contract

_REQUIRED_COLUMNS = (
    "id",
    "sequence",
    "densegen__plan",
    "densegen__used_tfbs_detail",
)
_OVERLAY_TEXT_COLUMN = "densegen__video_overlay_text"
_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_segment(text: str) -> str:
    cleaned = _SAFE_SEGMENT_RE.sub("_", str(text).strip())
    cleaned = cleaned.strip("._-")
    if cleaned in {"", ".", ".."}:
        return "densegen"
    return cleaned


def _require_columns(df: pd.DataFrame) -> None:
    missing = [name for name in _REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"Dense-array video requires columns {list(_REQUIRED_COLUMNS)}; missing={missing}")


def _normalize_source_rows(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df)
    frame = df.loc[:, list(_REQUIRED_COLUMNS)].copy()
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
    encoded_annotations: list[str] = []
    for value in frame["densegen__used_tfbs_detail"].tolist():
        is_null_annotation = False
        if value is None:
            is_null_annotation = True
        else:
            try:
                is_null_annotation = bool(pd.isna(value))
            except Exception:
                is_null_annotation = False
        if is_null_annotation:
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
            if not isinstance(parsed, (list, dict)):
                raise ValueError("Dense-array video densegen__used_tfbs_detail must encode a JSON list/dict per row.")
            encoded_annotations.append(json.dumps(parsed, separators=(",", ":"), allow_nan=False))
            continue
        if not isinstance(value, (list, dict)):
            raise ValueError(
                "Dense-array video densegen__used_tfbs_detail must be a JSON string, list, or dict per row."
            )
        try:
            encoded_annotations.append(json.dumps(value, separators=(",", ":"), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("Dense-array video densegen__used_tfbs_detail includes non-JSON values.") from exc
    frame["densegen__used_tfbs_detail"] = encoded_annotations
    return frame


def _display_record_id(record_id: str) -> str:
    record_text = str(record_id or "unknown")
    if len(record_text) > 16:
        return f"{record_text[:8]}...{record_text[-4:]}"
    return record_text


def _attach_overlay_text(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched[_OVERLAY_TEXT_COLUMN] = [
        f"TFBS arrangement {_display_record_id(record_id)}" for record_id in enriched["id"].astype(str).tolist()
    ]
    return enriched


def _ordered_rows_for_mode(frame: pd.DataFrame, *, video_cfg: PlotVideoConfig) -> pd.DataFrame:
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


def _sample_rows(
    frame: pd.DataFrame,
    *,
    stride: int,
    max_source_rows: int,
    max_snapshots: int,
) -> tuple[pd.DataFrame, int]:
    total_rows = len(frame)
    if total_rows < 1:
        raise ValueError("Dense-array video has no source rows to sample.")
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


def _output_path(out_path: Path, *, video_cfg: PlotVideoConfig) -> Path:
    out_dir = out_path.parent.resolve()
    stage_b_root = (out_dir / "stage_b").resolve()
    if str(video_cfg.mode) == "single_plan_single_video":
        plan_name = str(video_cfg.single_plan_name or "").strip()
        plan_segment = _safe_segment(plan_name)
    else:
        plan_segment = "all_plans"
    candidate = (stage_b_root / plan_segment / str(video_cfg.output_name)).resolve()
    try:
        candidate.relative_to(stage_b_root)
    except ValueError as exc:
        raise ValueError("Dense-array video output path escaped stage_b workspace scope.") from exc
    return candidate


def _write_selection_csv(path: Path, *, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id"])
        writer.writeheader()
        for record_id in ids:
            writer.writerow({"id": str(record_id)})


def plot_dense_array_video_showcase(
    dense_arrays_df: pd.DataFrame,
    out_path: Path,
    *,
    video_cfg: PlotVideoConfig,
) -> Path:
    if dense_arrays_df is None or dense_arrays_df.empty:
        raise ValueError("Dense-array video requires non-empty accepted output rows.")

    frame = _normalize_source_rows(dense_arrays_df)
    ordered = _ordered_rows_for_mode(frame, video_cfg=video_cfg)

    target_total_frames = int(round(float(video_cfg.playback.target_duration_sec) * float(video_cfg.playback.fps)))
    if target_total_frames < 1:
        raise ValueError("Dense-array video target frame budget must be >= 1.")
    if target_total_frames > int(video_cfg.limits.max_total_frames):
        raise ValueError(
            "Dense-array video target frames exceed plots.video.limits.max_total_frames; "
            "reduce target_duration_sec/fps or increase max_total_frames."
        )
    snapshot_cap = max(1, min(int(video_cfg.sampling.max_snapshots), int(target_total_frames)))
    sampled, effective_stride = _sample_rows(
        ordered,
        stride=int(video_cfg.sampling.stride),
        max_source_rows=int(video_cfg.sampling.max_source_rows),
        max_snapshots=snapshot_cap,
    )
    sampled = _attach_overlay_text(sampled)
    if len(sampled) > target_total_frames:
        raise ValueError(
            "Dense-array video snapshot count exceeds playback frame budget; "
            "increase sampling.stride or reduce max_snapshots."
        )
    estimated_render_sec = float(target_total_frames / max(1, int(video_cfg.playback.fps))) * 1.5
    if estimated_render_sec > float(video_cfg.limits.max_estimated_render_sec):
        raise ValueError(
            "Dense-array video estimated render time exceeds plots.video.limits.max_estimated_render_sec; "
            "reduce duration/fps or raise max_estimated_render_sec."
        )

    out_file = _output_path(out_path, video_cfg=video_cfg)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    contract = densegen_notebook_render_contract()
    with tempfile.TemporaryDirectory(prefix="dense-video-", dir=str(out_path.parent)) as tmpdir:
        tmp_root = Path(tmpdir)
        records_path = tmp_root / "records.parquet"
        selection_path = tmp_root / "selection.csv"
        sampled.to_parquet(records_path, index=False, engine="pyarrow")
        _write_selection_csv(selection_path, ids=[str(item) for item in sampled["id"].astype(str).tolist()])

        job_mapping: dict[str, object] = {
            "version": 3,
            "input": {
                "kind": "parquet",
                "path": str(records_path),
                "adapter": {
                    "kind": str(contract.adapter_kind),
                    "columns": {
                        **dict(contract.adapter_columns),
                        "overlay_text": _OVERLAY_TEXT_COLUMN,
                    },
                    "policies": dict(contract.adapter_policies),
                },
                "alphabet": "DNA",
            },
            "selection": {
                "path": str(selection_path),
                "match_on": "id",
                "column": "id",
                "keep_order": True,
                "on_missing": "error",
            },
            "render": {
                "renderer": "sequence_rows",
                "style": {
                    "preset": str(contract.style_preset),
                    "overrides": {
                        "overlay_align": "center",
                        "font_size_label": 15,
                    },
                },
            },
            "outputs": [
                {
                    "kind": "video",
                    "path": str(out_file),
                    "fmt": "mp4",
                    "fps": int(video_cfg.playback.fps),
                    "frames_per_record": 1,
                    "total_duration": float(video_cfg.playback.target_duration_sec),
                }
            ],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        }
        run_job(job_mapping, kind="sequence_rows_v3", caller_root=tmp_root)

    if not out_file.exists():
        raise ValueError(f"Dense-array video output was not created: {out_file}")
    return out_file
