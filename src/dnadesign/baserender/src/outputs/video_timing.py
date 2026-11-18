"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/outputs/video_timing.py

Plan record holds and pauses independently of the frame compositor and encoder.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

from ..config import VideoOutputCfg
from ..core import Record, SchemaError


def _pause_frames(record_id: str, *, output: VideoOutputCfg) -> int:
    raw = output.pauses.get(record_id, 0.0)
    pause_seconds = float(raw)
    if pause_seconds < 0:
        raise SchemaError(f"outputs.video.pauses[{record_id!r}] must be >= 0")
    return int(round(pause_seconds * output.fps))


def effective_video_frames_per_record(records: Sequence[Record], *, output: VideoOutputCfg) -> int:
    if not records:
        raise SchemaError("Cannot plan video frames with no records.")
    frames_per_record = int(output.frames_per_record)
    if output.total_duration is not None:
        pause_total = sum(_pause_frames(rec.id, output=output) for rec in records)
        frame_budget = max(1, int(round(float(output.total_duration) * output.fps)) - pause_total)
        frames_per_record = max(1, frame_budget // len(records))
    return int(frames_per_record)


def planned_video_frame_count(records: Sequence[Record], *, output: VideoOutputCfg) -> int:
    frames_per_record = effective_video_frames_per_record(records, output=output)
    return int(sum(max(1, frames_per_record + _pause_frames(rec.id, output=output)) for rec in records))
