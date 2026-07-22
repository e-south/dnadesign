"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/video/montage.py

Sequence-still montage rendering for Retron review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Mapping

from ...compiler.exceptions import RetronMsdCompilerError
from ..sequence.index import SequenceReviewFrame
from ..sequence.variant_identity import variant_id
from .frame_naming import frame_evidence_label, review_construct_id
from .stills import STILL_SIZE_PX, write_review_stills

VideoWriter = Callable[..., None]
VIDEO_SIZE_PX = (1920, 1080)


def write_sequence_montage(
    frames: Sequence[SequenceReviewFrame],
    *,
    out_dir: Path,
    deliverable_plan_id: str,
    materialized_root: Path,
    review_variant_ids: Mapping[str, str],
    video_writer: VideoWriter | None = None,
    fps: int = 1,
    seconds_per_frame: int = 2,
) -> tuple[Path, Path]:
    video_dir = out_dir / "reviews" / "video"
    stills_dir = video_dir / "stills"
    video_path = video_dir / f"{deliverable_plan_id}.sequence_montage.mp4"
    manifest_path = video_dir / f"{deliverable_plan_id}.sequence_montage.manifest.json"
    still_paths = write_review_stills(frames, stills_dir=stills_dir, review_variant_ids=review_variant_ids)
    writer = video_writer or _write_montage_video
    writer(
        frames=frames,
        still_paths=still_paths,
        output_path=video_path,
        fps=fps,
        seconds_per_frame=seconds_per_frame,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "contract": "retron_hairpin_sequence_montage_manifest_v1",
        "deliverable_plan_id": deliverable_plan_id,
        "video_path": _relative_to(video_path, out_dir),
        "frame_count": len(frames),
        "still_count": len(still_paths),
        "still_resolution_px": {"width": STILL_SIZE_PX[0], "height": STILL_SIZE_PX[1]},
        "video_resolution_px": {"width": VIDEO_SIZE_PX[0], "height": VIDEO_SIZE_PX[1]},
        "fps": fps,
        "seconds_per_frame": seconds_per_frame,
        "source_materialized_root": materialized_root.as_posix(),
        "review_variant_ids": dict(review_variant_ids),
        "frames": [
            _frame_manifest(
                frame,
                materialized_root=materialized_root,
                still_path=still_path,
                out_dir=out_dir,
                review_variant_ids=review_variant_ids,
            )
            for frame, still_path in zip(frames, still_paths, strict=True)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return video_path, manifest_path


def _write_montage_video(
    *,
    frames: Sequence[SequenceReviewFrame],
    still_paths: Sequence[Path],
    output_path: Path,
    fps: int,
    seconds_per_frame: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    if not frames:
        raise RetronMsdCompilerError("Retron sequence montage requires at least one frame.")
    if len(still_paths) != len(frames):
        raise RetronMsdCompilerError("Retron sequence montage still count must match frame count.")
    if not animation.writers.is_available("ffmpeg"):
        raise RetronMsdCompilerError("FFmpeg writer is required to render Retron sequence montage MP4 output.")
    arrays = [_load_still_array(path) for path in still_paths for _ in range(max(1, int(seconds_per_frame)))]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 9), dpi=VIDEO_SIZE_PX[0] // 16, frameon=False)
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_axis_off()
    image = axis.imshow(arrays[0], interpolation="nearest")

    def update(index: int):
        image.set_data(arrays[index])
        return [image]

    anim = animation.FuncAnimation(fig, update, frames=len(arrays), interval=1000 / max(1, fps), blit=True)
    writer = animation.FFMpegWriter(
        fps=max(1, fps),
        codec="libx264",
        bitrate=-1,
        extra_args=["-crf", "16", "-preset", "slow", "-tune", "stillimage", "-pix_fmt", "yuv420p"],
    )
    anim.save(output_path, writer=writer)
    plt.close(fig)


def _load_still_array(path: Path):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _frame_manifest(
    frame: SequenceReviewFrame,
    *,
    materialized_root: Path,
    still_path: Path,
    out_dir: Path,
    review_variant_ids: Mapping[str, str],
) -> dict[str, object]:
    compact_variant_id = variant_id(frame)
    return {
        "order": frame.order,
        "variant_id": compact_variant_id,
        "review_construct_id": review_construct_id(frame, review_variant_ids=review_variant_ids),
        "evidence_label": frame_evidence_label(frame, review_variant_ids=review_variant_ids),
        "construct_id": frame.construct_id,
        "msd_design_id": frame.msd_design_id,
        "payload_trim_id": frame.payload_trim_id,
        "scaffold_context": frame.scaffold_context,
        "variant_role": frame.variant_role,
        "rt_mode": frame.rt_mode,
        "review_still_png": _relative_to(still_path, out_dir),
        "composition_overview_png": _relative_to(frame.composition_overview_png, materialized_root),
    }


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = ["STILL_SIZE_PX", "VIDEO_SIZE_PX", "VideoWriter", "write_sequence_montage"]
