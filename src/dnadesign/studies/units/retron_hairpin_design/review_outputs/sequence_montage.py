"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/sequence_montage.py

Sequence-still montage rendering for Retron review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from ..compiler.exceptions import RetronMsdCompilerError
from .sequence_index import SequenceReviewFrame

VideoWriter = Callable[..., None]
EDGE_COLUMN_MAX_RGB_THRESHOLD = 250
EDGE_MAX_COLUMNS = 16
EDGE_MIN_FRACTION = 0.85


def write_sequence_montage(
    frames: Sequence[SequenceReviewFrame],
    *,
    out_dir: Path,
    deliverable_plan_id: str,
    materialized_root: Path,
    video_writer: VideoWriter | None = None,
    fps: int = 1,
    seconds_per_frame: int = 2,
) -> tuple[Path, Path]:
    video_dir = out_dir / "reviews" / "video"
    stills_dir = video_dir / "stills"
    video_path = video_dir / f"{deliverable_plan_id}.sequence_montage.mp4"
    manifest_path = video_dir / f"{deliverable_plan_id}.sequence_montage.manifest.json"
    still_paths = tuple(_write_review_still(frame, stills_dir=stills_dir) for frame in frames)
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
        "fps": fps,
        "seconds_per_frame": seconds_per_frame,
        "source_materialized_root": materialized_root.as_posix(),
        "frames": [
            _frame_manifest(frame, materialized_root=materialized_root, still_path=still_path, out_dir=out_dir)
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
    fig = plt.figure(figsize=(16, 9), dpi=120, frameon=False)
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_axis_off()
    image = axis.imshow(arrays[0], interpolation="nearest")

    def update(index: int):
        image.set_data(arrays[index])
        return [image]

    anim = animation.FuncAnimation(fig, update, frames=len(arrays), interval=1000 / max(1, fps), blit=True)
    writer = animation.FFMpegWriter(fps=max(1, fps), codec="libx264", bitrate=2400)
    anim.save(output_path, writer=writer)
    plt.close(fig)


def _write_review_still(frame: SequenceReviewFrame, *, stills_dir: Path) -> Path:
    from PIL import Image, ImageOps

    source = _trim_edge_artifact_columns(Image.open(frame.composition_overview_png).convert("RGB"))
    canvas = Image.new("RGB", (1600, 900), color="white")
    fitted = ImageOps.contain(source, (1600, 900))
    canvas.paste(fitted, ((1600 - fitted.width) // 2, (900 - fitted.height) // 2))
    stills_dir.mkdir(parents=True, exist_ok=True)
    path = stills_dir / f"{frame_filename_stem(frame)}.png"
    canvas.save(path)
    return path


def _trim_edge_artifact_columns(image):
    left = 0
    right = image.width
    while left < min(EDGE_MAX_COLUMNS, right - 1) and _is_edge_artifact_column(image, left):
        left += 1
    while right > max(left + 1, image.width - EDGE_MAX_COLUMNS) and _is_edge_artifact_column(image, right - 1):
        right -= 1
    if left == 0 and right == image.width:
        return image
    return image.crop((left, 0, right, image.height))


def _is_edge_artifact_column(image, x: int) -> bool:
    artifact = sum(1 for y in range(image.height) if max(image.getpixel((x, y))[:3]) < EDGE_COLUMN_MAX_RGB_THRESHOLD)
    return artifact / image.height >= EDGE_MIN_FRACTION


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
) -> dict[str, object]:
    return {
        "order": frame.order,
        "evidence_label": frame_evidence_label(frame),
        "construct_id": frame.construct_id,
        "msd_design_id": frame.msd_design_id,
        "payload_trim_id": frame.payload_trim_id,
        "scaffold_context": frame.scaffold_context,
        "variant_role": frame.variant_role,
        "rt_mode": frame.rt_mode,
        "review_still_png": _relative_to(still_path, out_dir),
        "composition_overview_png": _relative_to(frame.composition_overview_png, materialized_root),
    }


def frame_filename_stem(frame: SequenceReviewFrame) -> str:
    role = {"scaffold_target": "target", "rescue_candidate": "rescue"}.get(frame.variant_role, frame.variant_role)
    return f"{frame.order:02d}_{_slug(role)}_{_slug(frame.scaffold_context)}_{_slug(frame.payload_trim_id)}"


def frame_evidence_label(frame: SequenceReviewFrame) -> str:
    role = {"scaffold_target": "target", "rescue_candidate": "rescue"}.get(frame.variant_role, frame.variant_role)
    return f"{role} | {frame.scaffold_context} | {frame.payload_trim_id}"


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() or character == "_" else "_" for character in value).strip("_")


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = ["VideoWriter", "write_sequence_montage"]
