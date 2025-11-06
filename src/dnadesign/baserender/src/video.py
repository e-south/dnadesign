"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/video.py
--------------------------------------------------------------------------------

Improvements in this version:
- Guarantees even video dimensions for H.264 (yuv420p) by applying an FFmpeg
  scale filter:  scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1
  This prevents failures like:
    [libx264] height not divisible by 2 (1518x505)

- Adds fail-fast checks and clearer ExportError messages with practical nudges:
  * no records → "No records to render."
  * invalid fps / frames_per_record
  * ffmpeg writer missing
  * target canvas smaller than first frame (letterbox guard)
  * if FFmpeg still fails, we surface likely cause and how to fix (width_px,
    height_px, aspect).

- Keeps start/frame/finish progress callbacks intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from .contracts import ExportError
from .layout import assign_tracks
from .model import SeqRecord
from .palette import Palette
from .render import render_figure
from .style import Style

ReportFn = Callable[[str, dict], None]


def _even_ceil(x: int) -> int:
    """Return the smallest even integer >= x."""
    return x if (x % 2 == 0) else (x + 1)


def _target_size(
    nat_w: int,
    nat_h: int,
    *,
    width_px: Optional[int],
    height_px: Optional[int],
    aspect_ratio: Optional[float],
) -> tuple[int, int]:
    """
    Choose a final frame size that is:
      - >= the natural (first frame) size in both dimensions,
      - optionally expanded to satisfy width/height/aspect,
      - even in both dimensions (H.264 / yuv420p requirement),
      - never smaller than the source after rounding.
    """
    W, H = nat_w, nat_h
    if width_px is not None:
        if width_px <= 0:
            raise ExportError("width_px must be a positive integer.")
        W = max(W, int(width_px))
    if height_px is not None:
        if height_px <= 0:
            raise ExportError("height_px must be a positive integer.")
        H = max(H, int(height_px))
    if aspect_ratio:
        ar = float(aspect_ratio)
        if ar <= 0:
            raise ExportError("aspect_ratio must be > 0.")
        # Expand minimally to meet the aspect while keeping >= W/H.
        Wc = max(W, int(round(H * ar)))
        Hc = int(round(Wc / ar))
        W, H = Wc, Hc
    # Enforce >= natural size AFTER making dimensions even (ceil).
    W = max(_even_ceil(int(W)), _even_ceil(int(nat_w)))
    H = max(_even_ceil(int(H)), _even_ceil(int(nat_h)))
    return W, H


def _letterbox(arr, W: int, H: int):
    import numpy as np

    h, w = arr.shape[:2]
    if H < h or W < w:
        # Assertive: never silently crop. Caller must choose a large enough target.
        raise ExportError(
            f"Target canvas too small for frame ({W}x{H}) < source ({w}x{h}). "
            "Increase width_px/height_px or set an aspect ratio in the job config."
        )
    out = np.ones((H, W, 4), dtype=arr.dtype) * 255  # white background
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    out[y0 : y0 + h, x0 : x0 + w, :] = arr
    return out


def _legend_entries_for_record(r: SeqRecord) -> list[tuple[str, str]]:
    """
    Build the legend for a *single* record.
    - TFs: 'tf:<name>' → '<name>' (dedup, stable order)
    - σ70: show **one** entry (σ high|medium|low). We detect it from either:
        • plugin-added 'sigma' boxes (preferred), or
        • pre-existing 'tf:sigma70_*' annotations in the dataset.
      All σ70 strengths share one color via tag 'sigma'.
    """
    entries: list[tuple[str, str]] = []
    seen_tfs: set[str] = set()
    sigma_strength: str | None = None

    for a in r.annotations:
        if a.tag.startswith("tf:"):
            name = a.tag[3:]
            # If dataset already contains sigma70_* as TFs, fold into σ legend
            low = name.lower()
            if low.startswith("sigma70_"):
                # derive 'high'|'medium'|'low' from tf name suffix
                st = low.split("_", 1)[-1]
                st = {"mid": "medium"}.get(st, st)
                if sigma_strength is None:
                    sigma_strength = st
                # do not list sigma70_* again as a plain TF
                continue
            if name not in seen_tfs:
                seen_tfs.add(name)
                entries.append((a.tag, name))
        elif a.tag == "sigma":
            try:
                st = (a.payload or {}).get("strength")  # type: ignore[assignment]
                if isinstance(st, str):
                    sigma_strength = sigma_strength or st.lower()
            except Exception:
                pass

    if sigma_strength:
        label = f"σ {sigma_strength}"
        entries.append(("sigma", label))
    return entries


def _ffmpeg_extra_args() -> list[str]:
    """
    Output options for FFmpeg that make the produced MP4 broadly compatible and resilient:
    - yuv420p: required by many players (PPT/QuickTime)
    - +faststart: moov atom front-loaded for streaming/drag‑and‑drop
    - profile main: compatibility profile
    - scale filter: force even width/height so libx264 never complains
    - setsar=1: square pixels
    """
    return [
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-profile:v",
        "main",
    ]


def render_video(
    records: Iterable[SeqRecord],
    *,
    out_path: Path,
    fps: int = 2,
    style: Style | None = None,
    palette: Palette | None = None,
    fmt: str = "mp4",
    frames_per_record: int = 1,
    pauses: Optional[Mapping[str, float]] = None,
    width_px: Optional[int] = None,
    height_px: Optional[int] = None,
    aspect_ratio: Optional[float] = None,
    total_duration: Optional[float] = None,
    report: Optional[ReportFn] = None,
) -> None:
    style = style or Style()
    palette = palette or Palette(style.palette)
    out_path = Path(out_path)
    emit = (lambda *_: None) if report is None else (lambda ev, **kw: report(ev, kw))

    if fmt != "mp4":
        raise ExportError("Only mp4 is supported for video export in v0.")
    if not animation.writers.is_available("ffmpeg"):
        raise ExportError(
            "FFmpeg writer is not available to Matplotlib. "
            "Install FFmpeg and ensure it's on your PATH (see README)."
        )
    if not isinstance(fps, int) or fps < 1:
        raise ExportError("fps must be a positive integer (e.g., 1, 2, 24, 30).")
    if not isinstance(frames_per_record, int) or frames_per_record < 1:
        raise ExportError("frames_per_record must be a positive integer (>=1).")

    pauses = dict(pauses or {})
    # Let caller know we are about to enumerate and size records
    emit("prepare", stage="enumerate")

    recs = list(records)
    if not recs:
        raise ExportError(
            "No records to render. "
            "Check your dataset path, filters/limits, or plugin configuration."
        )

    # Freeze layout across frames
    global_n = max(len(r.sequence) for r in recs)
    max_up = 0
    max_dn = 0
    for r in recs:
        up = [a for a in r.annotations if a.strand == "fwd"]
        dn = [a for a in r.annotations if a.strand == "rev"]
        up_tracks = assign_tracks(up)
        dn_tracks = assign_tracks(dn)
        max_up = max(max_up, (max(up_tracks) + 1) if up_tracks else 0)
        max_dn = max(max_dn, (max(dn_tracks) + 1) if dn_tracks else 0)
    fixed_tracks = (max_up, max_dn)

    # Duration budget (optional): pick frames_per_record to fit total_duration.
    if total_duration is not None:
        # Convert pauses (seconds) to frames at this fps.
        pause_frames = {
            r.id: max(0, int(round(float(pauses.get(r.id, 0.0)) * fps))) for r in recs
        }
        extra = sum(pause_frames.values())
        frame_budget = max(1, int(round(float(total_duration) * fps)) - extra)
        frames_per_record = max(1, frame_budget // len(recs))

    # Probe first frame (with frozen layout) to get natural size
    first_fig = render_figure(
        recs[0],
        style=style,
        palette=palette,
        out_path=None,
        fixed_tracks=fixed_tracks,
        fixed_n=global_n,
        legend_entries=_legend_entries_for_record(recs[0]),
    )
    first_fig.canvas.draw()
    import numpy as np

    first_arr = np.asarray(first_fig.canvas.buffer_rgba())
    nat_h, nat_w = first_arr.shape[:2]
    plt.close(first_fig)

    # Final frame size (letterbox if aspect requested; never stretch)
    frame_w, frame_h = _target_size(
        nat_w, nat_h, width_px=width_px, height_px=height_px, aspect_ratio=aspect_ratio
    )

    # Progress
    def _frames_for(r: SeqRecord) -> int:
        pause_frames = int(round(float(pauses.get(r.id, 0.0)) * fps))
        return max(1, frames_per_record + pause_frames)

    total_frames = sum(_frames_for(r) for r in recs)
    emit(
        "start",
        total_frames=total_frames,
        n_records=len(recs),
        fps=fps,
        width=frame_w,
        height=frame_h,
    )

    # Writer (QT/PPT-safe) with resilient extra args
    fig, ax = plt.subplots(
        figsize=(frame_w / style.dpi, frame_h / style.dpi), dpi=style.dpi
    )
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(
        np.zeros((frame_h, frame_w, 4), dtype=np.uint8),
        interpolation="nearest",
        origin="upper",
    )
    # Data is already letterboxed to exact frame size; do not rescale it.
    ax.set_aspect("auto")

    Writer = animation.FFMpegWriter
    writer = Writer(
        fps=fps,
        codec="libx264",
        extra_args=_ffmpeg_extra_args(),
    )

    frames_done = 0
    try:
        with writer.saving(fig, str(out_path), dpi=style.dpi):
            for rec in recs:
                f = render_figure(
                    rec,
                    style=style,
                    palette=palette,
                    out_path=None,
                    fixed_tracks=fixed_tracks,
                    fixed_n=global_n,
                    legend_entries=_legend_entries_for_record(rec),
                )
                f.canvas.draw()
                arr = np.asarray(f.canvas.buffer_rgba())
                plt.close(f)

                if (arr.shape[1], arr.shape[0]) != (frame_w, frame_h):
                    # Defensive: ensure final array matches requested canvas.
                    arr = _letterbox(arr, frame_w, frame_h)

                im.set_data(arr)
                ax.set_xlim(0, frame_w)
                ax.set_ylim(frame_h, 0)

                repeats = _frames_for(rec)
                for _ in range(repeats):
                    writer.grab_frame()
                    frames_done += 1
                    emit("frame", frames_done=frames_done)
    except Exception as e:  # pragma: no cover
        msg = str(e)
        # Friendly guidance for the most common encoding pitfall.
        if (
            "not divisible by 2" in msg
            or "height not divisible" in msg
            or "width not divisible" in msg
        ):
            raise ExportError(
                "FFmpeg/x264 requires even dimensions. This run should already "
                "auto-correct via a scale filter. If you still encounter this, set "
                "an explicit even width/height (e.g., width_px: 1400) or choose an "
                "aspect that produces even pixels (e.g., aspect: '16:9'). "
                f"(Underlying error: {msg})"
            ) from e
        # Generic fallback with nudge
        raise ExportError(
            "FFmpeg failed while writing frames. "
            "Try setting an explicit width_px/height_px or aspect in your job, "
            "and confirm FFmpeg is recent (H.264, yuv420p supported). "
            f"(Underlying error: {msg})"
        ) from e

    emit("finish", out=str(out_path), frames=frames_done)
