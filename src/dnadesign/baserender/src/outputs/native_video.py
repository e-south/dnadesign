"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/outputs/native_video.py

Encode admitted native compositions without resizing, cropping or repainting text.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import io
import math
import shutil
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from ..config import ImagesOutputCfg, SequenceLayout, Style, VideoOutputCfg
from ..config.native_svg import NativeSvgCfg
from ..core import RenderingError, SchemaError
from ..core.vector import RasterizerInfo
from ..integrations import validate_records_output_policy
from .names import _safe_stem, _unique_stem
from .native_batch import PreparedNativeBatch
from .raster import resolve_rasterizer
from .video_timing import _pause_frames, effective_video_frames_per_record


@dataclass(frozen=True)
class NativeEncoder:
    executable: Path
    sha256: str
    version: str

    def to_mapping(self):
        return {
            "executable": self.executable.name,
            "executable_sha256": self.sha256,
            "version": self.version,
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "crf": 18,
        }


def resolve_encoder() -> NativeEncoder:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RenderingError("FFmpeg is unavailable; native MP4 requires its existing CLI")
    path = Path(executable).resolve()
    try:
        digest = sha256(path.read_bytes()).hexdigest()
        probe = subprocess.run([str(path), "-version"], capture_output=True, text=True, timeout=10, check=True)
        codecs = subprocess.run([str(path), "-encoders"], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RenderingError("FFmpeg identity/encoder probe failed") from exc
    if not probe.stdout.startswith("ffmpeg version ") or not any(
        len(parts := line.split()) > 1 and parts[1] == "libx264" for line in codecs.stdout.splitlines()
    ):
        raise RenderingError("FFmpeg must report its version and provide the libx264 encoder")
    return NativeEncoder(path, digest, probe.stdout.splitlines()[0])


def composite_frame(source: Image.Image, background: str) -> Image.Image:
    """Flatten alpha onto the declared background, leaving all pixel positions fixed."""
    canvas = Image.new("RGBA", source.size, background)
    canvas.alpha_composite(source.convert("RGBA"))
    return canvas.convert("RGB")


def encode_native_frames(encoder: NativeEncoder, *, frames_dir: Path, repeats: list[int], output: VideoOutputCfg):
    try:
        if sha256(encoder.executable.read_bytes()).hexdigest() != encoder.sha256:
            raise RenderingError("FFmpeg executable changed since encoder admission")
    except OSError as exc:
        raise RenderingError("FFmpeg executable became unavailable") from exc
    # One lossless source per composition; holds repeat encoded frames, not rendering.
    lines = ["ffconcat version 1.0"]
    for index, count in enumerate(repeats):
        lines += [
            f"file 'frame{index:06d}.png'",
            f"option framerate {output.fps}",
            f"duration {count / output.fps:.12f}",
        ]
    lines += [f"file 'frame{len(repeats) - 1:06d}.png'", f"option framerate {output.fps}"]
    manifest = frames_dir / "frames.ffconcat"
    manifest.write_text("\n".join(lines) + "\n")
    command = [
        str(encoder.executable),
        "-v",
        "error",
        "-nostdin",
        "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest),
        "-an",
        "-frames:v",
        str(sum(repeats)),
        "-vf",
        f"fps={output.fps},setsar=1",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "main",
        "-movflags",
        "+faststart",
        str(output.path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=output.encode_timeout_s or 120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RenderingError("native MP4 encoding failed or exceeded encode_timeout_s") from exc
    if result.returncode != 0 or not output.path.is_file() or output.path.stat().st_size == 0:
        raise RenderingError(f"native MP4 encoding failed (exit {result.returncode})")


@dataclass(frozen=True)
class NativeVideoWriter:
    config: NativeSvgCfg
    rasterizer: RasterizerInfo
    encoder: NativeEncoder

    @classmethod
    def resolve(cls, config: NativeSvgCfg):
        encoder = resolve_encoder()
        return cls(config, resolve_rasterizer(config.rasterization), encoder)

    def write(self, records, *, output: VideoOutputCfg, images: ImagesOutputCfg, style: Style, layout: SequenceLayout):
        if not records:
            raise SchemaError("No records to render")
        if images.path is not None and len(records) != 1:
            raise SchemaError(
                "native single-file images require exactly one record; use an image directory for a batch"
            )
        if any(record.display.video_subtitle for record in records):
            raise SchemaError("native video does not add video_subtitle overlays; use the normal record display title")
        batch = PreparedNativeBatch.prepare(
            records, config=self.config, style=style, layout=layout, output_kind="video"
        )
        validate_records_output_policy(
            records, output_kind="images", image_output_mode="single_file" if images.path else "directory"
        )
        sizes, origins, pitches = set(), set(), set()
        for plan in batch.plans:
            measured, scale = plan.rows.layout, plan.px_to_pt
            sizes.add((measured.width * scale, measured.height * scale))
            origins.add((measured.x_left * scale, (measured.height - measured.y_forward) * scale))
            pitches.add(measured.cw * scale)
        if any(len(values) != 1 for values in (sizes, origins, pitches)):
            raise RenderingError("native video frames must share canvas, sequence origin and base pitch")
        width_pt, height_pt = next(iter(sizes))
        width, height = (
            math.floor(float(f"{value:.6f}") * self.config.rasterization.dpi / 72 + 0.5)
            for value in (width_pt, height_pt)
        )
        if width < 2 or height < 2 or width % 2 or height % 2:
            raise SchemaError("native MP4 needs even pixel dimensions; adjust canvas_size_pt or rasterization.dpi")
        if (
            (output.width_px is not None and output.width_px != width)
            or (output.height_px is not None and output.height_px != height)
            or (output.aspect_ratio is not None and not math.isclose(output.aspect_ratio, width / height, abs_tol=1e-6))
        ):
            raise SchemaError(
                f"native video size settings must match its sampled canvas ({width}x{height}); no resizing is performed"
            )
        frames_per_record = effective_video_frames_per_record(records, output=output)
        repeats = [max(1, frames_per_record + _pause_frames(record.id, output=output)) for record in records]
        root = images.path.parent if images.path else images.dir
        if root is None:
            raise SchemaError("native video requires image fallback output")
        root.mkdir(parents=True, exist_ok=True)
        output.path.parent.mkdir(parents=True, exist_ok=True)
        used, metrics = set(), []
        with TemporaryDirectory(prefix=".native-frames-", dir=output.path.parent) as temporary:
            frames_dir = Path(temporary)
            for index, (svg, png, item) in enumerate(batch.draw(rasterizer=self.rasterizer)):
                if (png.width_px, png.height_px) != (width, height):
                    raise RenderingError("native PNG dimensions differ from the admitted video canvas")
                name = (
                    images.path.name
                    if images.path
                    else f"{_unique_stem(_safe_stem(item['record_id']), used)}.{images.fmt}"
                )
                item["file"] = name
                with (root / name).open("xb") as handle:
                    handle.write(png.png if images.fmt == "png" else svg.svg.encode("utf-8"))
                with Image.open(io.BytesIO(png.png)) as source:
                    composite_frame(source, output.background).save(frames_dir / f"frame{index:06d}.png")
                metrics.append(item)
            batch.verify_sources()
            encode_native_frames(self.encoder, frames_dir=frames_dir, repeats=repeats, output=output)
            batch.verify_sources()
        return {
            "images": {"representation": "native_svg", "record_count": len(records), "records": metrics},
            "video": {
                "representation": "native_svg",
                "record_count": len(records),
                "fps": output.fps,
                "frames_per_record": frames_per_record,
                "planned_frame_count": sum(repeats),
                "frame_repeats": repeats,
                "width_px": width,
                "height_px": height,
                "background": output.background,
                "encoder": self.encoder.to_mapping(),
                "rasterizer": self.rasterizer.to_mapping(portable=True),
            },
        }
