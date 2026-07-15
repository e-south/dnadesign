"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/movie_runtime.py

Shared ChimeraX execution and streaming movie-encoding runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    sha256,
)


@dataclass(frozen=True)
class MovieRenderSpec:
    """Explicit execution and encoding contract for one ChimeraX movie."""

    schema_id: str
    schema_version: int
    renderer: str
    output_key: str
    frame_width: int
    frame_height: int
    frame_rate: int
    frames_per_scene: int
    hold_frames_per_scene: int
    timeout_seconds: int = 900

    def encoding_contract(self) -> dict[str, Any]:
        """Return fields that invalidate a previously encoded movie when changed."""

        return {
            "renderer": self.renderer,
            "encoder": "ffmpeg",
            "background": "white",
            "frame_rate": self.frame_rate,
            "width": self.frame_width,
            "height": self.frame_height,
            "frames_per_scene": self.frames_per_scene,
            "hold_frames_per_scene": self.hold_frames_per_scene,
        }


def quoted_absolute_path(path: Path) -> str:
    """Return a quoted absolute path that is independent of ChimeraX's working directory."""

    value = Path(os.path.abspath(path)).as_posix().replace('"', '\\"')
    return f'"{value}"'


def materialize_chimerax_movie(
    *,
    script_path: Path,
    movie_path: Path,
    frame_directory: Path,
    render_manifest_path: Path,
    log_path: Path,
    source_paths: dict[str, Path],
    render_requested: bool,
    spec: MovieRenderSpec,
    expected_raw_frame_count: int,
    encoding_metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Reuse or explicitly render one provenance-linked ChimeraX movie."""

    _require_files({**source_paths, "movie_script": script_path})
    if expected_raw_frame_count <= 0:
        raise ValueError("expected_raw_frame_count must be positive")
    input_hashes = file_hashes({**source_paths, "movie_script": script_path})
    encoding_contract = {
        **spec.encoding_contract(),
        "raw_frame_count": expected_raw_frame_count,
        **dict(encoding_metadata or {}),
    }
    if _existing_render_is_current(
        render_manifest_path=render_manifest_path,
        movie_path=movie_path,
        input_hashes=input_hashes,
        schema_id=spec.schema_id,
        schema_version=spec.schema_version,
        encoding_contract=encoding_contract,
        output_key=spec.output_key,
        output_path=movie_path.name,
    ):
        return "rendered", ""
    if not render_requested:
        shutil.rmtree(frame_directory, ignore_errors=True)
        stale_paths = [path for path in (movie_path, render_manifest_path, log_path) if path.exists()]
        if stale_paths:
            return (
                "skipped_stale_optional_render_retained",
                "A stale ChimeraX movie remains on disk but is not linked; request this target to rerender it.",
            )
        return "skipped_optional_render_disabled", "This ChimeraX movie target was not requested."

    for path in (movie_path, render_manifest_path):
        path.unlink(missing_ok=True)
    shutil.rmtree(frame_directory, ignore_errors=True)
    frame_directory.mkdir(parents=True)
    run_status, run_reason = run_chimerax_script(
        script_path=script_path,
        log_path=log_path,
        timeout_seconds=spec.timeout_seconds,
    )
    if run_status != "completed":
        return run_status, run_reason
    try:
        frame_count = encode_movie_frames(
            frame_directory=frame_directory,
            movie_path=movie_path,
            log_path=log_path,
            frame_width=spec.frame_width,
            frame_height=spec.frame_height,
            frames_per_scene=spec.frames_per_scene,
            hold_frames_per_scene=spec.hold_frames_per_scene,
            frame_rate=spec.frame_rate,
            expected_raw_frame_count=expected_raw_frame_count,
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\nmovie_encoding_error: {type(error).__name__}: {error}\n")
        return "errored", f"Movie encoding failed; inspect {log_path.name}."

    payload = {
        "schema_id": spec.schema_id,
        "schema_version": spec.schema_version,
        "status": "rendered",
        "input_hashes": input_hashes,
        "movie_encoding": {**encoding_contract, "frame_count": frame_count},
        "output": {
            "key": spec.output_key,
            "path": movie_path.name,
            "sha256": "sha256:" + sha256(movie_path),
        },
    }
    render_manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return "rendered", ""


def encode_movie_frames(
    *,
    frame_directory: Path,
    movie_path: Path,
    log_path: Path,
    frame_width: int,
    frame_height: int,
    frames_per_scene: int,
    hold_frames_per_scene: int,
    frame_rate: int,
    expected_raw_frame_count: int,
) -> int:
    """Stream checked fixed-size PNG scenes into one H.264 MP4."""

    frame_paths = sorted(frame_directory.glob("frame-*.png"))
    if not frame_paths:
        raise RuntimeError(f"No ChimeraX movie frames were written to {frame_directory}")
    if len(frame_paths) != expected_raw_frame_count:
        raise RuntimeError(
            f"Expected {expected_raw_frame_count} ChimeraX frames, found {len(frame_paths)} in {frame_directory}"
        )
    if frames_per_scene <= 0 or len(frame_paths) % frames_per_scene:
        raise RuntimeError(
            "ChimeraX frame count does not contain complete scenes: "
            f"{len(frame_paths)} frames for {frames_per_scene} frames per scene"
        )
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is required to encode ChimeraX movie frames")
    command = [
        executable,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{frame_width}x{frame_height}",
        "-r",
        str(frame_rate),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(movie_path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stderr is None:
        process.kill()
        raise RuntimeError("ffmpeg pipe initialization failed")
    try:
        encoded_frame_count = 0
        for frame_index, frame_path in enumerate(frame_paths, start=1):
            frame_bytes = flatten_movie_frame(frame_path, width=frame_width, height=frame_height)
            validate_white_frame_corners(frame_bytes, width=frame_width, height=frame_height, path=frame_path)
            process.stdin.write(frame_bytes)
            encoded_frame_count += 1
            if frame_index % frames_per_scene == 0:
                for _ in range(hold_frames_per_scene):
                    process.stdin.write(frame_bytes)
                    encoded_frame_count += 1
        process.stdin.close()
        return_code = process.wait(timeout=300)
        stderr = process.stderr.read().decode("utf-8", errors="replace")
    except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
        process.kill()
        process.wait()
        raise
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\nffmpeg_command: {command!r}\nffmpeg_returncode: {return_code}\n\nffmpeg_stderr:\n{stderr}\n")
    if return_code != 0 or not movie_path.exists():
        raise RuntimeError("ffmpeg did not produce the expected MP4")
    shutil.rmtree(frame_directory)
    return encoded_frame_count


def flatten_movie_frame(path: Path, *, width: int, height: int) -> bytes:
    """Flatten one RGBA ChimeraX frame against white."""

    from PIL import Image

    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        if rgba.size != (width, height):
            raise RuntimeError(f"Unexpected ChimeraX frame dimensions for {path}: {rgba.size}")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB").tobytes()


def validate_white_frame_corners(frame_bytes: bytes, *, width: int, height: int, path: Path) -> None:
    """Reject captures whose corners retain stale scene pixels."""

    for x, y in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        offset = (y * width + x) * 3
        if any(channel < 245 for channel in frame_bytes[offset : offset + 3]):
            raise RuntimeError(f"ChimeraX frame background was not cleared at a corner: {path}")


def find_chimerax() -> str:
    """Resolve a command-line ChimeraX executable without launching it."""

    executable = shutil.which("ChimeraX") or shutil.which("chimerax")
    if executable:
        return executable
    for app_root in (Path("/Applications"), Path.home() / "Applications"):
        for app_path in sorted(app_root.glob("ChimeraX*.app")):
            candidate = app_path / "Contents" / "MacOS" / "ChimeraX"
            if candidate.exists():
                return str(candidate)
    return ""


def run_chimerax_script(*, script_path: Path, log_path: Path, timeout_seconds: int = 900) -> tuple[str, str]:
    """Run one graphical ChimeraX script and persist stdout and stderr."""

    executable = find_chimerax()
    if not executable:
        return "skipped_runtime_unavailable", "ChimeraX executable was not found on this machine."
    command = [executable, "--exit", "--script", str(script_path)]
    try:
        completed = subprocess.run(
            command,
            cwd=script_path.parent,
            check=False,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        log_path.write_text(
            f"command: {command!r}\nerror: {type(error).__name__}: {error}\n",
            encoding="utf-8",
        )
        return "errored", f"ChimeraX did not complete; inspect {log_path.name}."
    log_path.write_text(
        f"command: {command!r}\nreturncode: {completed.returncode}\n\n"
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return "errored", f"ChimeraX returned a nonzero exit status; inspect {log_path.name}."
    return "completed", ""


def _existing_render_is_current(
    *,
    render_manifest_path: Path,
    movie_path: Path,
    input_hashes: dict[str, str],
    schema_id: str,
    schema_version: int,
    encoding_contract: dict[str, Any],
    output_key: str,
    output_path: str,
) -> bool:
    if not render_manifest_path.is_file() or not movie_path.is_file():
        return False
    payload = yaml.safe_load(render_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return False
    output = payload.get("output")
    if not isinstance(output, dict):
        return False
    observed_encoding = payload.get("movie_encoding")
    return (
        payload.get("schema_id") == schema_id
        and payload.get("schema_version") == schema_version
        and payload.get("status") == "rendered"
        and payload.get("input_hashes") == input_hashes
        and isinstance(observed_encoding, dict)
        and all(observed_encoding.get(key) == value for key, value in encoding_contract.items())
        and output.get("key") == output_key
        and output.get("path") == output_path
        and output.get("sha256") == "sha256:" + sha256(movie_path)
    )


def _require_files(paths: dict[str, Path]) -> None:
    missing = [(label, path) for label, path in paths.items() if not path.is_file()]
    if missing:
        label, path = missing[0]
        raise FileNotFoundError(f"Required movie input is missing ({label}): {path}")


__all__ = [
    "MovieRenderSpec",
    "encode_movie_frames",
    "find_chimerax",
    "flatten_movie_frame",
    "materialize_chimerax_movie",
    "quoted_absolute_path",
    "run_chimerax_script",
    "validate_white_frame_corners",
]
