"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/chimerax_story.py

ChimeraX script generation and optional render materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    sha256,
)

from ..molecular_scene_contract import chimerax_reference_complex_style_commands
from .pose import (
    CHIMERAX_CAMERA_MATRIX,
    CHIMERAX_HOLD_FRAMES_PER_SCENE,
    CHIMERAX_MOVIE_FRAME_RATE,
    CHIMERAX_MOVIE_HEIGHT,
    CHIMERAX_MOVIE_VIEW_ZOOM,
    CHIMERAX_MOVIE_WIDTH,
    CHIMERAX_PROTECTED_SURFACE_TRANSPARENCY_PERCENT,
    CHIMERAX_ROTATION_DEGREES_PER_SCENE,
    CHIMERAX_ROTATION_FRAMES_PER_SCENE,
    CHIMERAX_VIEW_PADDING,
)
from .style import DNA_COLOR, PROTEIN_SURFACE_COLOR, PROTEIN_SURFACE_OPACITY, RNA_COLOR

CHIMERAX_LOG_FILE_NAME = "eco1_structure_story_chimerax.log"
CHIMERAX_FRAME_DIRECTORY_NAME = ".eco1_structure_story_frames"


def write_chimerax_script(
    *,
    script_path: Path,
    reference_structure_path: Path,
    frame_directory: Path,
    reference_number_by_canonical: dict[int, int],
    scene_specs: tuple[dict[str, Any], ...],
) -> None:
    """Write the deterministic protected-evidence rotation-movie script."""

    script_path.parent.mkdir(parents=True, exist_ok=True)
    reference_relative = quoted_relative_path(reference_structure_path, script_path.parent)
    lines = [
        "# Eco1 RT scientific-communication structure story",
        "# Generated from the active policy-position manifest and approved communication pose.",
        f"# Visual contract: DNA {DNA_COLOR}; RNA {RNA_COLOR}; ladder nucleotides; surface alpha "
        f"{PROTEIN_SURFACE_OPACITY:.2f}.",
        f"windowsize {CHIMERAX_MOVIE_WIDTH} {CHIMERAX_MOVIE_HEIGHT}",
        "camera ortho",
        "lighting soft",
        "graphics silhouettes true",
        "set bgColor white",
        f"open {reference_relative}",
        *chimerax_reference_complex_style_commands(include_protein_surface=False),
        "surface #1/A",
        "rename #1.1 protein_surface",
        f"color #1/A {PROTEIN_SURFACE_COLOR} target s",
        "color #1/A #8C959F target c",
        f"view matrix camera {CHIMERAX_CAMERA_MATRIX}",
        f"view all pad {CHIMERAX_VIEW_PADDING}",
        f"zoom {CHIMERAX_MOVIE_VIEW_ZOOM}",
        "set bgColor white",
    ]
    frame_number = 1
    for scene in scene_specs:
        scene_commands, frame_number = _chimerax_scene_commands(
            scene,
            reference_number_by_canonical=reference_number_by_canonical,
            frame_directory=frame_directory,
            script_root=script_path.parent,
            first_frame_number=frame_number,
        )
        lines.extend(scene_commands)
    lines.append("exit")
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _chimerax_scene_commands(
    scene: dict[str, Any],
    *,
    reference_number_by_canonical: dict[int, int],
    frame_directory: Path,
    script_root: Path,
    first_frame_number: int,
) -> tuple[list[str], int]:
    residue_numbers = sorted(
        reference_number_by_canonical[position]
        for position in scene["positions"]
        if position in reference_number_by_canonical
    )
    commands = [
        f"color #1/A {PROTEIN_SURFACE_COLOR} target s",
        "hide #1/A atoms",
        "2dlabels delete all",
        "set bgColor white",
        f'2dlabels text "{scene["label"]}" xpos 0.035 ypos 0.955 size 25 color black bgColor white',
    ]
    if residue_numbers:
        residue_selection = f"#1/A:{_residue_ranges(residue_numbers)}"
        sidechain_selection = f"{residue_selection} & sidechain"
        commands.extend(
            [
                f"color {residue_selection} {scene['color']} target s",
                f"show {sidechain_selection} atoms",
                f"style {sidechain_selection} stick",
                f"size {sidechain_selection} stickRadius 0.24",
                f"color {sidechain_selection} {scene['color']} target a",
            ]
        )
    commands.append(f"transparency #1/A {CHIMERAX_PROTECTED_SURFACE_TRANSPARENCY_PERCENT} target s")
    angle_per_frame = CHIMERAX_ROTATION_DEGREES_PER_SCENE / CHIMERAX_ROTATION_FRAMES_PER_SCENE
    next_frame_number = first_frame_number
    for _ in range(CHIMERAX_ROTATION_FRAMES_PER_SCENE):
        frame_path = frame_directory / f"frame-{next_frame_number:05d}.png"
        frame_relative = quoted_relative_path(frame_path, script_root)
        commands.extend(
            [
                f"turn y {angle_per_frame:.6f}",
                "wait 1",
                f"save {frame_relative} width {CHIMERAX_MOVIE_WIDTH} height {CHIMERAX_MOVIE_HEIGHT} "
                "supersample 1 transparentBackground false",
            ]
        )
        next_frame_number += 1
    return commands, next_frame_number


def _residue_ranges(values: list[int]) -> str:
    runs: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    runs.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(runs)


def quoted_relative_path(path: Path, root: Path) -> str:
    """Return a ChimeraX-safe quoted path relative to a script root."""

    value = os.path.relpath(path.resolve(), root.resolve()).replace('"', '\\"')
    return f'"{value}"'


def materialize_chimerax_outputs(
    *,
    script_path: Path,
    reference_structure_path: Path,
    movie_path: Path,
    frame_directory: Path,
    render_manifest_path: Path,
    render_chimerax: bool,
) -> tuple[str, str]:
    """Optionally run ChimeraX and encode its captured PNG frames."""

    if _existing_render_is_current(
        render_manifest_path=render_manifest_path,
        script_path=script_path,
        reference_structure_path=reference_structure_path,
        movie_path=movie_path,
    ):
        return "rendered", ""
    if not render_chimerax:
        return "skipped_optional_render_disabled", "ChimeraX communication rendering was not requested."
    log_path = script_path.parent / CHIMERAX_LOG_FILE_NAME
    for stale_path in (movie_path, render_manifest_path):
        stale_path.unlink(missing_ok=True)
    shutil.rmtree(frame_directory, ignore_errors=True)
    frame_directory.mkdir(parents=True)
    run_status, run_reason = run_chimerax_script(script_path=script_path, log_path=log_path)
    if run_status != "completed":
        return run_status, run_reason
    try:
        frame_count = encode_movie_frames(
            frame_directory=frame_directory,
            movie_path=movie_path,
            log_path=log_path,
            frame_width=CHIMERAX_MOVIE_WIDTH,
            frame_height=CHIMERAX_MOVIE_HEIGHT,
            frames_per_scene=CHIMERAX_ROTATION_FRAMES_PER_SCENE,
            hold_frames_per_scene=CHIMERAX_HOLD_FRAMES_PER_SCENE,
            frame_rate=CHIMERAX_MOVIE_FRAME_RATE,
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\nmovie_encoding_error: {type(error).__name__}: {error}\n")
        return "errored", f"Movie encoding failed; inspect {log_path.name}."
    payload = {
        "schema_id": "eco1_rt.communication_structure_story_render",
        "schema_version": 3,
        "status": "rendered",
        "script_hash": "sha256:" + sha256(script_path),
        "reference_hash": "sha256:" + sha256(reference_structure_path),
        "movie_encoding": {
            "renderer": "ChimeraX 16:9 PNG saves",
            "encoder": "ffmpeg",
            "background": "white",
            "frame_rate": CHIMERAX_MOVIE_FRAME_RATE,
            "frame_count": frame_count,
            "width": CHIMERAX_MOVIE_WIDTH,
            "height": CHIMERAX_MOVIE_HEIGHT,
            "rotation_degrees_per_scene": CHIMERAX_ROTATION_DEGREES_PER_SCENE,
            "rotation_frames_per_scene": CHIMERAX_ROTATION_FRAMES_PER_SCENE,
        },
        "outputs": {
            "structure_story_movie": {"path": movie_path.name, "sha256": "sha256:" + sha256(movie_path)},
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
) -> int:
    """Encode checked fixed-size PNG scenes into an H.264 MP4."""

    frame_paths = sorted(frame_directory.glob("frame-*.png"))
    if not frame_paths:
        raise RuntimeError(f"No ChimeraX movie frames were written to {frame_directory}")
    if len(frame_paths) % frames_per_scene:
        raise RuntimeError(
            "ChimeraX frame count does not contain complete scenes: "
            f"{len(frame_paths)} frames for {frames_per_scene} rotation frames per scene"
        )
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is required to composite ChimeraX movie frames")
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
            validate_white_frame_corners(
                frame_bytes,
                width=frame_width,
                height=frame_height,
                path=frame_path,
            )
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


def _existing_render_is_current(
    *,
    render_manifest_path: Path,
    script_path: Path,
    reference_structure_path: Path,
    movie_path: Path,
) -> bool:
    if not all(path.exists() for path in (render_manifest_path, movie_path)):
        return False
    payload = yaml.safe_load(render_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return False
    return payload.get("script_hash") == "sha256:" + sha256(script_path) and payload.get(
        "reference_hash"
    ) == "sha256:" + sha256(reference_structure_path)


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
    command = [executable, "--script", str(script_path)]
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
