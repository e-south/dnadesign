#!/usr/bin/env python3
"""Verify ChimeraX stills, frame series, and encoded movie metadata."""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

_COLOR_NAMES = {"black": (0, 0, 0), "white": (255, 255, 255)}


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found in {path}")
    stream = dict(streams[0])
    stream["duration"] = (payload.get("format") or {}).get("duration")
    return stream


def _sample_rgb(path: Path, *, width: int, height: int) -> bytes:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"scale={width}:{height}:flags=area,format=rgb24",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    expected = width * height * 3
    if len(completed.stdout) != expected:
        raise ValueError(f"Expected {expected} RGB bytes from {path}, got {len(completed.stdout)}")
    return completed.stdout


def _parse_color(value: str) -> tuple[int, int, int]:
    normalized = value.strip().lower()
    if normalized in _COLOR_NAMES:
        return _COLOR_NAMES[normalized]
    if normalized.startswith("#") and len(normalized) == 7:
        try:
            return tuple(int(normalized[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid RGB color: {value}") from exc
    raise argparse.ArgumentTypeError("Background color must be white, black, or #RRGGBB")


def _near_color(pixel: tuple[int, int, int], expected: tuple[int, int, int], tolerance: int) -> bool:
    return max(abs(pixel[index] - expected[index]) for index in range(3)) <= tolerance


def _verify_image(
    path: Path,
    *,
    expected_width: int | None,
    expected_height: int | None,
    background: tuple[int, int, int],
    tolerance: int,
    minimum_content_fraction: float,
    minimum_content_extent: float,
) -> dict[str, Any]:
    stream = _probe(path)
    width = int(stream["width"])
    height = int(stream["height"])
    failures: list[str] = []
    if expected_width is not None and width != expected_width:
        failures.append(f"width {width} != {expected_width}")
    if expected_height is not None and height != expected_height:
        failures.append(f"height {height} != {expected_height}")

    sample_width = min(width, 96)
    sample_height = min(height, 96)
    raw = _sample_rgb(path, width=sample_width, height=sample_height)
    pixels = [tuple(raw[index : index + 3]) for index in range(0, len(raw), 3)]
    corner_indices = (0, sample_width - 1, (sample_height - 1) * sample_width, sample_height * sample_width - 1)
    corners = [pixels[index] for index in corner_indices]
    if not all(_near_color(pixel, background, tolerance) for pixel in corners):
        failures.append(f"one or more corners do not match background {background} within tolerance {tolerance}")

    content_coordinates = [
        (index % sample_width, index // sample_width)
        for index, pixel in enumerate(pixels)
        if not _near_color(pixel, background, tolerance)
    ]
    content_fraction = len(content_coordinates) / len(pixels)
    if content_fraction < minimum_content_fraction:
        failures.append(f"content fraction {content_fraction:.4f} < {minimum_content_fraction:.4f}")
    content_extent = (0.0, 0.0)
    if content_coordinates:
        x_values = [coord[0] for coord in content_coordinates]
        y_values = [coord[1] for coord in content_coordinates]
        content_extent = (
            (max(x_values) - min(x_values) + 1) / sample_width,
            (max(y_values) - min(y_values) + 1) / sample_height,
        )
        if min(content_extent) < minimum_content_extent:
            failures.append(
                f"content extent {content_extent[0]:.3f} x {content_extent[1]:.3f} is below "
                f"minimum {minimum_content_extent:.3f}"
            )
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "corners_rgb": corners,
        "content_fraction": round(content_fraction, 6),
        "content_extent_fraction": [round(value, 6) for value in content_extent],
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }


def _verify_movie(
    path: Path,
    *,
    expected_width: int | None,
    expected_height: int | None,
    expected_frame_count: int | None,
    expected_duration_seconds: float | None,
    duration_tolerance_seconds: float,
) -> dict[str, Any]:
    stream = _probe(path)
    width = int(stream["width"])
    height = int(stream["height"])
    failures: list[str] = []
    if expected_width is not None and width != expected_width:
        failures.append(f"width {width} != {expected_width}")
    if expected_height is not None and height != expected_height:
        failures.append(f"height {height} != {expected_height}")
    frame_count = int(stream["nb_frames"]) if str(stream.get("nb_frames") or "").isdigit() else None
    duration = float(stream["duration"]) if stream.get("duration") not in {None, "N/A"} else None
    if expected_frame_count is not None and frame_count != expected_frame_count:
        failures.append(f"frame count {frame_count} != {expected_frame_count}")
    if expected_duration_seconds is not None:
        if duration is None or abs(duration - expected_duration_seconds) > duration_tolerance_seconds:
            failures.append(
                f"duration {duration} differs from {expected_duration_seconds} by more than "
                f"{duration_tolerance_seconds} seconds"
            )
    frame_rate_text = str(stream.get("avg_frame_rate") or "0/1")
    frame_rate = float(Fraction(frame_rate_text)) if frame_rate_text != "0/0" else 0.0
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "frame_rate": round(frame_rate, 6),
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify dimensions, framing, and metadata for ChimeraX renders.")
    parser.add_argument("--image", action="append", type=Path, default=[])
    parser.add_argument("--frame-glob", action="append", default=[])
    parser.add_argument("--movie", type=Path)
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    parser.add_argument("--expected-frame-count", type=int)
    parser.add_argument("--expected-duration-seconds", type=float)
    parser.add_argument("--duration-tolerance-seconds", type=float, default=0.05)
    parser.add_argument("--background", type=_parse_color, default=_parse_color("white"))
    parser.add_argument("--color-tolerance", type=int, default=12)
    parser.add_argument("--minimum-content-fraction", type=float, default=0.002)
    parser.add_argument("--minimum-content-extent", type=float, default=0.0)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        parser.error("ffmpeg and ffprobe are required")
    if args.expected_width is not None and args.expected_width <= 0:
        parser.error("--expected-width must be positive")
    if args.expected_height is not None and args.expected_height <= 0:
        parser.error("--expected-height must be positive")
    if not 0 <= args.color_tolerance <= 255:
        parser.error("--color-tolerance must be from 0 to 255")
    for name, value in (
        ("--minimum-content-fraction", args.minimum_content_fraction),
        ("--minimum-content-extent", args.minimum_content_extent),
    ):
        if not 0 <= value <= 1:
            parser.error(f"{name} must be from 0 to 1")

    image_paths = list(args.image)
    for pattern in args.frame_glob:
        image_paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    if not image_paths and args.movie is None:
        parser.error("provide at least one --image, --frame-glob, or --movie")
    for path in [*image_paths, *([args.movie] if args.movie is not None else [])]:
        if not path.exists():
            parser.error(f"render does not exist: {path}")

    image_results = [
        _verify_image(
            path,
            expected_width=args.expected_width,
            expected_height=args.expected_height,
            background=args.background,
            tolerance=args.color_tolerance,
            minimum_content_fraction=args.minimum_content_fraction,
            minimum_content_extent=args.minimum_content_extent,
        )
        for path in image_paths
    ]
    movie_result = None
    if args.movie is not None:
        movie_result = _verify_movie(
            args.movie,
            expected_width=args.expected_width,
            expected_height=args.expected_height,
            expected_frame_count=args.expected_frame_count,
            expected_duration_seconds=args.expected_duration_seconds,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
        )
    results = [*image_results, *([movie_result] if movie_result is not None else [])]
    payload = {
        "schema_id": "chimerax_render_verification_v1",
        "status": "pass" if all(result["status"] == "pass" for result in results) else "fail",
        "images": image_results,
        "movie": movie_result,
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
