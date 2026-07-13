#!/usr/bin/env python3
"""Render a py3Dmol HTML artifact in Chrome and verify scene pixels and audit data."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROLE_COLORS = {
    "dna": (185, 119, 0),
    "rna": (200, 76, 90),
}
EXPECTED_REPRESENTATION = "backbone_ribbon_with_base_spokes"
EXPECTED_RIBBON_WIDTH = 1.35
EXPECTED_RIBBON_THICKNESS = 0.28
_AUDIT_PATTERN = re.compile(
    r'<script id="dnadesign-structure-scene-audit" type="application/json">(.*?)</script>',
    flags=re.DOTALL,
)


def _chrome_path(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path(shutil.which("google-chrome") or ""),
        Path(shutil.which("chromium") or ""),
    ]
    for candidate in candidates:
        if candidate is not None and str(candidate) and candidate.is_file():
            return candidate
    raise RuntimeError("A Chrome or Chromium executable is required for WebGL verification")


def _scene_audits(path: Path) -> list[dict[str, object]]:
    document = html.unescape(path.read_text(encoding="utf-8"))
    rows = [json.loads(value) for value in _AUDIT_PATTERN.findall(document)]
    if not rows:
        raise ValueError("No dnadesign py3Dmol scene audit was found in the HTML artifact")
    return rows


def _audit_failures(audits: list[dict[str, object]]) -> list[str]:
    failures: list[str] = []
    for index, audit in enumerate(audits):
        if audit.get("representation") != EXPECTED_REPRESENTATION:
            failures.append(f"scene {index}: representation is not {EXPECTED_REPRESENTATION}")
        geometry_rows = audit.get("nucleic_geometry")
        if not isinstance(geometry_rows, list) or not geometry_rows:
            failures.append(f"scene {index}: no nucleic geometry rows were declared")
            continue
        classes = {str(row.get("molecule_class")) for row in geometry_rows if isinstance(row, dict)}
        for molecule_class in ROLE_COLORS:
            if molecule_class not in classes:
                failures.append(f"scene {index}: {molecule_class} geometry is missing")
        for row in geometry_rows:
            if not isinstance(row, dict):
                failures.append(f"scene {index}: nucleic geometry row is not a mapping")
                continue
            nucleotide_count = int(row.get("nucleotide_count", 0))
            spoke_count = int(row.get("base_spoke_count", -1))
            mesh_count = int(row.get("ribbon_mesh_count", 0))
            vertex_count = int(row.get("ribbon_vertex_count", 0))
            triangle_count = int(row.get("ribbon_triangle_count", 0))
            segment_count = int(row.get("backbone_segment_count", -1))
            width = float(row.get("ribbon_width_angstrom", 0.0))
            thickness = float(row.get("ribbon_thickness_angstrom", 0.0))
            if nucleotide_count <= 0:
                failures.append(f"scene {index}: nucleic geometry has no nucleotides")
            if spoke_count != nucleotide_count:
                failures.append(f"scene {index}: one base spoke per nucleotide is required")
            if nucleotide_count > 1 and mesh_count <= 0:
                failures.append(f"scene {index}: multi-residue nucleic geometry has no ribbon mesh")
            if mesh_count > 0 and (vertex_count <= 0 or triangle_count <= 0):
                failures.append(f"scene {index}: ribbon mesh has no vertices or triangles")
            if segment_count < max(0, nucleotide_count - mesh_count):
                failures.append(f"scene {index}: backbone segment count is incomplete")
            if width != EXPECTED_RIBBON_WIDTH or thickness != EXPECTED_RIBBON_THICKNESS:
                failures.append(
                    f"scene {index}: ribbon dimensions must be "
                    f"{EXPECTED_RIBBON_WIDTH} x {EXPECTED_RIBBON_THICKNESS} A; observed {width} x {thickness} A"
                )
            if width <= thickness:
                failures.append(f"scene {index}: nucleic ribbon is not wider than it is thick")
    return failures


def _render_chrome(*, chrome: Path, html_path: Path, screenshot_path: Path, width: int, height: int) -> str:
    command = [
        str(chrome),
        "--headless=new",
        "--enable-webgl",
        "--ignore-gpu-blocklist",
        "--use-angle=metal" if sys.platform == "darwin" else "--use-angle=swiftshader",
        "--virtual-time-budget=8000",
        f"--window-size={width},{height}",
        f"--screenshot={screenshot_path}",
        html_path.resolve().as_uri(),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    if completed.returncode != 0 or not screenshot_path.is_file():
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Chrome WebGL capture failed ({completed.returncode}): {message}")
    return (completed.stderr or completed.stdout).strip()


def _pixel_counts(path: Path, *, tolerance: int) -> tuple[int, dict[str, int]]:
    image = Image.open(path).convert("RGB")
    pixels = tuple(image.get_flattened_data())
    nonwhite = sum(1 for pixel in pixels if min(pixel) < 245)
    role_counts = {
        role: sum(
            1 for pixel in pixels if all(abs(pixel[channel] - target[channel]) <= tolerance for channel in range(3))
        )
        for role, target in ROLE_COLORS.items()
    }
    return nonwhite, role_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, required=True)
    parser.add_argument("--chrome", type=Path)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--color-tolerance", type=int, default=25)
    parser.add_argument("--minimum-role-pixels", type=int, default=100)
    args = parser.parse_args()

    failures = _audit_failures(_scene_audits(args.html))
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    chrome_log = _render_chrome(
        chrome=_chrome_path(args.chrome),
        html_path=args.html,
        screenshot_path=args.screenshot,
        width=args.width,
        height=args.height,
    )
    nonwhite_pixels, role_pixels = _pixel_counts(args.screenshot, tolerance=args.color_tolerance)
    if nonwhite_pixels < args.minimum_role_pixels:
        failures.append("rendered screenshot is blank or nearly blank")
    for role, count in role_pixels.items():
        if count < args.minimum_role_pixels:
            failures.append(f"rendered {role} color is absent or too sparse: {count} pixels")
    payload = {
        "schema_id": "dnadesign_py3dmol_webgl_verification_v1",
        "status": "pass" if not failures else "fail",
        "html": str(args.html.resolve()),
        "screenshot": str(args.screenshot.resolve()),
        "nonwhite_pixels": nonwhite_pixels,
        "role_pixels": role_pixels,
        "chrome_log": chrome_log,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
