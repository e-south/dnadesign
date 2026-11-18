"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/outputs/raster.py

Rasterize admitted native SVG at a final, explicit Inkscape output boundary.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import io
import math
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from PIL import Image, UnidentifiedImageError

from ..core import RenderingError
from ..core.vector import PngRenderResult, RasterizerInfo, ResolvedFont, SvgRenderDiagnostics, SvgRenderResult
from ..render.fonts import resolve_font
from ..render.svg_assets import admit_composed_svg


def _positive_number(value, name):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise RenderingError(f"{name} must be a positive finite number")


def _run(command, *, timeout_s, payload=None):
    try:
        result = subprocess.run(command, input=payload, capture_output=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RenderingError(f"Inkscape timed out after {timeout_s:g} s") from exc
    except OSError as exc:
        raise RenderingError("Inkscape could not execute; check the selected executable") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[-1000:]
        raise RenderingError(f"Inkscape exited with status {result.returncode}: {detail}")
    return result.stdout


def rasterize_svg(
    source: SvgRenderResult,
    *,
    dpi: float = 180,
    inkscape: str | Path = "inkscape",
    timeout_s: float = 30,
    max_pixels: int = 64_000_000,
) -> PngRenderResult:
    """Convert the existing composition to RGBA PNG, without layout or fitting."""
    if not isinstance(source, SvgRenderResult) or not isinstance(source.diagnostics, SvgRenderDiagnostics):
        raise RenderingError("rasterize_svg requires a measured SvgRenderResult")
    _positive_number(dpi, "dpi")
    _positive_number(timeout_s, "timeout_s")
    _positive_number(source.width_pt, "source.width_pt")
    _positive_number(source.height_pt, "source.height_pt")
    if not isinstance(max_pixels, int) or isinstance(max_pixels, bool) or max_pixels <= 0:
        raise RenderingError("max_pixels must be a positive integer pixel budget")
    document = admit_composed_svg(source.svg, width_pt=source.width_pt, height_pt=source.height_pt)
    # The serialized page is the conversion authority. Its six-decimal point
    # dimensions can differ slightly from the unrounded layout diagnostics.
    w, h = document.width_pt * dpi / 72, document.height_pt * dpi / 72
    if not math.isfinite(w) or not math.isfinite(h) or math.ceil(w) * math.ceil(h) > max_pixels:
        raise RenderingError(
            f"native PNG exceeds the {max_pixels:,} pixel budget; reduce dpi or explicitly increase max_pixels"
        )
    if min(w, h) < 1:
        raise RenderingError("native PNG dimensions must each cover at least one pixel at the selected dpi")
    if (
        not isinstance(source.diagnostics.fonts, Mapping)
        or not {"sequence", "label"}.issubset(source.diagnostics.fonts)
        or any(not isinstance(font, ResolvedFont) for font in source.diagnostics.fonts.values())
    ):
        raise RenderingError("native SVG must retain its declared font identities")
    for role, declared in source.diagnostics.fonts.items():
        current = resolve_font(declared.requested_family)
        if current != declared:
            raise RenderingError(f"declared font {role!r} changed since SVG preparation; regenerate explicitly")
    if not isinstance(inkscape, (str, Path)) or not str(inkscape).strip():
        raise RenderingError("inkscape must name an executable or an explicit executable path")
    executable = shutil.which(str(inkscape))
    if executable is None:
        raise RenderingError(
            "Inkscape is unavailable; make its CLI available on PATH or pass inkscape=<executable path>"
        )
    path = Path(executable).resolve()
    try:
        digest = sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RenderingError("Inkscape executable cannot be read for identity") from exc
    version = _run([str(path), "--version"], timeout_s=timeout_s).decode("utf-8", errors="replace").strip()
    if not version.startswith("Inkscape "):
        raise RenderingError("selected executable did not report an Inkscape version")
    png = _run(
        [
            str(path),
            "--pipe",
            "--export-type=png",
            "--export-filename=-",
            "--export-area-page",
            f"--export-dpi={dpi:.12g}",
            "--export-background-opacity=0",
            "--export-png-color-mode=RGBA_8",
        ],
        timeout_s=timeout_s,
        payload=document.payload,
    )
    try:
        image = Image.open(io.BytesIO(png))
        if image.format != "PNG" or image.mode != "RGBA":
            raise RenderingError("Inkscape did not produce an RGBA PNG")
        width, height = image.size
        if width * height > max_pixels or abs(width - w) > 0.50001 or abs(height - h) > 0.50001:
            raise RenderingError(f"PNG dimensions {width} x {height} disagree with measured {w:g} x {h:g} px")
        image.verify()
        with Image.open(io.BytesIO(png)) as decoded:
            decoded.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise RenderingError("Inkscape returned an invalid or incomplete PNG") from exc
    return PngRenderResult(
        png,
        width,
        height,
        float(dpi),
        sha256(source.svg.encode("utf-8")).hexdigest(),
        source.diagnostics,
        RasterizerInfo(str(path), digest, version),
    )
