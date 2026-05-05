"""
Notebook rendering helpers for generated LatentDNA marimo notebooks.
"""

from __future__ import annotations

import base64
import re
from functools import lru_cache
from io import BytesIO, StringIO
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import marimo as mo
import matplotlib.pyplot as plt

from ..visual_style import (
    DEFAULT_NOTEBOOK_FIG_DPI,
    PANEL_BACKGROUND_COLOR,
    PLOT_FONT_FAMILY,
)

PREFERRED_RASTER_PLOT_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
PREFERRED_PLOT_RENDER_SUFFIXES = (".svg", *PREFERRED_RASTER_PLOT_SUFFIXES, ".pdf")
MAX_INLINE_SVG_BYTES = 5_000_000
MAX_INLINE_NOTEBOOK_ASSET_BYTES = 600_000
_DISPLAY_MATH_RE = re.compile(r"\$\$(.*?)\$\$", flags=re.DOTALL)
_PLOT_ASSET_WRAPPER_STYLE = (
    "width: 100%; overflow-x: auto; padding: 0.2rem 0 0.35rem 0; "
    "display: flex; align-items: flex-start; justify-content: center;"
)
_PLOT_ASSET_MEDIA_STYLE = (
    f"display: block; width: auto; height: auto; max-width: 100%; flex: 0 1 auto; "
    f"border-radius: 14px; background: {PANEL_BACKGROUND_COLOR};"
)
_MATH_BLOCK_STYLE = "padding: 0.1rem 0 0.25rem 0;"
_MATH_IMAGE_STYLE = "display: block; max-width: 100%; height: auto;"
_MATH_COMMAND_NORMALIZATIONS = (
    (r"\\le\b", r"\\leq"),
    (r"\\ge\b", r"\\geq"),
)


def _image_data_uri(image_bytes: bytes, *, suffix: str) -> str:
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _inline_plot_image(image_bytes: bytes, *, suffix: str, alt: str) -> mo.Html:
    return mo.Html(
        (
            '<div class="latentdna-plot-asset" role="img" aria-label="'
            + escape(alt)
            + '" style="'
            + escape(_PLOT_ASSET_WRAPPER_STYLE)
            + '"><img src="'
            + _image_data_uri(image_bytes, suffix=suffix)
            + '" alt="'
            + escape(alt)
            + '" style="'
            + escape(_PLOT_ASSET_MEDIA_STYLE)
            + '" /></div>'
        )
    )


def _svg_data_uri(svg_markup: str) -> str:
    encoded = base64.b64encode(svg_markup.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def fig_to_svg(fig, *, dpi: int = DEFAULT_NOTEBOOK_FIG_DPI, alt: str = "latent geometry plot"):
    buf = StringIO()
    fig.patch.set_facecolor(PANEL_BACKGROUND_COLOR)
    fig.patch.set_alpha(1.0)
    fig.savefig(
        buf,
        format="svg",
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    svg_markup = buf.getvalue()
    plt.close(fig)
    return mo.Html(
        (
            '<div class="latentdna-plot-asset" role="img" aria-label="'
            + escape(alt)
            + '" style="'
            + escape(_PLOT_ASSET_WRAPPER_STYLE)
            + '"><img src="'
            + _svg_data_uri(svg_markup)
            + '" alt="'
            + escape(alt)
            + '" style="'
            + escape(_PLOT_ASSET_MEDIA_STYLE)
            + '" /></div>'
        )
    )


def render_matplotlib_figure(fig, *, alt: str = "latent geometry plot"):
    if mo.app_meta().mode == "run":
        return fig_to_svg(fig, alt=alt)
    return mo.mpl.interactive(fig)


def select_plot_render_path(plot_files: Iterable[Path]) -> Path | None:
    existing_paths = [Path(path) for path in plot_files if Path(path).is_file()]
    for suffix in PREFERRED_PLOT_RENDER_SUFFIXES:
        candidate = next((path for path in existing_paths if path.suffix.lower() == suffix), None)
        if candidate is not None:
            return candidate
    return existing_paths[0] if existing_paths else None


def _is_same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _alternate_plot_render_path(path: Path) -> Path | None:
    for suffix in PREFERRED_RASTER_PLOT_SUFFIXES:
        candidate = path.with_suffix(suffix)
        if _is_same_path(candidate, path) or not candidate.is_file():
            continue
        if candidate.stat().st_size <= MAX_INLINE_NOTEBOOK_ASSET_BYTES:
            return candidate
    pdf_candidate = path.with_suffix(".pdf")
    if not _is_same_path(pdf_candidate, path) and pdf_candidate.is_file():
        return pdf_candidate
    return None


def resolve_plot_render_asset(path: Path) -> tuple[Path | None, str | None]:
    if not path.is_file():
        return None, f"Plot asset is missing: `{path.name}`."
    asset_size = path.stat().st_size
    suffix = path.suffix.lower()
    if suffix in PREFERRED_RASTER_PLOT_SUFFIXES:
        if asset_size <= MAX_INLINE_NOTEBOOK_ASSET_BYTES:
            return path, None
        alternate_path = _alternate_plot_render_path(path)
        if alternate_path is not None:
            return (
                alternate_path,
                (
                    f"Displaying `{alternate_path.name}` because `{path.name}` exceeds the inline notebook "
                    f"asset limit ({asset_size:,} bytes)."
                ),
            )
        return (
            None,
            (
                f"`{path.name}` exceeds the inline notebook asset limit ({asset_size:,} bytes) "
                "and no PDF alternate is available."
            ),
        )
    if suffix == ".svg":
        if asset_size <= MAX_INLINE_NOTEBOOK_ASSET_BYTES:
            return path, None
        alternate_path = _alternate_plot_render_path(path)
        if alternate_path is not None:
            return (
                alternate_path,
                (
                    f"Displaying `{alternate_path.name}` because `{path.name}` is large for inline notebook "
                    f"rendering ({asset_size:,} bytes)."
                ),
            )
        limit_name = "inline SVG" if asset_size > MAX_INLINE_SVG_BYTES else "inline notebook asset"
        return (
            None,
            (
                f"`{path.name}` exceeds the {limit_name} limit ({asset_size:,} bytes) "
                "and no raster or PDF alternate is available."
            ),
        )
    return path, None


def render_plot_asset(path: Path, *, workspace_dir: Path, alt_text: str | None = None):
    render_path, notice = resolve_plot_render_asset(path)
    if render_path is None:
        return mo.md(notice or f"`{path.relative_to(workspace_dir).as_posix()}`")
    alt = str(alt_text or render_path.name)
    suffix = render_path.suffix.lower()
    if suffix == ".svg":
        svg_html = render_path.read_text(encoding="utf-8")
        rendered = mo.Html(
            (
                "<div class='latentdna-plot-asset' role='img' aria-label='"
                + escape(alt)
                + "' style='"
                + escape(_PLOT_ASSET_WRAPPER_STYLE)
                + "'>"
                f"<img src='{_svg_data_uri(svg_html)}' alt='{escape(alt)}' style='{escape(_PLOT_ASSET_MEDIA_STYLE)}' />"
                "</div>"
            )
        )
    elif suffix == ".pdf":
        rendered = mo.pdf(
            src=render_path,
            width="100%",
            height="78vh",
            style={
                "border-radius": "14px",
                "background": PANEL_BACKGROUND_COLOR,
            },
        )
    elif suffix in PREFERRED_RASTER_PLOT_SUFFIXES:
        rendered = _inline_plot_image(render_path.read_bytes(), suffix=suffix, alt=alt)
    else:
        rendered = mo.md(f"`{render_path.relative_to(workspace_dir).as_posix()}`")
    if notice:
        return mo.vstack([mo.callout(notice, kind="warn"), rendered], gap=0.25)
    return rendered


@lru_cache(maxsize=256)
def _render_math_svg_bytes(expression: str) -> bytes:
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image

    normalized = " ".join(str(expression).split()).strip()
    for pattern, replacement in _MATH_COMMAND_NORMALIZATIONS:
        normalized = re.sub(pattern, replacement, normalized)
    if not normalized:
        return b""
    buffer = BytesIO()
    math_to_image(
        f"${normalized}$",
        buffer,
        format="svg",
        dpi=220,
        prop=FontProperties(size=15.0, family=PLOT_FONT_FAMILY),
        color="#E5EDF5",
    )
    return buffer.getvalue()


def _clean_markdown_prose(text: str) -> str:
    return text.replace(r"\(", "").replace(r"\)", "").strip()


def _transparent_math_svg_markup(svg_markup: str) -> str:
    return re.sub(
        r'(<g id="patch_1">\s*<path\b[^>]*)(/>)',
        r'\1 style="fill: none; stroke: none;"\2',
        svg_markup,
        count=1,
        flags=re.DOTALL,
    )


def render_math_markdown(text: str):
    content = str(text or "").strip()
    if not content:
        return mo.md("")
    parts: list[object] = []
    cursor = 0
    for match in _DISPLAY_MATH_RE.finditer(content):
        prose = _clean_markdown_prose(content[cursor : match.start()])
        if prose:
            parts.append(mo.md(prose))
        formula = match.group(1).strip()
        svg_bytes = _render_math_svg_bytes(formula)
        if svg_bytes:
            svg_markup = _transparent_math_svg_markup(svg_bytes.decode("utf-8"))
            parts.append(
                mo.Html(
                    (
                        "<div class='latentdna-math-block' style='" + escape(_MATH_BLOCK_STYLE) + "'>"
                        f"<img src='{_svg_data_uri(svg_markup)}' alt='Math expression' "
                        f"style='{escape(_MATH_IMAGE_STYLE)}' />"
                        "</div>"
                    )
                )
            )
        cursor = match.end()
    trailing = _clean_markdown_prose(content[cursor:])
    if trailing:
        parts.append(mo.md(trailing))
    if not parts:
        return mo.md(_clean_markdown_prose(content))
    if len(parts) == 1:
        return parts[0]
    return mo.vstack(parts, gap=0.25)
