"""
Shared browser runtime support helpers for generated latentdna marimo notebooks.
"""

from __future__ import annotations

import base64
import json
import re
from functools import lru_cache
from io import BytesIO, StringIO
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..annotation_layout import choose_annotation_placement
from ..contracts.notebook import WorkspaceNotebookControls
from ..labels import humanize_column_name
from ..visual_style import (
    ANNOTATION_LABEL_BOX_ALPHA,
    DEFAULT_NOTEBOOK_FIG_DPI,
    GRID_COLOR,
    NOTEBOOK_FONT_STACK,
    PANEL_BACKGROUND_COLOR,
    PLOT_FONT_FAMILY,
    PLOT_LABEL_FONT_SIZE,
    PLOT_LEGEND_FONT_SIZE,
    PLOT_TICK_FONT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    SPINE_COLOR,
    TEXT_COLOR,
    categorical_color_map,
    display_category_text,
    humanize_display_text,
    ordered_categories,
)
from ..visual_style import scatter_style as shared_scatter_style

REFERENCE_DISPLAY = {
    "spyp": "spyP",
    "sulap": "sulAp",
    "j23105": "J23105",
}

PREFERRED_RASTER_PLOT_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
PREFERRED_PLOT_RENDER_SUFFIXES = (".svg", *PREFERRED_RASTER_PLOT_SUFFIXES, ".pdf")
MAX_INLINE_SVG_BYTES = 5_000_000
MAX_INLINE_NOTEBOOK_ASSET_BYTES = 600_000
NOTEBOOK_MEDIA_MAX_WIDTH_PX = 1400
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


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def read_text(path_text: str | None) -> str | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def load_workspace_notebook_controls(control_path: Path) -> dict[str, object]:
    return WorkspaceNotebookControls.model_validate(load_json(control_path)).model_dump(mode="json")


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


def fig_to_image(fig, *, dpi: int = DEFAULT_NOTEBOOK_FIG_DPI, alt: str = "latent geometry plot"):
    buf = BytesIO()
    fig.patch.set_facecolor(PANEL_BACKGROUND_COLOR)
    fig.patch.set_alpha(1.0)
    fig.savefig(
        buf,
        format="png",
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    image_bytes = buf.getvalue()
    plt.close(fig)
    return _inline_plot_image(image_bytes, suffix=".png", alt=alt)


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
    # marimo app-mode iframes can fail to hydrate interactive mpl payloads; prefer
    # inline SVG there so the audit surfaces stay usable without frontend JS errors.
    if mo.app_meta().mode == "run":
        try:
            return fig_to_svg(fig, alt=alt)
        except Exception:
            return fig_to_image(fig, alt=alt)
    try:
        return mo.mpl.interactive(fig)
    except Exception:
        try:
            return fig_to_svg(fig, alt=alt)
        except Exception:
            return fig_to_image(fig, alt=alt)


def notebook_theme():
    return mo.Html(
        f"""
        <style>
          .latentdna-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 5.5rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 650;
            letter-spacing: 0.02em;
            font-family: {NOTEBOOK_FONT_STACK};
            color: {TEXT_COLOR};
            background: rgba(226, 232, 240, 0.82);
          }}

          .latentdna-badge--primary {{
            background: rgba(59, 130, 246, 0.12);
          }}

          .latentdna-badge--appendix {{
            background: rgba(236, 201, 75, 0.22);
          }}
        </style>
        """
    )


def select_plot_render_path(plot_files: Iterable[Path]) -> Path | None:
    existing_paths = [Path(path) for path in plot_files if Path(path).is_file()]
    for suffix in PREFERRED_PLOT_RENDER_SUFFIXES:
        candidate = next((path for path in existing_paths if path.suffix.lower() == suffix), None)
        if candidate is not None:
            return candidate
    return existing_paths[0] if existing_paths else None


def _fallback_plot_render_path(path: Path) -> Path | None:
    for suffix in (*PREFERRED_RASTER_PLOT_SUFFIXES, ".pdf"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def resolve_plot_render_asset(path: Path) -> tuple[Path | None, str | None]:
    if not path.is_file():
        return None, f"Plot asset is missing: `{path.name}`."
    asset_size = path.stat().st_size
    if path.suffix.lower() == ".svg" and asset_size > MAX_INLINE_NOTEBOOK_ASSET_BYTES:
        fallback_path = _fallback_plot_render_path(path)
        if fallback_path is not None:
            return (
                fallback_path,
                (
                    f"Displaying `{fallback_path.name}` because `{path.name}` is large for inline notebook "
                    f"rendering ({asset_size:,} bytes)."
                ),
            )
    if path.suffix.lower() != ".svg":
        return path, None
    svg_size = asset_size
    if svg_size <= MAX_INLINE_SVG_BYTES:
        return path, None
    fallback_path = _fallback_plot_render_path(path)
    if fallback_path is not None:
        return (
            fallback_path,
            (
                f"Displaying `{fallback_path.name}` because `{path.name}` exceeds the inline notebook "
                f"limit ({svg_size:,} bytes)."
            ),
        )
    return (
        None,
        (
            f"`{path.name}` exceeds the inline notebook limit ({svg_size:,} bytes) "
            "and no raster or PDF fallback is available."
        ),
    )


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
        try:
            svg_bytes = _render_math_svg_bytes(formula)
        except ValueError:
            parts.append(
                mo.callout(
                    "Math rendering fell back to plain text for one formula because the syntax was not supported.",
                    kind="warn",
                )
            )
            parts.append(mo.md(f"`{formula}`"))
            cursor = match.end()
            continue
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


def unique_in_order(values):
    seen = set()
    ordered = []
    for value in values:
        key = str(value or "Unsectioned").strip() or "Unsectioned"
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def option_key_for_value(options: dict[str, object], target_value: object) -> str | None:
    target_text = str(target_value)
    for key, value in options.items():
        if str(value) == target_text:
            return key
    return None


def labeled_options(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    normalized: list[tuple[str, object]] = []
    counts: dict[str, int] = {}
    for label, value in pairs:
        value_text = str(value).strip()
        base_label = str(label).strip() or value_text
        normalized.append((base_label, value))
        counts[base_label] = counts.get(base_label, 0) + 1

    options: dict[str, object] = {}
    for base_label, value in normalized:
        value_text = str(value).strip()
        if not value_text:
            continue
        label = base_label if counts[base_label] == 1 else f"{base_label} [{value_text}]"
        if label in options:
            if str(options[label]).strip() == value_text:
                continue
            suffix = 2
            while f"{label} #{suffix}" in options:
                suffix += 1
            label = f"{label} #{suffix}"
        options[label] = value
    return options


def normalize_label(value) -> str:
    return str(value or "").strip().lower()


def display_reference_label(value) -> str:
    text = str(value or "")
    return REFERENCE_DISPLAY.get(normalize_label(text), text)


def display_hue_label(column: str) -> str:
    if column == "design_regulator_composition":
        return "Reg. comp."
    if column == "log_likelihood_per_token_7b":
        return "7B log likelihood / token"
    if column == "log_likelihood_per_token_20b":
        return "20B log likelihood / token"
    if column.startswith("log_likelihood_per_token_"):
        return humanize_display_text(column)
    if column.startswith("infer__evo2_") and "__log_likelihood__mean_per_token" in column:
        model = "7B" if "__7b__" in column else "20B"
        scope = "1 kb construct context" if "__template_1kb_" in column else "60 bp anchor"
        return f"{model} log likelihood / token ({scope})"
    if column.startswith("cluster_label__"):
        return humanize_display_text(column.replace("cluster_label__", ""))
    return humanize_column_name(column)


def display_hue_value(column: str | None, value: object) -> str:
    return display_category_text(value, column=column)


def normalize_categorical_hue_value(column: str | None, value: object) -> str:
    if pd.isna(value):
        return "NA"
    if str(column or "").strip() == "spacer_length":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and np.isfinite(numeric):
            return str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"
    if str(column or "").strip() == "design_regulator_composition":
        normalized = display_hue_value(column, value)
        return normalized or "NA"
    text = str(value)
    return text if text.strip() else "NA"


def normalize_categorical_hue_series(column: str | None, values: pd.Series) -> pd.Series:
    return values.map(lambda value: normalize_categorical_hue_value(column, value)).astype(str)


def resolve_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> tuple[str, str] | None:
    candidate_pairs = [
        ("construct__anchor_id", "construct__anchor_id"),
        ("construct__anchor_id", "id"),
        # Anchor-only projection rows should join context summary tables by anchor id.
        ("id", "construct__anchor_id"),
        ("id", "id"),
        ("subject_id", "subject_id"),
        ("context_id", "context_id"),
    ]
    for left_key, right_key in candidate_pairs:
        if left_key in left.columns and right_key in right.columns:
            return left_key, right_key
    return None


def shared_join_key(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    resolved = resolve_join_keys(left, right)
    if resolved is None:
        return None
    left_key, right_key = resolved
    if left_key == right_key:
        return left_key
    return None


def geometry_map(geometry_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["view_id"]): row for row in geometry_rows}


def load_view_rows(view_id: str, *, output_root: Path) -> pd.DataFrame:
    return load_table(output_root / "views" / view_id / "rows.parquet")


def load_view_matrix(view_id: str, *, output_root: Path):
    matrix_path = output_root / "views" / view_id / "matrix.npy"
    if not matrix_path.is_file():
        return None
    return np.load(matrix_path, mmap_mode="r")


def include_hue_column(column: str, artifact_suffixes: set[str] | None = None) -> bool:
    blocked = {"x", "y", "left_count", "right_count", "left_indices", "right_indices"}
    if column in blocked:
        return False
    if column.startswith(("left_count__", "right_count__", "left_indices__", "right_indices__")):
        return False
    if column.startswith("cluster_label__"):
        return True
    if artifact_suffixes:
        suffix = column.rsplit("__", 1)[-1]
        if suffix in artifact_suffixes:
            return False
    return True


def candidate_hue_columns(
    frame: pd.DataFrame, preferred: list[str], artifact_suffixes: set[str] | None = None
) -> list[str]:
    if frame.empty:
        return []
    return [column for column in preferred if column in frame.columns and include_hue_column(column, artifact_suffixes)]


def normalize_hue_kind(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"categorical", "binary", "continuous", "ordinal"}:
        return text
    return None


def _finite_non_null_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=object)
    return frame[column].replace([np.inf, -np.inf], np.nan).dropna()


def available_hues_for_frames(
    frames: list[pd.DataFrame],
    *,
    preferred_hues: list[str],
    hue_kinds: dict[str, str],
) -> list[str]:
    if not frames:
        return []
    nonempty_frames = [frame for frame in frames if not frame.empty]
    if not nonempty_frames:
        return []
    available: list[str] = []
    for hue in preferred_hues:
        kind = normalize_hue_kind(hue_kinds.get(hue))
        if kind is None:
            continue
        continuous_values: list[pd.Series] = []
        supported = True
        for frame in nonempty_frames:
            series = _finite_non_null_series(frame, hue)
            if series.empty:
                supported = False
                break
            if kind == "continuous":
                numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if numeric.empty:
                    supported = False
                    break
                continuous_values.append(numeric)
        if not supported:
            continue
        if kind == "continuous":
            combined = pd.concat(continuous_values, ignore_index=True) if continuous_values else pd.Series(dtype=float)
            if combined.nunique() < 2:
                continue
        available.append(hue)
    return available


def classify_hue_series(series: pd.Series, *, configured_kind: object = None) -> str:
    explicit_kind = normalize_hue_kind(configured_kind)
    if explicit_kind is not None:
        return explicit_kind
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if (
        pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "continuous"
    return "categorical"


def continuous_hue_render_params(column: str | None, values: pd.Series) -> dict[str, object]:
    from matplotlib import colors as mcolors

    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {"cmap": "viridis", "norm": None, "vmin": None, "vmax": None}

    column_name = str(column or "").strip()
    if "margin" in column_name:
        robust_limit = float(np.nanquantile(np.abs(numeric.to_numpy(dtype=float)), 0.99))
        if not np.isfinite(robust_limit) or robust_limit <= 0.0:
            robust_limit = float(np.nanmax(np.abs(numeric.to_numpy(dtype=float))))
        robust_limit = max(robust_limit, 1e-6)
        return {
            "cmap": "coolwarm",
            "norm": mcolors.TwoSlopeNorm(vmin=-robust_limit, vcenter=0.0, vmax=robust_limit),
            "vmin": None,
            "vmax": None,
        }

    lower = float(np.nanquantile(numeric.to_numpy(dtype=float), 0.01))
    upper = float(np.nanquantile(numeric.to_numpy(dtype=float), 0.99))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower = float(numeric.min())
        upper = float(numeric.max())
    return {
        "cmap": "viridis",
        "norm": None,
        "vmin": lower,
        "vmax": upper,
    }


def scatter_style(row_count: int) -> tuple[float, float]:
    style = shared_scatter_style(row_count)
    return style.point_size, style.alpha


def category_color_map(categories: list[str], *, column: str | None = None) -> dict[str, str]:
    return categorical_color_map(ordered_categories(categories, column=column), column=column)


def table_from_records(
    records: pd.DataFrame | list[dict[str, object]],
    *,
    columns: list[str] | None = None,
    page_size: int | None = None,
):
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=columns)
    if columns is not None:
        frame = frame.reindex(columns=columns)
    if page_size is None:
        return mo.ui.table(frame)
    return mo.ui.table(frame, page_size=page_size)


def key_value_table(
    rows: list[tuple[str, object]],
    *,
    field_name: str = "Field",
    value_name: str = "Value",
):
    normalized_rows = [{field_name: str(field), value_name: value} for field, value in rows]
    return table_from_records(
        normalized_rows,
        columns=[field_name, value_name],
        page_size=min(max(len(normalized_rows), 1), 12),
    )


def style_notebook_axes(ax, *, grid: bool = True, square: bool = False) -> None:
    ax.set_facecolor(PANEL_BACKGROUND_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(colors=TEXT_COLOR, labelsize=PLOT_TICK_FONT_SIZE, length=4.5, width=0.8, direction="out")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_fontsize(PLOT_LABEL_FONT_SIZE)
    ax.yaxis.label.set_fontsize(PLOT_LABEL_FONT_SIZE)
    ax.title.set_color(TEXT_COLOR)
    ax.title.set_fontsize(PLOT_TITLE_FONT_SIZE)
    ax.title.set_fontweight("semibold")
    ax.title.set_fontfamily(PLOT_FONT_FAMILY)
    ax.margins(x=0.04, y=0.05)
    if square:
        ax.set_box_aspect(1)
    if grid:
        ax.grid(True, color=GRID_COLOR, linewidth=0.75, alpha=0.58)
        ax.set_axisbelow(True)


def style_notebook_legend(legend) -> None:
    if legend is None:
        return
    title = legend.get_title()
    if title is not None:
        title.set_visible(False)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
        text.set_fontsize(PLOT_LEGEND_FONT_SIZE)
        text.set_fontfamily(PLOT_FONT_FAMILY)


def draw_reference_labels(
    ax,
    frame: pd.DataFrame,
    *,
    reference_labels: list[str],
    x_column: str = "x",
    y_column: str = "y",
    right_padding_px: float = 0.0,
    left_padding_px: float = 0.0,
) -> None:
    if frame.empty or "usr_label__primary" not in frame.columns:
        return
    if x_column not in frame.columns or y_column not in frame.columns:
        return
    selected = frame[
        frame["usr_label__primary"]
        .astype(str)
        .map(normalize_label)
        .isin({normalize_label(label) for label in reference_labels})
    ].copy()
    if selected.empty:
        return
    placed_boxes: list[tuple[float, float, float, float]] = []
    axes_box = ax.get_window_extent()
    display_x_mid = float((axes_box.x0 + axes_box.x1) / 2.0)
    display_y_mid = float((axes_box.y0 + axes_box.y1) / 2.0)
    ax.scatter(
        selected[x_column].to_numpy(dtype=float),
        selected[y_column].to_numpy(dtype=float),
        c="#111111",
        s=125,
        marker="*",
        linewidths=0.8,
        edgecolors="white",
        zorder=5,
    )
    for row in selected.sort_values("usr_label__primary").to_dict(orient="records"):
        point_x = float(row[x_column])
        point_y = float(row[y_column])
        label = display_reference_label(row["usr_label__primary"])
        display_x, display_y = ax.transData.transform((point_x, point_y))
        placement = choose_annotation_placement(
            display_x=display_x,
            display_y=display_y,
            label_text=label,
            axes_box=axes_box,
            placed_boxes=placed_boxes,
            x_mid=display_x_mid,
            y_mid=display_y_mid,
            font_size=PLOT_TICK_FONT_SIZE,
            left_padding_px=left_padding_px,
            right_padding_px=right_padding_px,
        )
        placed_boxes.append(placement.box)
        annotation = ax.annotate(
            label,
            xy=(point_x, point_y),
            xytext=(placement.offset_x, placement.offset_y),
            textcoords="offset pixels",
            fontsize=PLOT_TICK_FONT_SIZE,
            fontweight="semibold",
            ha=placement.ha,
            va=placement.va,
            color=TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "none",
                "alpha": ANNOTATION_LABEL_BOX_ALPHA,
            },
            arrowprops={"arrowstyle": "-", "color": SPINE_COLOR, "linewidth": 0.9},
            zorder=6,
        )
        annotation.set_clip_on(True)
        if annotation.arrow_patch is not None:
            annotation.arrow_patch.set_clip_on(True)
