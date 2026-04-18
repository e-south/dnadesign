"""
Shared browser runtime support helpers for generated latentdna marimo notebooks.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO, StringIO
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..contracts.notebook import WorkspaceNotebookControls
from ..visual_style import (
    DEFAULT_NOTEBOOK_FIG_DPI,
    GRID_COLOR,
    NOTEBOOK_FONT_STACK,
    PANEL_BACKGROUND_COLOR,
    PLOT_LABEL_FONT_SIZE,
    PLOT_TICK_FONT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    SPINE_COLOR,
    TEXT_COLOR,
    categorical_color_map,
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
NOTEBOOK_MEDIA_MAX_WIDTH_PX = 900


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
    return mo.image(image_bytes, alt=alt, width=f"min(100%, {NOTEBOOK_MEDIA_MAX_WIDTH_PX}px)")


def _svg_data_uri(svg_markup: str) -> str:
    encoded = base64.b64encode(svg_markup.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def fig_to_svg(fig, *, alt: str = "latent geometry plot"):
    buf = StringIO()
    fig.patch.set_facecolor(PANEL_BACKGROUND_COLOR)
    fig.patch.set_alpha(1.0)
    fig.savefig(
        buf,
        format="svg",
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
            + '"><img src="'
            + _svg_data_uri(svg_markup)
            + '" alt="'
            + escape(alt)
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
          .latentdna-plot-asset {{
            --latentdna-surface: {PANEL_BACKGROUND_COLOR};
            --latentdna-media-max-width: {NOTEBOOK_MEDIA_MAX_WIDTH_PX}px;
            width: 100%;
            overflow-x: auto;
            padding: 0.2rem 0;
            display: flex;
            justify-content: center;
          }}

          .latentdna-plot-asset img,
          .latentdna-plot-asset canvas,
          .latentdna-plot-asset svg {{
            width: min(100%, var(--latentdna-media-max-width));
            max-width: min(100%, var(--latentdna-media-max-width));
            height: auto;
            display: block;
            margin-inline: auto;
            border-radius: 14px;
            background: var(--latentdna-surface);
          }}

          .latentdna-scope-note,
          .latentdna-plot-card,
          .latentdna-audit-note {{
            border: 1px solid rgba(92, 104, 116, 0.16);
            border-radius: 16px;
            background: rgba(248, 250, 252, 0.92);
            padding: 1rem 1.1rem;
          }}

          .latentdna-plot-card {{
            display: grid;
            gap: 0.7rem;
          }}

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


def resolve_plot_render_asset(path: Path) -> tuple[Path | None, str | None]:
    if not path.is_file():
        return None, f"Plot asset is missing: `{path.name}`."
    if path.suffix.lower() != ".svg":
        return path, None
    svg_size = path.stat().st_size
    if svg_size <= MAX_INLINE_SVG_BYTES:
        return path, None
    raster_paths = []
    for suffix in PREFERRED_RASTER_PLOT_SUFFIXES:
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            raster_paths.append(candidate)
    if raster_paths:
        raster_path = raster_paths[0]
        return (
            raster_path,
            (
                f"Displaying `{raster_path.name}` because `{path.name}` exceeds the inline notebook "
                f"limit ({svg_size:,} bytes)."
            ),
        )
    return (
        None,
        (f"`{path.name}` exceeds the inline notebook limit ({svg_size:,} bytes) and no raster fallback is available."),
    )


def render_plot_asset(path: Path, *, workspace_dir: Path):
    render_path, notice = resolve_plot_render_asset(path)
    if render_path is None:
        return mo.md(notice or f"`{path.relative_to(workspace_dir).as_posix()}`")
    suffix = render_path.suffix.lower()
    if suffix == ".svg":
        svg_html = render_path.read_text(encoding="utf-8")
        rendered = mo.Html(
            (
                "<div class='latentdna-plot-asset'>"
                f"<img src='{_svg_data_uri(svg_html)}' alt='{escape(render_path.name)}' />"
                "</div>"
            )
        )
    elif suffix in PREFERRED_RASTER_PLOT_SUFFIXES:
        rendered = mo.image(
            render_path.read_bytes(),
            alt=render_path.name,
            width=f"min(100%, {NOTEBOOK_MEDIA_MAX_WIDTH_PX}px)",
        )
    else:
        rendered = mo.md(f"`{render_path.relative_to(workspace_dir).as_posix()}`")
    if notice is None:
        return rendered
    return mo.vstack([mo.md(notice), rendered])


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


def normalize_label(value) -> str:
    return str(value or "").strip().lower()


def display_reference_label(value) -> str:
    text = str(value or "")
    return REFERENCE_DISPLAY.get(normalize_label(text), text)


def display_hue_label(column: str) -> str:
    if column.startswith("log_likelihood_per_token_"):
        return column.replace("log_likelihood_per_token_", "log likelihood per token ").replace("_", " ")
    if column.startswith("infer__evo2_") and "__log_likelihood__mean_per_token" in column:
        model = "7B" if "__7b__" in column else "20B"
        scope = "1 kb expanded-context" if "__template_1kb_" in column else "60 bp anchor-only"
        return f"{model} log likelihood / token ({scope})"
    if column.startswith("cluster_label__"):
        return column.replace("cluster_label__", "").replace("_", " ")
    return column.replace("_", " ")


def shared_join_key(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    preferred = ["construct__anchor_id", "id", "subject_id", "context_id"]
    for column in preferred:
        if column in left.columns and column in right.columns:
            return column
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
    if text in {"categorical", "binary", "continuous"}:
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
    available: list[str] = []
    for hue in preferred_hues:
        kind = normalize_hue_kind(hue_kinds.get(hue))
        if kind is None:
            continue
        continuous_values: list[pd.Series] = []
        supported = True
        for frame in frames:
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


def scatter_style(row_count: int) -> tuple[float, float]:
    style = shared_scatter_style(row_count)
    return style.point_size, style.alpha


def category_color_map(categories: list[str]) -> dict[str, str]:
    return categorical_color_map(ordered_categories(categories))


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
    ax.margins(x=0.04, y=0.05)
    if square:
        ax.set_box_aspect(1)
    if grid:
        ax.grid(True, color=GRID_COLOR, linewidth=0.75, alpha=0.58)
        ax.set_axisbelow(True)


def draw_reference_labels(
    ax,
    frame: pd.DataFrame,
    *,
    reference_labels: list[str],
    right_padding_px: float = 0.0,
    left_padding_px: float = 0.0,
) -> None:
    if frame.empty or "usr_label__primary" not in frame.columns:
        return
    selected = frame[
        frame["usr_label__primary"]
        .astype(str)
        .map(normalize_label)
        .isin({normalize_label(label) for label in reference_labels})
    ].copy()
    if selected.empty:
        return
    offsets = [
        (10.0, 10.0),
        (10.0, -18.0),
        (-72.0, 10.0),
        (-72.0, -18.0),
        (18.0, 24.0),
        (-80.0, 24.0),
        (18.0, -32.0),
        (-80.0, -32.0),
        (38.0, 0.0),
        (-92.0, 0.0),
    ]
    placed: list[tuple[float, float]] = []
    axes_box = ax.get_window_extent()
    ax.scatter(
        selected["x"].to_numpy(dtype=float),
        selected["y"].to_numpy(dtype=float),
        c="#111111",
        s=125,
        marker="*",
        linewidths=0.8,
        edgecolors="white",
        zorder=5,
    )
    for row in selected.sort_values("usr_label__primary").to_dict(orient="records"):
        point_x = float(row["x"])
        point_y = float(row["y"])
        label = display_reference_label(row["usr_label__primary"])
        display_x, display_y = ax.transData.transform((point_x, point_y))
        target_offset = offsets[0]
        for offset_x, offset_y in offsets:
            candidate_x = display_x + offset_x
            candidate_y = display_y + offset_y
            candidate_x = min(candidate_x, axes_box.x1 - right_padding_px)
            candidate_x = max(candidate_x, axes_box.x0 + left_padding_px)
            if all(
                abs(candidate_x - placed_x) > 52.0 or abs(candidate_y - placed_y) > 22.0
                for placed_x, placed_y in placed
            ):
                target_offset = (candidate_x - display_x, offset_y)
                break
        placed.append((display_x + target_offset[0], display_y + target_offset[1]))
        annotation = ax.annotate(
            label,
            xy=(point_x, point_y),
            xytext=target_offset,
            textcoords="offset points",
            fontsize=9.5,
            fontweight="semibold",
            color=TEXT_COLOR,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.94},
            arrowprops={"arrowstyle": "-", "color": SPINE_COLOR, "linewidth": 0.9},
            zorder=6,
        )
        annotation.set_clip_on(True)
        if annotation.arrow_patch is not None:
            annotation.arrow_patch.set_clip_on(True)
