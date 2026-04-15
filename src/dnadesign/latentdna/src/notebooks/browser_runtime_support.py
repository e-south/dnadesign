"""
Shared browser runtime support helpers for generated latentdna marimo notebooks.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..contracts.notebook import WorkspaceNotebookControls

REFERENCE_DISPLAY = {
    "spyp": "spyP",
    "sulap": "sulAp",
    "soxsp": "soxSp",
    "j23105": "J23105",
}

CONTROL_PLANE_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


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


def fig_to_image(fig, *, dpi: int = 150):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return mo.image(buf.getvalue(), alt="geometry switchboard plot", width="100%")


def render_plot_asset(path: Path, *, workspace_dir: Path):
    suffix = path.suffix.lower()
    if suffix == ".svg":
        return mo.Html(
            f"<div style='width: 100%; overflow-x: auto; padding: 0.5rem 0;'>{path.read_text(encoding='utf-8')}</div>"
        )
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return mo.image(path.read_bytes(), alt=path.name, width="100%")
    return mo.md(f"`{path.relative_to(workspace_dir).as_posix()}`")


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
    if column.startswith("infer__evo2_") and "__log_likelihood__mean_per_token" in column:
        model = "7B" if "__7b__" in column else "20B"
        scope = "1 kb" if "__template_1kb_" in column else "60 bp"
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
    ordered = [
        column for column in preferred if column in frame.columns and include_hue_column(column, artifact_suffixes)
    ]
    for column in frame.columns:
        if column in ordered or not include_hue_column(column, artifact_suffixes):
            continue
        series = frame[column]
        sample_value = next(
            (value for value in series if value is not None and not (isinstance(value, float) and pd.isna(value))),
            None,
        )
        if isinstance(sample_value, (list, tuple, dict, set, np.ndarray)):
            continue
        try:
            unique_count = int(series.nunique(dropna=False))
        except TypeError:
            continue
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series) or unique_count <= 16:
            ordered.append(column)
    return ordered


def scatter_style(row_count: int) -> tuple[float, float]:
    if row_count <= 250:
        return 38.0, 0.92
    if row_count <= 1_000:
        return 24.0, 0.82
    if row_count <= 5_000:
        return 13.0, 0.74
    return 8.0, 0.58


def draw_reference_labels(ax, frame: pd.DataFrame, *, reference_labels: list[str]) -> None:
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
    x_span = max(float(selected["x"].max() - selected["x"].min()), 1.0)
    y_span = max(float(selected["y"].max() - selected["y"].min()), 1.0)
    x_pad = x_span * 0.06
    y_pad = y_span * 0.06
    offsets = [
        (x_pad, y_pad),
        (x_pad, -y_pad),
        (-4.5 * x_pad, y_pad),
        (-4.5 * x_pad, -y_pad),
        (1.8 * x_pad, 2.2 * y_pad),
        (-5.6 * x_pad, 2.2 * y_pad),
    ]
    placed: list[tuple[float, float]] = []
    ax.scatter(
        selected["x"].to_numpy(dtype=float),
        selected["y"].to_numpy(dtype=float),
        c="#111111",
        s=135,
        marker="*",
        linewidths=0.8,
        edgecolors="white",
        zorder=5,
    )
    for row in selected.sort_values("usr_label__primary").to_dict(orient="records"):
        point_x = float(row["x"])
        point_y = float(row["y"])
        label = display_reference_label(row["usr_label__primary"])
        target_x = point_x + offsets[0][0]
        target_y = point_y + offsets[0][1]
        for offset_x, offset_y in offsets:
            candidate_x = point_x + offset_x
            candidate_y = point_y + offset_y
            if all(
                abs(candidate_x - placed_x) > (0.9 * x_pad) or abs(candidate_y - placed_y) > (0.9 * y_pad)
                for placed_x, placed_y in placed
            ):
                target_x = candidate_x
                target_y = candidate_y
                break
        placed.append((target_x, target_y))
        ax.annotate(
            label,
            xy=(point_x, point_y),
            xytext=(target_x, target_y),
            textcoords="data",
            fontsize=9,
            fontweight="semibold",
            color="#16202A",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.86},
            arrowprops={"arrowstyle": "-", "color": "#5C6874", "linewidth": 0.9},
            zorder=6,
        )
