"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/three_axis_scatter.py

Interactive three-axis inspection for manifest-backed layered scatters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from ...plots._mpl_utils import compact_batch_label
from .three_axis_scatter_data import (
    THREE_AXIS_SCATTER_ADAPTER,
    require_finite_three_axis_rows,
    resolve_three_axis_interactive_contract,
    sample_notebook_three_axis_rows,
)

THREE_AXIS_PUBLICATION_MODE = "publication_2d"
THREE_AXIS_INTERACTIVE_MODE = "interactive_3d"

_OBSERVED_COLORS = (
    "#7C3AED",
    "#059669",
    "#DC2626",
    "#0891B2",
    "#A16207",
    "#DB2777",
    "#4F46E5",
    "#0F766E",
)


def build_notebook_three_axis_scatter_figure(
    rows: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
) -> Any:
    """Build an interactive Plotly inspection view of three declared coordinates."""

    interactive = resolve_three_axis_interactive_contract(contract)
    view = _mapping(contract.get("view"))
    runtime = _mapping(contract.get("runtime"))
    score_column = str(interactive["score_column"])
    columns = {
        "id",
        str(view["record_kind_column"]),
        str(view["selection_column"]),
        str(view["batch_column"]),
        str(view["label_column"]),
        str(view["x_column"]),
        str(view["y_column"]),
        str(view["color_column"]),
        score_column,
    }
    missing = sorted(columns - set(rows.columns))
    if missing:
        raise ValueError(f"Three-axis scatter rows are missing columns: {missing}.")
    require_finite_three_axis_rows(
        rows,
        columns=(
            str(view["x_column"]),
            str(view["y_column"]),
            str(view["color_column"]),
            score_column,
        ),
    )
    displayed = sample_notebook_three_axis_rows(rows, contract=contract)

    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError("Interactive three-axis plots require the project plotly dependency.") from exc

    record_column = str(view["record_kind_column"])
    selection_column = str(view["selection_column"])
    batch_column = str(view["batch_column"])
    prediction_value = str(view["prediction_value"])
    observed_value = str(view["observed_value"])
    kinds = displayed[record_column].astype(str)
    predictions = displayed.loc[kinds.eq(prediction_value) & ~displayed[selection_column].fillna(False).astype(bool)]
    selected = displayed.loc[kinds.eq(prediction_value) & displayed[selection_column].fillna(False).astype(bool)]
    observed = displayed.loc[kinds.eq(observed_value)]

    traces: list[Any] = []
    if not predictions.empty:
        complete_background = int(displayed.attrs.get("complete_background_count", len(predictions)))
        traces.append(
            _trace(
                go,
                predictions,
                contract=contract,
                name=(f"Deterministic prediction sample (n={len(predictions):,} of {complete_background:,})"),
                marker={"size": 2.4, "color": "#2563EB", "opacity": 0.20},
                showlegend=True,
            )
        )
    if not selected.empty:
        traces.append(
            _trace(
                go,
                selected,
                contract=contract,
                name=f"Selected (n={len(selected):,})",
                marker={
                    "size": 7.2,
                    "color": "#F59E0B",
                    "opacity": 1.0,
                    "symbol": "diamond",
                    "line": {"color": "#111827", "width": 1.5},
                },
                showlegend=True,
            )
        )
    batch_labels = {str(item["id"]): str(item["label"]) for item in contract.get("observed_batches") or []}
    for index, batch_id in enumerate(batch_labels):
        batch = observed.loc[observed[batch_column].astype(str).eq(batch_id)]
        if batch.empty:
            continue
        traces.append(
            _trace(
                go,
                batch,
                contract=contract,
                name=f"Observed · {compact_batch_label(batch_id)} (n={len(batch):,})",
                marker={
                    "size": 5.8,
                    "color": _OBSERVED_COLORS[index % len(_OBSERVED_COLORS)],
                    "opacity": 0.95,
                    "symbol": "circle",
                    "line": {"color": "#111827", "width": 1.0},
                },
                showlegend=True,
            )
        )
    if not traces:
        raise ValueError("Three-axis scatter requires at least one visible row.")

    axis_style = {
        "showbackground": True,
        "backgroundcolor": "#FAFAFA",
        "gridcolor": "#D1D5DB",
        "gridwidth": 1.0,
        "zeroline": True,
        "zerolinecolor": "#6B7280",
        "zerolinewidth": 1.5,
        "showspikes": False,
        "tickfont": {"size": 13, "color": "#252525"},
        "title": {"font": {"size": 16, "color": "#111827"}},
    }
    title = _title(runtime)
    figure = go.Figure(data=traces)
    figure.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.96,
            "yanchor": "top",
            "font": {"size": 21, "color": "#111827"},
        },
        scene={
            "xaxis": {**axis_style, "title": {"text": str(runtime["x_label"])}},
            "yaxis": {**axis_style, "title": {"text": str(runtime["y_label"])}},
            "zaxis": {**axis_style, "title": {"text": str(runtime["color_label"])}},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.2}},
            "bgcolor": "white",
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.08,
            "yanchor": "top",
            "font": {"size": 13},
            "bgcolor": "rgba(255,255,255,0.88)",
        },
        font={"family": "Arial, Helvetica, sans-serif", "size": 14, "color": "#252525"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=760,
        margin={"l": 12, "r": 12, "t": 96, "b": 82},
        hovermode="closest",
        uirevision=str(contract.get("key") or THREE_AXIS_SCATTER_ADAPTER),
        meta={
            "complete_row_count": int(len(rows)),
            "displayed_row_count": int(len(displayed)),
            "background_sample_limit": int(interactive["prediction_sample_limit"]),
        },
    )
    return figure


def render_notebook_three_axis_scatter(
    rows: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
    mo: Any,
) -> Any:
    """Wrap the three-axis figure and its sampling disclosure in Marimo UI."""

    figure = build_notebook_three_axis_scatter_figure(rows, contract=contract)
    widget = mo.ui.plotly(
        figure,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
        label="Interactive three-family candidate landscape",
    )
    meta = dict(figure.layout.meta or {})
    displayed = int(meta.get("displayed_row_count") or 0)
    complete = int(meta.get("complete_row_count") or 0)
    background_limit = int(meta.get("background_sample_limit") or 0)
    caption = mo.md(
        "**Interactive inspection.** The rotatable view shows a deterministic SHA-256-ID "
        f"sample capped at {background_limit:,} background predictions and retains every selected "
        f"and observed record ({displayed:,} of {complete:,} ledger rows displayed). Hover for exact "
        "candidate identity and family scores. The 2D figure remains the complete publication artifact; "
        "use the selected-candidate control for sequence inspection."
    )
    return mo.vstack([widget, caption], gap=0.2)


def _trace(
    go: Any,
    rows: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
    name: str,
    marker: Mapping[str, Any],
    showlegend: bool,
) -> Any:
    view = _mapping(contract["view"])
    interactive = resolve_three_axis_interactive_contract(contract)
    label_column = str(view["label_column"])
    score_column = str(interactive["score_column"])
    x_column = str(view["x_column"])
    y_column = str(view["y_column"])
    z_column = str(view["color_column"])
    labels = rows[label_column].where(rows[label_column].notna(), rows["id"]).astype(str)
    customdata = [
        [str(candidate_id), str(label), float(score)]
        for candidate_id, label, score in zip(
            rows["id"],
            labels,
            rows[score_column],
            strict=True,
        )
    ]
    x_label = _plain_label(_mapping(contract["runtime"])["x_label"])
    y_label = _plain_label(_mapping(contract["runtime"])["y_label"])
    z_label = _plain_label(_mapping(contract["runtime"])["color_label"])
    score_label = _plain_label(interactive["score_label"])
    return go.Scatter3d(
        x=rows[x_column],
        y=rows[y_column],
        z=rows[z_column],
        customdata=customdata,
        mode="markers",
        marker=dict(marker),
        name=name,
        showlegend=showlegend,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Candidate: %{customdata[0]}<br>"
            + x_label
            + ": %{x:.3f}<br>"
            + y_label
            + ": %{y:.3f}<br>"
            + z_label
            + ": %{z:.3f}<br>"
            + score_label
            + ": %{customdata[2]:.3f}<extra>"
            + name
            + "</extra>"
        ),
    )


def _title(runtime: Mapping[str, Any]) -> str:
    title = str(runtime.get("title") or "Three-family candidate landscape").strip()
    context = str(runtime.get("context") or "").strip()
    return f"{title}<br><sup>{context}</sup>" if context else title


def _plain_label(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"_\{([^{}]*)\}", r"_\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    return " ".join(text.split())


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "THREE_AXIS_INTERACTIVE_MODE",
    "THREE_AXIS_PUBLICATION_MODE",
    "THREE_AXIS_SCATTER_ADAPTER",
    "build_notebook_three_axis_scatter_figure",
    "render_notebook_three_axis_scatter",
    "sample_notebook_three_axis_rows",
]
