"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/three_axis_scatter.py

Interactive three-axis inspection for manifest-backed layered scatters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from ...plots._mpl_utils import compact_batch_label
from . import three_axis_scatter_style as style
from .selection_round_encoding import selection_round_palette_index
from .three_axis_camera_state import render_three_axis_camera_state
from .three_axis_scatter_data import (
    THREE_AXIS_SCATTER_ADAPTER,
    require_finite_three_axis_rows,
    resolve_three_axis_interactive_contract,
    sample_notebook_three_axis_rows,
)

THREE_AXIS_PUBLICATION_MODE = "publication_2d"
THREE_AXIS_INTERACTIVE_MODE = "interactive_3d"
THREE_AXIS_CAMERA_REVISION = f"{THREE_AXIS_SCATTER_ADAPTER}:camera"


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
    batch_column = str(view["batch_column"])
    prediction_value = str(view["prediction_value"])
    observed_value = str(view["observed_value"])
    kinds = displayed[record_column].astype(str)
    selection_round_column = "__notebook_selection_round"
    if selection_round_column not in displayed:
        raise ValueError("Three-axis scatter rows are missing categorical selection-round provenance.")
    selection_rounds = displayed[selection_round_column]
    show_selected = bool(displayed.attrs.get("show_selected", True))
    predictions = displayed.loc[kinds.eq(prediction_value) & selection_rounds.isna()]
    selected = (
        displayed.loc[kinds.eq(prediction_value) & selection_rounds.notna()] if show_selected else displayed.iloc[0:0]
    )
    observed = displayed.loc[kinds.eq(observed_value)]
    round_palette_index = selection_round_palette_index(contract)

    traces: list[Any] = []
    if not predictions.empty:
        complete_background = int(displayed.attrs.get("complete_background_count", len(predictions)))
        traces.append(
            _trace(
                go,
                predictions,
                contract=contract,
                name=f"Prediction sample ({len(predictions):,} / {complete_background:,})",
                marker={"size": 2.4, "color": "#2563EB", "opacity": 0.20},
                showlegend=True,
            )
        )
    for round_k in sorted(selected[selection_round_column].astype(int).unique()):
        if round_k not in round_palette_index:
            raise ValueError(f"Three-axis scatter selection round {round_k} is absent from the contract.")
        index = round_palette_index[round_k]
        round_selected = selected.loc[selected[selection_round_column].astype(int).eq(round_k)]
        categorical_marker = style.selection_marker(index)
        traces.append(
            _trace(
                go,
                round_selected,
                contract=contract,
                name=f"Selected for Round {round_k} (n={len(round_selected):,})",
                marker={
                    "size": 7.2,
                    "color": categorical_marker["color"],
                    "opacity": 1.0,
                    "symbol": categorical_marker["symbol"],
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
        categorical_marker = style.observed_marker(index)
        traces.append(
            _trace(
                go,
                batch,
                contract=contract,
                name=f"Observed · {compact_batch_label(batch_id)} (n={len(batch):,})",
                marker={
                    "size": 5.8,
                    "color": categorical_marker["color"],
                    "opacity": 0.95,
                    "symbol": categorical_marker["symbol"],
                    "line": {"color": "#111827", "width": 1.0},
                },
                showlegend=True,
            )
        )
    if not traces:
        raise ValueError("Three-axis scatter requires at least one visible row.")

    title = _title(runtime)
    subtitle = _subtitle(runtime)
    return style.apply_three_axis_layout(
        go.Figure(data=traces),
        title=title,
        subtitle=subtitle,
        xaxis_title=_plotly_axis_label(runtime["x_label"]),
        yaxis_title=_plotly_axis_label(runtime["y_label"]),
        zaxis_title=_plotly_axis_label(runtime["color_label"]),
        camera_revision=THREE_AXIS_CAMERA_REVISION,
        complete_row_count=len(rows),
        displayed_row_count=len(displayed),
        background_sample_limit=int(interactive["prediction_sample_limit"]),
    )


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
    plot_frame = widget.style({"width": "min(100%, 900px)", "margin": "0 auto"})
    camera_state = render_three_axis_camera_state(
        mo=mo,
        revision=THREE_AXIS_CAMERA_REVISION,
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
    return mo.vstack([camera_state, plot_frame, caption], gap=0.2)


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
    x_label = _plotly_label(_mapping(contract["runtime"])["x_label"])
    y_label = _plotly_label(_mapping(contract["runtime"])["y_label"])
    z_label = _plotly_label(_mapping(contract["runtime"])["color_label"])
    score_label = _plotly_label(interactive["score_label"])
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
    return _plotly_label(title)


def _subtitle(runtime: Mapping[str, Any]) -> str:
    return _plotly_label(runtime.get("context") or "")


def _plotly_label(value: object) -> str:
    """Translate small inline-math labels to Plotly's WebGL-safe HTML subset."""

    text = str(value or "").strip()
    pieces = re.split(r"(\$[^$]*\$)", text)
    rendered = [
        _plotly_math(piece[1:-1]) if piece.startswith("$") and piece.endswith("$") else html.escape(piece)
        for piece in pieces
        if piece
    ]
    return " ".join("".join(rendered).split())


def _plotly_axis_label(value: object) -> str:
    """Wrap a terminal unit phrase so long three-dimensional axes do not collide."""

    label = _plotly_label(value)
    return re.sub(r"\s+(\([^()]+ units\))$", r"<br>\1", label)


def _plotly_math(value: str) -> str:
    normalized = re.sub(r"\\(?:mathrm|text)\{([^{}]*)\}", r"\1", value.strip())
    symbol = re.fullmatch(
        r"([A-Za-z])(?:_\{([^{}]+)\}|_([A-Za-z0-9]+))?(?:\^\{([^{}]+)\}|\^([A-Za-z0-9]+))?",
        normalized,
    )
    if symbol:
        base, subscript_braced, subscript_plain, superscript_braced, superscript_plain = symbol.groups()
        rendered = f"<i>{html.escape(base)}</i>"
        if subscript := subscript_braced or subscript_plain:
            rendered += f"<sub>{html.escape(subscript)}</sub>"
        if superscript := superscript_braced or superscript_plain:
            rendered += f"<sup>{html.escape(superscript)}</sup>"
        return rendered

    plain = normalized.replace("{", "").replace("}", "")
    plain = re.sub(r"\\([A-Za-z]+)", r"\1", plain)
    return html.escape(plain)


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
