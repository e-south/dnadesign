"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/reader_evidence_triptych.py

Reader-backed SFXI triptych rendering for OPAL notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .reader_evidence_media import select_reader_media_artifact
from .reader_evidence_triptych_runtime import (
    ReaderSFXITriptychRuntime,
    finite_float,
    load_reader_sfxi_triptych_runtime,
    sfxi_time_step,
)

SFXI_TRIPTYCH_SEMANTIC_KIND = "intensity_overview"


def is_reader_sfxi_triptych_artifact(row: Mapping[str, Any]) -> bool:
    """Return true for Reader SFXI time-series plus snapshot artifacts."""

    return str(row.get("semantic_kind") or "").strip() == SFXI_TRIPTYCH_SEMANTIC_KIND


def render_notebook_reader_evidence_time_control(
    surface: Mapping[str, Any],
    *,
    selected_plot_type_label: str | None,
    selected_artifact_label: str | None,
    mo: Any,
) -> Any | None:
    """Render a time slider when the selected plot supports live Reader snapshots."""

    selected = _select_triptych_row(
        surface,
        selected_plot_type_label=selected_plot_type_label,
        selected_artifact_label=selected_artifact_label,
    )
    if selected is None:
        return None
    try:
        metadata = reader_sfxi_triptych_time_metadata(selected)
    except RuntimeError as exc:
        return mo.md(f"Time control unavailable: `{exc}`")
    return mo.ui.slider(
        start=metadata["start"],
        stop=metadata["stop"],
        value=metadata["value"],
        step=metadata["step"],
        debounce=True,
        show_value=True,
        label="Time (h)",
        full_width=True,
    )


def reader_sfxi_triptych_time_metadata(row: Mapping[str, Any]) -> dict[str, float]:
    """Return slider bounds and the Vec8 snapshot time for a Reader evidence row."""

    runtime = load_reader_sfxi_triptych_runtime(row)
    ground_truth_time = finite_float(row.get("time_selected_h"))
    default_time = ground_truth_time if ground_truth_time is not None else runtime.default_time_h
    if default_time < runtime.common_times[0] or default_time > runtime.common_times[-1]:
        default_time = runtime.default_time_h
    return {
        "start": runtime.common_times[0],
        "stop": runtime.common_times[-1],
        "step": sfxi_time_step(runtime.common_times),
        "value": default_time,
        "ground_truth_time_h": ground_truth_time if ground_truth_time is not None else runtime.default_time_h,
    }


def render_reader_sfxi_triptych_visual(row: Mapping[str, Any], *, selected_time_h: float, mo: Any) -> Any:
    """Render a Reader dual-reporter triptych for one selected evidence row."""

    runtime = load_reader_sfxi_triptych_runtime(row)
    chart_builders = _reader_triptych_builders()
    try:
        import altair as alt  # noqa: PLC0415

        alt.data_transformers.disable_max_rows()
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(f"Altair is required for Reader triptych rendering: {exc}") from exc
    snapshot_time = chart_builders["choose_time"](runtime.common_times, float(selected_time_h), "nearest")
    if snapshot_time is None:
        raise RuntimeError(f"No common SFXI time is available near {float(selected_time_h):.3f} h.")
    triptych_data = chart_builders["build_triptych_data"](
        _triptych_rows_for_design(runtime, design_id=str(row.get("design_id") or "").strip()),
        time_col=runtime.time_col,
        treatment_col="sfxi_condition",
        growth_channel="OD600",
        ratio_channel=runtime.logic_channel,
        snapshot_channel=runtime.logic_channel,
        snapshot_time=float(snapshot_time),
        treatment_order=runtime.sfxi_condition_order,
    )
    chart = chart_builders["build_dual_reporter_triptych_chart"](
        alt=alt,
        pd_module=pd,
        data=triptych_data,
        time_col=runtime.time_col,
        treatment_col="sfxi_condition",
        induction_time_h=runtime.induction_time_h,
        width=206,
        height=206,
        spacing=10,
    )
    header = _triptych_header(row, target_time=float(selected_time_h), snapshot_time=float(snapshot_time), mo=mo)
    chart_view = mo.ui.altair_chart(chart, chart_selection=False, legend_selection=False)
    return mo.vstack([header, chart_view], gap=0.35).style(
        {"min-height": "500px", "width": "100%", "max-width": "100%"}
    )


def _select_triptych_row(
    surface: Mapping[str, Any],
    *,
    selected_plot_type_label: str | None,
    selected_artifact_label: str | None,
) -> Mapping[str, Any] | None:
    if not selected_plot_type_label or not selected_artifact_label:
        return None
    selected = select_reader_media_artifact(
        surface,
        selected_plot_type_label=selected_plot_type_label,
        selected_artifact_label=selected_artifact_label,
    )
    if selected is None or not is_reader_sfxi_triptych_artifact(selected):
        return None
    return selected


def _reader_triptych_builders() -> dict[str, Any]:
    from reader.workbench.notebooks.dual_reporter_triptych import (  # noqa: PLC0415
        build_dual_reporter_triptych_chart,
        build_triptych_data,
        choose_time,
    )

    return {
        "build_dual_reporter_triptych_chart": build_dual_reporter_triptych_chart,
        "build_triptych_data": build_triptych_data,
        "choose_time": choose_time,
    }


def _triptych_rows_for_design(runtime: ReaderSFXITriptychRuntime, *, design_id: str) -> pd.DataFrame:
    if not design_id:
        raise RuntimeError("Reader triptych row has no design_id.")
    subset = runtime.tidy[runtime.tidy[runtime.label_col].astype(str) == design_id].copy()
    if subset.empty:
        raise RuntimeError(f"Reader tidy record has no rows for design_id `{design_id}`.")
    raw_treatment = subset[runtime.treatment_col].astype(str)
    key = raw_treatment if runtime.treatment_case_sensitive else raw_treatment.str.strip().str.casefold()
    subset["sfxi_condition"] = key.map(runtime.sfxi_condition_map)
    subset = subset[subset["sfxi_condition"].isin(runtime.sfxi_condition_order)].copy()
    if subset.empty:
        raise RuntimeError(f"Reader tidy record has no SFXI treatment-map rows for `{design_id}`.")
    return subset


def _triptych_header(row: Mapping[str, Any], *, target_time: float, snapshot_time: float, mo: Any) -> Any:
    truth = finite_float(row.get("time_selected_h"))
    truth_text = "not recorded" if truth is None else f"{truth:.2f} h"
    return mo.md(
        f"**Design:** `{row.get('design_id') or ''}`  \n"
        f"**Vec8 snapshot:** {truth_text}. **Displayed snapshot:** {snapshot_time:.2f} h "
        f"(slider target {target_time:.2f} h)."
    )


__all__ = [
    "SFXI_TRIPTYCH_SEMANTIC_KIND",
    "is_reader_sfxi_triptych_artifact",
    "reader_sfxi_triptych_time_metadata",
    "render_notebook_reader_evidence_time_control",
    "render_reader_sfxi_triptych_visual",
]
