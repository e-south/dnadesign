"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/visual_panel_baserender.py

BaseRender-backed panels for OPAL notebook visual review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from operator import index as integer_index
from typing import Any, Mapping

from ._support import display_name
from .baserender_record_sources import compact_record_id


def build_notebook_baserender_panel_title(selection_record: Mapping[str, Any]) -> str:
    """Build a compact in-canvas title from one authoritative selection record."""

    view_id = str(selection_record.get("selection_view_id") or "").strip()
    if not view_id:
        raise ValueError("BaseRender panel title requires a non-empty selection_view_id.")
    record_id = str(selection_record.get("record_id") or "").strip()
    if not record_id:
        raise ValueError("BaseRender panel title requires a non-empty record_id.")
    rank_value = selection_record.get("view_rank")
    if isinstance(rank_value, bool):
        raise ValueError("BaseRender panel title requires view_rank to be a positive integer.")
    try:
        rank = integer_index(rank_value)
    except TypeError as exc:
        raise ValueError("BaseRender panel title requires view_rank to be a positive integer.") from exc
    if rank <= 0:
        raise ValueError("BaseRender panel title requires view_rank to be a positive integer.")
    view_label = "AND" if view_id.lower() == "and" else display_name(view_id)
    return f"{view_label} selection · competition rank {rank} · candidate {compact_record_id(record_id)}"


def render_notebook_baserender_panel(
    *,
    baserender_campaign_model: Mapping[str, Any] | None,
    baserender_record_id: Any,
    baserender_record_row: Mapping[str, Any] | None,
    baserender_selection_record: Mapping[str, Any] | None,
    build_notebook_baserender_contract_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None,
    build_notebook_baserender_label_rows: Callable[..., list[dict[str, Any]]] | None,
    controls: Any | None,
    mo: Any,
    opal_table: Callable[..., Any],
    pl: Any,
    render_notebook_baserender_record: Callable[..., Mapping[str, Any]] | None,
    selected_baserender_round: int | None,
    selected_baserender_status_rows: Iterable[Mapping[str, Any]],
    selected_campaign_baserender_contract: Mapping[str, Any] | None,
    selected_campaign_labels_df: Any,
) -> Any:
    """Render one selected sequence with its binding and label evidence."""

    contract = selected_campaign_baserender_contract or {}
    label_rows = _require_callable(
        build_notebook_baserender_label_rows,
        "build_notebook_baserender_label_rows",
    )(
        selected_campaign_labels_df,
        record_id="" if baserender_record_id is None else str(baserender_record_id),
        round_value=selected_baserender_round,
    )
    label_panel = (
        opal_table(pl.DataFrame(label_rows), page_size=5)
        if label_rows
        else mo.md("No observed label is available for this selected sequence and round.")
    )
    if baserender_record_row is None:
        visual = mo.md("No contract-valid selected sequence is available for this round/run.")
    else:
        visual = _render_selected_sequence(
            baserender_campaign_model=baserender_campaign_model,
            baserender_record_row=baserender_record_row,
            baserender_selection_record=baserender_selection_record,
            contract=contract,
            mo=mo,
            render_notebook_baserender_record=render_notebook_baserender_record,
            selected_baserender_round=selected_baserender_round,
        )
    selection_detail_rows = list(selected_baserender_status_rows)
    if baserender_selection_record:
        selection_detail_rows.extend(
            [
                {"field": "selected record", "value": str(baserender_record_id)},
                {"field": "competition rank", "value": int(baserender_selection_record["view_rank"])},
            ]
        )
    detail_items = [
        opal_table(pl.DataFrame(selection_detail_rows), page_size=8),
        label_panel,
        opal_table(
            pl.DataFrame(
                _require_callable(
                    build_notebook_baserender_contract_rows,
                    "build_notebook_baserender_contract_rows",
                )(contract)
            ),
            page_size=8,
        ),
    ]
    return _render_panel_stack(
        mo=mo,
        items=[
            controls,
            visual,
            mo.accordion({"Sequence and selection evidence": mo.vstack(detail_items, gap=0.35)}, multiple=True),
        ],
    )


def render_notebook_three_axis_sequence_companion(
    *,
    baserender_record_id: Any,
    baserender_record_row: Mapping[str, Any] | None,
    baserender_record_selector: Any,
    baserender_selection_record: Mapping[str, Any] | None,
    contract: Mapping[str, Any] | None,
    mo: Any,
    render_notebook_baserender_record: Callable[..., Mapping[str, Any]] | None,
) -> Any:
    """Render exact selected-candidate sequence evidence below the exploratory 3D view."""

    if baserender_record_selector is None:
        return mo.callout(
            mo.md("Selected-candidate sequence inspection is unavailable for this campaign run."),
            kind="neutral",
        )
    guidance = mo.md(
        "**Selected candidate sequence.** Choose an allocated prediction to inspect its campaign-declared "
        "sequence annotation. Observed controls remain hover evidence because their sequence adapters are "
        "study-issued rather than campaign-global."
    )
    if baserender_record_row is None:
        rendered = mo.callout(
            mo.md("The selected prediction has no record valid under this campaign's BaseRender contract."),
            kind="warn",
        )
    else:
        selection_record = dict(baserender_selection_record or {})
        title = build_notebook_baserender_panel_title(selection_record)
        expected_record_id = _validate_selected_record_identity(
            selection_record=selection_record,
            record_row=baserender_record_row,
            selected_record_id=baserender_record_id,
        )
        payload = _require_callable(
            render_notebook_baserender_record,
            "render_notebook_baserender_record",
        )(baserender_record_row, dict(contract or {}), title=title)
        _validate_rendered_record_identity(payload=payload, expected_record_id=expected_record_id)
        rendered = _render_sequence_image(
            payload=payload,
            mo=mo,
        )
    return mo.vstack(
        [
            mo.hstack([baserender_record_selector], justify="start", align="end", wrap=True, gap=0.35),
            guidance,
            rendered,
        ],
        gap=0.25,
    )


def _render_selected_sequence(
    *,
    baserender_campaign_model: Mapping[str, Any] | None,
    baserender_record_row: Mapping[str, Any],
    baserender_selection_record: Mapping[str, Any] | None,
    contract: Mapping[str, Any],
    mo: Any,
    render_notebook_baserender_record: Callable[..., Mapping[str, Any]] | None,
    selected_baserender_round: int | None,
) -> Any:
    selection_record = dict(baserender_selection_record or {})
    title = build_notebook_baserender_panel_title(selection_record)
    expected_record_id = _validate_selected_record_identity(
        selection_record=selection_record,
        record_row=baserender_record_row,
    )
    payload = _require_callable(
        render_notebook_baserender_record,
        "render_notebook_baserender_record",
    )(baserender_record_row, contract, title=title)
    _validate_rendered_record_identity(payload=payload, expected_record_id=expected_record_id)
    slug = str((baserender_campaign_model or {}).get("campaign", {}).get("slug") or "unknown campaign")
    round_text = f"round {selected_baserender_round}" if selected_baserender_round is not None else "unknown round"
    return _render_sequence_image(
        payload=payload,
        mo=mo,
        alt_suffix=f" Selected in campaign {slug}, {round_text}.",
    )


def _validate_selected_record_identity(
    *,
    selection_record: Mapping[str, Any],
    record_row: Mapping[str, Any],
    selected_record_id: Any = None,
) -> str:
    expected_record_id = str(selection_record.get("record_id") or "").strip()
    row_record_id = str(record_row.get("id") or "").strip()
    if row_record_id != expected_record_id:
        raise ValueError(
            "Selected sequence render record does not match its authoritative selection record: "
            f"expected `{expected_record_id}`, received `{row_record_id or 'missing'}`."
        )
    control_record_id = str(selected_record_id or "").strip()
    if control_record_id and control_record_id != expected_record_id:
        raise ValueError(
            "Selected sequence control does not match its authoritative selection record: "
            f"expected `{expected_record_id}`, received `{control_record_id}`."
        )
    return expected_record_id


def _validate_rendered_record_identity(*, payload: Mapping[str, Any], expected_record_id: str) -> None:
    payload_record_id = str(payload.get("record_id") or "").strip()
    if payload_record_id != expected_record_id:
        raise ValueError(
            "BaseRender payload does not match its authoritative selection record: "
            f"expected `{expected_record_id}`, received `{payload_record_id or 'missing'}`."
        )


def _render_sequence_image(
    *,
    payload: Mapping[str, Any],
    mo: Any,
    alt_suffix: str = "",
) -> Any:
    return mo.image(
        payload["image_bytes"],
        alt=f"{payload['alt_text']}{alt_suffix}",
        caption=str(payload["caption"]),
        rounded=True,
        style={
            "width": "100%",
            "max-width": "100%",
            "height": "auto",
            "object-fit": "contain",
            "margin": "0",
            "display": "block",
            "background-color": "#FFFFFF",
        },
    )


def _render_panel_stack(*, mo: Any, items: Iterable[Any | None]) -> Any:
    return mo.vstack([item for item in items if item is not None], gap=0.45)


def _require_callable(value: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
    if value is None:
        raise ValueError(f"{name} is required for this OPAL notebook visual surface.")
    return value


__all__ = [
    "build_notebook_baserender_panel_title",
    "render_notebook_baserender_panel",
    "render_notebook_three_axis_sequence_companion",
]
