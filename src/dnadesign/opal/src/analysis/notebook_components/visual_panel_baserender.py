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
from html import escape
from typing import Any, Mapping

from ._support import display_name
from .baserender_record_sources import compact_record_id


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
        payload = _require_callable(
            render_notebook_baserender_record,
            "render_notebook_baserender_record",
        )(baserender_record_row, dict(contract or {}))
        compact_id = compact_record_id(str(baserender_record_id or payload["record_id"]))
        rendered = mo.vstack(
            [
                mo.md(
                    '<h4 style="text-align:center; margin:0 0 0.1rem 0;">'
                    f"Candidate <code>{escape(compact_id)}</code></h4>"
                ),
                _render_sequence_image(payload=payload, mo=mo),
            ],
            gap=0.1,
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
    view_id = str(selection_record.get("selection_view_id") or "").strip()
    rank_value = selection_record.get("view_rank")
    if not view_id or rank_value is None:
        raise ValueError("Selected sequence render requires its selection view and competition rank.")
    rank = int(rank_value)
    if rank <= 0:
        raise ValueError(f"Selected sequence render has invalid competition rank {rank}.")
    payload = _require_callable(
        render_notebook_baserender_record,
        "render_notebook_baserender_record",
    )(baserender_record_row, contract)
    slug = str((baserender_campaign_model or {}).get("campaign", {}).get("slug") or "unknown campaign")
    round_text = f"round {selected_baserender_round}" if selected_baserender_round is not None else "unknown round"
    view_label = "AND" if view_id.lower() == "and" else display_name(view_id)
    compact_id = compact_record_id(str(payload["record_id"]))
    heading = mo.md(
        '<h4 style="text-align:center; margin:0 0 0.15rem 0;">'
        f"{escape(view_label)} selection · competition rank {rank} · "
        f"candidate <code>{escape(compact_id)}</code></h4>"
    )
    image = _render_sequence_image(
        payload=payload,
        mo=mo,
        alt_suffix=f" Selected in campaign {slug}, {round_text}.",
    )
    return mo.vstack([heading, image], gap=0.1)


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
    "render_notebook_baserender_panel",
    "render_notebook_three_axis_sequence_companion",
]
