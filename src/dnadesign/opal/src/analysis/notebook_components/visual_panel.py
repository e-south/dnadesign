"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/visual_panel.py

Notebook component builders for selected OPAL visual panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Mapping


def render_notebook_visual_panel(
    *,
    active_view_mode: str,
    build_notebook_plot_card_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
    build_notebook_plot_method_sections: Callable[[Mapping[str, Any]], Mapping[str, str]],
    mo: Any,
    opal_table: Callable[..., Any],
    pl: Any,
    render_notebook_plot_choice_image: Callable[..., Any],
    selected_visual_choice: Mapping[str, Any] | None,
    select_notebook_plot_scope: Callable[..., Mapping[str, Any]],
    plot_ui: Any = None,
    plot_scope_ui: Any = None,
    build_notebook_collection_visual_card_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None = None,
    collection_visual_description: Callable[[Mapping[str, Any]], str] | None = None,
    campaign_set_baserender_surface_kind: str = "campaign_set_baserender",
    baserender_campaign_model: Mapping[str, Any] | None = None,
    baserender_record_id: Any = None,
    baserender_record_row: Mapping[str, Any] | None = None,
    baserender_record_selector: Any = None,
    baserender_role_ui: Any = None,
    baserender_round_ui: Any = None,
    baserender_run_ui: Any = None,
    build_notebook_baserender_contract_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None = None,
    build_notebook_baserender_label_rows: Callable[..., list[dict[str, Any]]] | None = None,
    render_notebook_baserender_record: Callable[..., Mapping[str, Any]] | None = None,
    selected_baserender_round: int | None = None,
    selected_baserender_status_rows: Iterable[Mapping[str, Any]] = (),
    selected_campaign_baserender_contract: Mapping[str, Any] | None = None,
    selected_campaign_labels_df: Any = None,
) -> Any:
    """Render the active OPAL visual surface with hierarchical notebook controls."""

    is_baserender = _is_baserender_visual(
        selected_visual_choice,
        campaign_set_baserender_surface_kind=campaign_set_baserender_surface_kind,
    )
    controls = _render_visual_controls(
        mo=mo,
        plot_ui=plot_ui,
        plot_scope_ui=plot_scope_ui,
        is_baserender=is_baserender,
        baserender_role_ui=baserender_role_ui,
        baserender_round_ui=baserender_round_ui,
        baserender_run_ui=baserender_run_ui,
        baserender_record_selector=baserender_record_selector,
    )
    if selected_visual_choice is None:
        return _render_empty_panel(active_view_mode=active_view_mode, controls=controls, mo=mo)
    if is_baserender:
        return _render_baserender_panel(
            baserender_campaign_model=baserender_campaign_model,
            baserender_record_id=baserender_record_id,
            baserender_record_row=baserender_record_row,
            build_notebook_baserender_contract_rows=build_notebook_baserender_contract_rows,
            build_notebook_baserender_label_rows=build_notebook_baserender_label_rows,
            controls=controls,
            mo=mo,
            opal_table=opal_table,
            pl=pl,
            render_notebook_baserender_record=render_notebook_baserender_record,
            selected_baserender_round=selected_baserender_round,
            selected_baserender_status_rows=selected_baserender_status_rows,
            selected_campaign_baserender_contract=selected_campaign_baserender_contract,
            selected_campaign_labels_df=selected_campaign_labels_df,
        )
    if active_view_mode == "Campaign set":
        return _render_collection_plot_panel(
            build_notebook_collection_visual_card_rows=build_notebook_collection_visual_card_rows,
            collection_visual_description=collection_visual_description,
            controls=controls,
            mo=mo,
            opal_table=opal_table,
            pl=pl,
            render_notebook_plot_choice_image=render_notebook_plot_choice_image,
            selected_visual_choice=selected_visual_choice,
        )
    return _render_campaign_plot_panel(
        build_notebook_plot_card_rows=build_notebook_plot_card_rows,
        build_notebook_plot_method_sections=build_notebook_plot_method_sections,
        controls=controls,
        mo=mo,
        opal_table=opal_table,
        pl=pl,
        plot_scope_ui=plot_scope_ui,
        render_notebook_plot_choice_image=render_notebook_plot_choice_image,
        selected_visual_choice=selected_visual_choice,
        select_notebook_plot_scope=select_notebook_plot_scope,
    )


def _is_baserender_visual(
    selected_visual_choice: Mapping[str, Any] | None,
    *,
    campaign_set_baserender_surface_kind: str,
) -> bool:
    if selected_visual_choice is None:
        return False
    return selected_visual_choice.get("surface_kind") in {"baserender", campaign_set_baserender_surface_kind}


def _render_visual_controls(
    *,
    mo: Any,
    plot_ui: Any,
    plot_scope_ui: Any,
    is_baserender: bool,
    baserender_role_ui: Any,
    baserender_round_ui: Any,
    baserender_run_ui: Any,
    baserender_record_selector: Any,
) -> Any:
    controls = [control for control in [plot_ui] if control is not None]
    if is_baserender:
        controls.extend(
            control
            for control in (
                baserender_role_ui,
                baserender_round_ui,
                baserender_run_ui,
                baserender_record_selector,
            )
            if control is not None
        )
    elif plot_scope_ui is not None:
        controls.append(plot_scope_ui)
    return mo.hstack(controls, justify="start", align="end", wrap=True, gap=0.35) if controls else mo.md("")


def _render_empty_panel(*, active_view_mode: str, controls: Any, mo: Any) -> Any:
    if active_view_mode == "Campaign set":
        lines = ["No campaign-set comparison visuals are available."]
    else:
        lines = ["No OPAL plot deliverables are available for this campaign."]
    return mo.vstack([controls, mo.md("\n".join(lines))], gap=0.45)


def _render_baserender_panel(
    *,
    baserender_campaign_model: Mapping[str, Any] | None,
    baserender_record_id: Any,
    baserender_record_row: Mapping[str, Any] | None,
    build_notebook_baserender_contract_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None,
    build_notebook_baserender_label_rows: Callable[..., list[dict[str, Any]]] | None,
    controls: Any,
    mo: Any,
    opal_table: Callable[..., Any],
    pl: Any,
    render_notebook_baserender_record: Callable[..., Mapping[str, Any]] | None,
    selected_baserender_round: int | None,
    selected_baserender_status_rows: Iterable[Mapping[str, Any]],
    selected_campaign_baserender_contract: Mapping[str, Any] | None,
    selected_campaign_labels_df: Any,
) -> Any:
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
        payload = _require_callable(
            render_notebook_baserender_record,
            "render_notebook_baserender_record",
        )(baserender_record_row, contract)
        slug = str((baserender_campaign_model or {}).get("campaign", {}).get("slug") or "unknown campaign")
        round_text = f"round {selected_baserender_round}" if selected_baserender_round is not None else "unknown round"
        visual = mo.image(
            payload["image_bytes"],
            alt=f"{payload['alt_text']} Selected in campaign {slug}, {round_text}.",
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
    detail_items = [
        opal_table(pl.DataFrame(list(selected_baserender_status_rows)), page_size=8),
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
    return mo.vstack(
        [
            controls,
            visual,
            mo.accordion({"Selection evidence": mo.vstack(detail_items, gap=0.35)}, multiple=True),
        ],
        gap=0.45,
    )


def _render_collection_plot_panel(
    *,
    build_notebook_collection_visual_card_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None,
    collection_visual_description: Callable[[Mapping[str, Any]], str] | None,
    controls: Any,
    mo: Any,
    opal_table: Callable[..., Any],
    pl: Any,
    render_notebook_plot_choice_image: Callable[..., Any],
    selected_visual_choice: Mapping[str, Any],
) -> Any:
    details = mo.vstack(
        [
            mo.md(
                _require_callable(
                    collection_visual_description,
                    "collection_visual_description",
                )(selected_visual_choice)
            ),
            opal_table(
                pl.DataFrame(
                    _require_callable(
                        build_notebook_collection_visual_card_rows,
                        "build_notebook_collection_visual_card_rows",
                    )(selected_visual_choice)
                ),
                page_size=12,
            ),
        ],
        gap=0.35,
    )
    return mo.vstack(
        [
            controls,
            render_notebook_plot_choice_image(selected_visual_choice, mo=mo),
            mo.accordion({"Collection plot evidence": details}, multiple=True),
        ],
        gap=0.45,
    )


def _render_campaign_plot_panel(
    *,
    build_notebook_plot_card_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
    build_notebook_plot_method_sections: Callable[[Mapping[str, Any]], Mapping[str, str]],
    controls: Any,
    mo: Any,
    opal_table: Callable[..., Any],
    pl: Any,
    plot_scope_ui: Any,
    render_notebook_plot_choice_image: Callable[..., Any],
    selected_visual_choice: Mapping[str, Any],
    select_notebook_plot_scope: Callable[..., Mapping[str, Any]],
) -> Any:
    choice = select_notebook_plot_scope(
        selected_visual_choice,
        str(plot_scope_ui.value) if plot_scope_ui is not None else None,
    )
    method_sections = build_notebook_plot_method_sections(choice)
    detail_items = [mo.md(text) for text in method_sections.values()]
    detail_items.append(opal_table(pl.DataFrame(build_notebook_plot_card_rows(choice)), page_size=12))
    return mo.vstack(
        [
            controls,
            render_notebook_plot_choice_image(choice, mo=mo),
            mo.accordion({"Plot evidence": mo.vstack(detail_items, gap=0.35)}, multiple=True),
        ],
        gap=0.45,
    )


def _require_callable(value: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
    if value is None:
        raise ValueError(f"{name} is required for this OPAL notebook visual surface.")
    return value


__all__ = ["render_notebook_visual_panel"]
