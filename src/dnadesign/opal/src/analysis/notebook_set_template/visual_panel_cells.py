# ruff: noqa: E501

from __future__ import annotations

from ._support import block


def render_visual_panel_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            CAMPAIGN_SET_BASERENDER_SURFACE_KIND, Path, active_view_mode, baserender_campaign_model,
            baserender_record_id, baserender_record_row, baserender_record_selector, baserender_role_ui,
            baserender_round_ui, baserender_run_ui, build_notebook_baserender_contract_rows,
            build_notebook_baserender_label_rows, build_notebook_collection_visual_card_rows,
            collection_visual_description, build_notebook_no_plot_scope_rows, build_notebook_plot_card_rows,
            build_notebook_plot_method_sections, mo, opal_table, pl, plot_inventory_counts, plot_inventory_rows,
            plot_scope_ui, plot_ui, render_notebook_baserender_record, selected_baserender_round,
            selected_baserender_status_rows, selected_campaign_baserender_contract, selected_campaign_labels_df,
            selected_campaign_model, selected_visual_choice, select_notebook_plot_scope,
        ):
            def _image(plot_choice):
                _path = Path(str(plot_choice.get("path") or ""))
                _path_label = str(plot_choice.get("path_label") or plot_choice.get("path") or "not generated")
                if not _path.exists():
                    return mo.md(f"Plot media missing: `{_path_label}`")
                return mo.image(
                    _path.read_bytes(),
                    alt=str(plot_choice.get("alt_text") or plot_choice.get("title") or plot_choice.get("label")),
                    caption=str(plot_choice.get("caption") or "") or None,
                    rounded=True,
                    style={
                        "width": "auto", "max-height": "min(68vh, 760px)", "max-width": "100%", "height": "auto",
                        "object-fit": "contain", "margin": "0 auto", "display": "block", "background": "white",
                    },
                )

            _is_baserender = (
                selected_visual_choice is not None
                and selected_visual_choice.get("surface_kind") in {"baserender", CAMPAIGN_SET_BASERENDER_SURFACE_KIND}
            )
            _control_items = []
            if plot_ui is not None:
                _control_items.append(plot_ui)
            if _is_baserender:
                for _control in (baserender_role_ui, baserender_round_ui, """
        "baserender_run_ui, baserender_record_selector):"
        """
                    if _control is not None:
                        _control_items.append(_control)
            elif plot_scope_ui is not None:
                _control_items.append(plot_scope_ui)
            _controls = mo.hstack(_control_items, justify="start", align="end", wrap=True, gap=0.35) if _control_items else mo.md("")

            if selected_visual_choice is None:
                if active_view_mode == "Campaign set":
                    _lines = ["No campaign-set comparison visuals are available."]
                else:
                    _lines = ["No plot media are available for this campaign."]
                    if plot_inventory_counts:
                        _parts = [f"{key}={value}" for key, value in sorted(plot_inventory_counts.items())]
                        _lines.append("Plot inventory: " + ", ".join(_parts))
                _items = [_controls, mo.md("\\n".join(_lines))]
                if active_view_mode != "Campaign set":
                    _scope_rows = build_notebook_no_plot_scope_rows(selected_campaign_model)
                    _scope_panel = opal_table(pl.DataFrame(_scope_rows), page_size=12)
                    _items.append(mo.accordion({"Current campaign and plot evidence": _scope_panel}, multiple=True))
                    if plot_inventory_rows:
                        _items.append(opal_table(pl.DataFrame(plot_inventory_rows), page_size=12))
                plot_panel = mo.vstack(_items, gap=0.45)
            elif _is_baserender:
                _record_id = str(baserender_record_id)
                _label_rows = build_notebook_baserender_label_rows(selected_campaign_labels_df, record_id=_record_id, round_value=selected_baserender_round)
                _label_panel = (
                    opal_table(pl.DataFrame(_label_rows), page_size=5)
                    if _label_rows
                    else mo.md("No observed label is available for this selected sequence and round.")
                )
                if baserender_record_row is None:
                    _visual = mo.md("No contract-valid selected sequence is available for this round/run.")
                else:
                    try:
                        _payload = render_notebook_baserender_record(baserender_record_row, selected_campaign_baserender_contract)
                        _slug = str(baserender_campaign_model["campaign"]["slug"])
                        _round_text = f"round {selected_baserender_round}" if selected_baserender_round is not None else "unknown round"
                        _visual = mo.image(
                            _payload["image_bytes"],
                            alt=f"{_payload['alt_text']} Selected in campaign {_slug}, {_round_text}.",
                            caption=str(_payload["caption"]),
                            rounded=True,
                            style={
                                "width": "100%", "max-width": "100%", "height": "auto", "object-fit": "contain",
                                "margin": "0", "display": "block", "background-color": "#FFFFFF",
                            },
                        )
                    except Exception as exc:
                        _visual = mo.md(f"Selected sequence render failed: `{exc}`")
                _details = {
                    "What this sequence view shows": mo.md(
                        "Selected sequence for the chosen role, round, and run; use the selector to change records."
                    ),
                    "Selection scope": opal_table(pl.DataFrame(selected_baserender_status_rows), page_size=8),
                    "Label": _label_panel,
                    "Render contract": opal_table(pl.DataFrame(build_notebook_baserender_contract_rows(selected_campaign_baserender_contract)), page_size=8),
                }
                plot_panel = mo.vstack([_controls, _visual, mo.accordion(_details, multiple=True)], gap=0.45)
            elif active_view_mode == "Campaign set":
                _visual = _image(selected_visual_choice)
                _details = {
                    "What this visual shows": mo.md(collection_visual_description(selected_visual_choice)),
                    "Evidence": opal_table(pl.DataFrame(build_notebook_collection_visual_card_rows(selected_visual_choice)), page_size=12)
                }
                plot_panel = mo.vstack([_controls, _visual, mo.accordion(_details, multiple=True)], gap=0.45)
            else:
                _choice = select_notebook_plot_scope(selected_visual_choice, str(plot_scope_ui.value) if plot_scope_ui is not None else None)
                _visual = _image(_choice)
                _method_sections = build_notebook_plot_method_sections(_choice)
                _details = {
                    **{label: mo.md(text) for label, text in _method_sections.items()},
                    "Evidence": opal_table(pl.DataFrame(build_notebook_plot_card_rows(_choice)), page_size=12),
                    "Plot inventory": opal_table(pl.DataFrame(plot_inventory_rows), page_size=12),
                }
                plot_panel = mo.vstack([_controls, _visual, mo.accordion(_details, multiple=True)], gap=0.45)
            return plot_panel
        """
    )


__all__ = ["render_visual_panel_cell"]
