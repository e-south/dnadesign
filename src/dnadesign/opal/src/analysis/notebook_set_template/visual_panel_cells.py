from __future__ import annotations

from ._support import block


def render_visual_panel_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            Path,
            active_view_mode,
            build_notebook_collection_visual_card_rows,
            build_notebook_no_plot_scope_rows,
            build_notebook_plot_card_rows,
            build_notebook_plot_method_sections,
            mo,
            opal_table,
            pl,
            plot_inventory_counts,
            plot_inventory_rows,
            plot_scope_ui,
            plot_ui,
            selected_campaign_model,
            selected_visual_choice,
            select_notebook_plot_scope,
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

            _control_items = []
            if plot_ui is not None:
                _control_items.append(plot_ui)
            if plot_scope_ui is not None:
                _control_items.append(plot_scope_ui)
            _controls = (
                mo.hstack(_control_items, justify="start", align="end", wrap=True, gap=0.35)
                if _control_items
                else mo.md("")
            )

            if selected_visual_choice is None:
                if active_view_mode == "Campaign set":
                    _lines = ["No manifest-backed campaign-set comparison visuals are available."]
                else:
                    _lines = ["No written manifest-backed plot media are available for this campaign."]
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
            elif active_view_mode == "Campaign set":
                _visual = _image(selected_visual_choice)
                _details = {
                    "Evidence": opal_table(
                        pl.DataFrame(build_notebook_collection_visual_card_rows(selected_visual_choice)), page_size=12
                    )
                }
                plot_panel = mo.vstack([_controls, _visual, mo.accordion(_details, multiple=True)], gap=0.45)
            else:
                _choice = select_notebook_plot_scope(
                    selected_visual_choice,
                    str(plot_scope_ui.value) if plot_scope_ui is not None else None,
                )
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
