from __future__ import annotations

from ._support import block


def render_visual_panel_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            Path,
            build_notebook_campaign_set_metric_comparison_rows,
            build_notebook_no_plot_scope_rows,
            build_notebook_plot_card_rows,
            build_notebook_plot_method_sections,
            campaigns,
            comparison_group_key,
            comparison_group_ui,
            mo,
            pl,
            plot_inventory_counts,
            plot_inventory_rows,
            plot_scope_ui,
            plot_ui,
            render_notebook_campaign_set_metric_comparison_image,
            selected_plot_choice,
            select_notebook_plot_scope,
            selected_campaign_model,
        ):
            if selected_plot_choice is None:
                _lines = ["No written manifest-backed plot media are available for this campaign."]
                if plot_inventory_counts:
                    _parts = [
                        f"{key}={value}"
                        for key, value in sorted(plot_inventory_counts.items())
                    ]
                    _lines.append("Plot inventory: " + ", ".join(_parts))
                _items = [mo.md("\\n".join(_lines))]
                _scope_rows = build_notebook_no_plot_scope_rows(selected_campaign_model)
                _scope_panel = mo.ui.table(pl.DataFrame(_scope_rows), page_size=12)
                _items.append(mo.accordion({"Current scope and probe implication": _scope_panel}, multiple=True))
                if plot_inventory_rows:
                    _items.append(mo.ui.table(pl.DataFrame(plot_inventory_rows), page_size=12))
                plot_panel = mo.vstack(_items, gap=0.45)
            else:
                _choice = select_notebook_plot_scope(
                    selected_plot_choice,
                    str(plot_scope_ui.value) if plot_scope_ui is not None else None,
                )
                def _plot_image(plot_choice):
                    _path = Path(plot_choice["path"])
                    if not _path.exists():
                        return mo.md(f"Plot media missing: `{plot_choice['path_label']}`")
                    return mo.image(
                        _path.read_bytes(),
                        alt=str(plot_choice.get("alt_text") or plot_choice["title"]),
                        caption=str(plot_choice.get("caption") or "") or None,
                        rounded=True,
                        style={
                            "width": "auto",
                            "max-height": "min(68vh, 760px)",
                            "max-width": "100%",
                            "height": "auto",
                            "object-fit": "contain",
                            "overflow": "auto",
                            "margin": "0 auto",
                            "display": "block",
                            "background": "white",
                        },
                    )
                _control_items = [plot_ui]
                if plot_scope_ui is not None:
                    _control_items.append(plot_scope_ui)
                if comparison_group_ui is not None:
                    _control_items.append(comparison_group_ui)
                _controls = mo.hstack(_control_items, justify="start", align="end", wrap=True, gap=0.35)
                _comparison_view = None
                _active_comparison_group = (
                    str(comparison_group_ui.value) if comparison_group_ui is not None else comparison_group_key
                )
                if _active_comparison_group is not None and _choice.get("kind") == "metric_over_rounds":
                    _comparison_rows = build_notebook_campaign_set_metric_comparison_rows(
                        campaigns,
                        plot_name=str(_choice.get("name") or ""),
                        group_key=str(_active_comparison_group),
                    )
                    _comparison_payload = render_notebook_campaign_set_metric_comparison_image(
                        _comparison_rows,
                        title=str(_choice.get("title") or _choice.get("name") or "Campaign-set comparison"),
                        group_key=str(_active_comparison_group),
                    )
                    if _comparison_payload is not None:
                        _comparison_view = mo.image(
                            _comparison_payload["image_bytes"],
                            alt=str(_comparison_payload["alt_text"]),
                            caption=str(_comparison_payload["caption"]),
                            rounded=True,
                            style={
                                "width": "auto",
                                "max-height": "min(58vh, 620px)",
                                "max-width": "100%",
                                "height": "auto",
                                "object-fit": "contain",
                                "overflow": "auto",
                                "margin": "0 auto",
                                "display": "block",
                                "background": "white",
                            },
                        )
                _visual_items = [_controls, _plot_image(_choice)]
                if _comparison_view is not None:
                    _visual_items.append(mo.md("#### Campaign-set comparison"))
                    _visual_items.append(_comparison_view)
                _method_sections = build_notebook_plot_method_sections(_choice)
                plot_panel = mo.vstack(
                    [
                        *_visual_items,
                        mo.accordion(
                            {
                                **{label: mo.md(text) for label, text in _method_sections.items()},
                                "Evidence": mo.ui.table(
                                    pl.DataFrame(build_notebook_plot_card_rows(_choice)),
                                    page_size=12,
                                ),
                                "Plot inventory": mo.ui.table(
                                    pl.DataFrame(plot_inventory_rows),
                                    page_size=12,
                                ),
                            },
                            multiple=True,
                        ),
                    ],
                    gap=0.45,
                )
            return plot_panel
        """
    )


__all__ = ["render_visual_panel_cell"]
