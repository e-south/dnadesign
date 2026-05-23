from __future__ import annotations

from ._support import block


def render_visual_cells() -> str:
    """Render campaign-set visual selector and manifest-backed plot panel cells."""

    return "\n\n".join((_visual_model_cell(), _visual_selector_cell(), _visual_panel_cell()))


def _visual_model_cell() -> str:
    return block(
        """
        @app.cell
        def _(build_notebook_plot_inventory_rows, build_notebook_visual_surface_model, selected_campaign_model):
            visual_surface_model = build_notebook_visual_surface_model(selected_campaign_model)
            plot_choices = visual_surface_model["choices"]
            plot_inventory_rows = build_notebook_plot_inventory_rows(visual_surface_model)
            plot_inventory_counts = visual_surface_model["inventory_status_counts"]
            return plot_choices, plot_inventory_rows, plot_inventory_counts
        """
    )


def _visual_selector_cell() -> str:
    return block(
        """
        @app.cell
        def _(mo, plot_choices):
            if plot_choices:
                _labels = [choice["label"] for choice in plot_choices]
                plot_ui = mo.ui.dropdown(_labels, value=_labels[0], label="Visual surface")
            else:
                plot_ui = None
            return plot_ui
        """
    )


def _visual_panel_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            Path,
            build_notebook_plot_card_rows,
            build_notebook_plot_method_sections,
            mo,
            pl,
            plot_choices,
            plot_inventory_counts,
            plot_inventory_rows,
            plot_ui,
        ):
            if plot_ui is None:
                _lines = ["No written manifest-backed plot media are available for this campaign."]
                if plot_inventory_counts:
                    _parts = [
                        f"{key}={value}"
                        for key, value in sorted(plot_inventory_counts.items())
                    ]
                    _lines.append("Plot inventory: " + ", ".join(_parts))
                _items = [mo.md("\\n".join(_lines))]
                if plot_inventory_rows:
                    _items.append(mo.ui.table(pl.DataFrame(plot_inventory_rows), page_size=12))
                plot_panel = mo.vstack(_items, gap=0.45)
            else:
                _selected = str(plot_ui.value)
                _choice = next(choice for choice in plot_choices if choice["label"] == _selected)
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
                _controls = mo.hstack([plot_ui], justify="start", align="end", wrap=True, gap=0.35)
                _method_sections = build_notebook_plot_method_sections(_choice)
                plot_panel = mo.vstack(
                    [
                        _controls,
                        _plot_image(_choice),
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


__all__ = ["render_visual_cells"]
