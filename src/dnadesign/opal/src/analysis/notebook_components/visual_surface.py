from __future__ import annotations

from textwrap import dedent


def render_visual_surface_cells() -> str:
    """Render generated marimo cells for the OPAL visual command surface."""

    return dedent(
        """
        @app.cell
        def _(
            Path,
            build_notebook_plot_inventory_rows,
            build_notebook_visual_surface_model,
            notebook_view_model,
            plot_entries,
        ):
            visual_surface_model = build_notebook_visual_surface_model(
                notebook_view_model,
                plot_entries=plot_entries,
            )
            plots_dir = Path(visual_surface_model["plots_dir"])
            plot_choices = visual_surface_model["choices"]
            missing_outputs = visual_surface_model["missing_outputs"]
            stale_plot_artifacts = visual_surface_model["stale_artifacts"]
            plot_inventory_rows = build_notebook_plot_inventory_rows(visual_surface_model)
            plot_inventory_counts = visual_surface_model["inventory_status_counts"]
            return (
                plots_dir,
                plot_choices,
                missing_outputs,
                stale_plot_artifacts,
                plot_inventory_rows,
                plot_inventory_counts,
            )


        @app.cell
        def _(
            mo,
            notebook_baserender_contract,
            plot_cfg_error,
            plot_choices,
            plot_inventory_counts,
            plots_dir,
            missing_outputs,
            stale_plot_artifacts,
        ):
            visual_surface_ui = None
            visual_surface_choices = []
            if notebook_baserender_contract.get("available"):
                visual_surface_choices.append(
                    {
                        "label": "Record render",
                        "kind": "baserender",
                        "plot_label": None,
                    }
                )
            if plot_cfg_error:
                visual_surface_note = f"Plot config unavailable: `{plot_cfg_error}`"
            else:
                visual_surface_choices.extend(
                    {
                        "label": plot_choice["label"],
                        "kind": "plot",
                        "plot_label": plot_choice["label"],
                    }
                    for plot_choice in plot_choices
                )
                _lines = []
                if not plot_choices:
                    _lines.append(f"No manifest-backed plot outputs found in `{plots_dir}`.")
                    _lines.append("Run `uv run opal plot -c <campaign.yaml>` to generate plots.")
                if missing_outputs:
                    _lines.append(f"Configured plots without outputs: {', '.join(missing_outputs)}")
                if stale_plot_artifacts:
                    _lines.append(f"Stale artifact warnings: `{len(stale_plot_artifacts)}`.")
                if plot_inventory_counts:
                    _parts = [
                        f"{key}={value}"
                        for key, value in sorted(plot_inventory_counts.items())
                    ]
                    _lines.append("Plot inventory: " + ", ".join(_parts))
                visual_surface_note = "\\n".join(_lines)
            if visual_surface_choices:
                labels = [choice["label"] for choice in visual_surface_choices]
                visual_surface_ui = mo.ui.dropdown(labels, value=labels[0], label="Visual surface")
            return visual_surface_ui, visual_surface_choices, visual_surface_note


        @app.cell
        def _(
            build_notebook_plot_scope_options,
            mo,
            plot_choices,
        ):
            plot_scope_controls = {}
            plot_scope_options_by_plot = {}
            for plot_choice in plot_choices:
                _scope_options = build_notebook_plot_scope_options(plot_choice)
                plot_scope_options_by_plot[plot_choice["label"]] = _scope_options
                if len(_scope_options) > 1:
                    _scope_labels = [option["label"] for option in _scope_options]
                    _scope_control_label = str(_scope_options[0].get("control_label") or "Plot scope")
                    plot_scope_controls[plot_choice["label"]] = mo.ui.dropdown(
                        _scope_labels,
                        value=_scope_labels[0],
                        label=_scope_control_label,
                    )
            return plot_scope_controls, plot_scope_options_by_plot


        @app.cell
        def _(
            Path,
            baserender_record_row,
            baserender_record_selector,
            build_notebook_baserender_label_rows,
            build_notebook_plot_method_sections,
            build_notebook_baserender_contract_rows,
            build_notebook_plot_card_rows,
            labels_df,
            mo,
            notebook_baserender_contract,
            pl,
            plot_choices,
            plot_inventory_rows,
            plot_scope_controls,
            render_notebook_baserender_record,
            round_ui,
            select_notebook_plot_scope,
            visual_surface_choices,
            visual_surface_note,
            visual_surface_ui,
        ):
            if visual_surface_ui is None:
                _items = [mo.md(visual_surface_note)]
                if plot_inventory_rows:
                    _items.append(mo.ui.table(pl.DataFrame(plot_inventory_rows), page_size=12))
                plot_panel = mo.vstack(_items, gap=0.45)
            else:
                selected = str(visual_surface_ui.value)
                surface = next(
                    (choice for choice in visual_surface_choices if choice["label"] == selected),
                    None,
                )
                if surface is None:
                    raise ValueError(f"Visual selection not found: {selected}")

                def _plot_image(plot_choice):
                    media_path = Path(plot_choice["path"])
                    if not media_path.exists():
                        return mo.md(f"Plot media missing: `{plot_choice['path_label']}`")
                    return mo.image(
                        media_path.read_bytes(),
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

                controls = mo.hstack([visual_surface_ui], justify="start", align="end", wrap=True, gap=0.35)
                if surface["kind"] == "baserender":
                    _controls = [visual_surface_ui, baserender_record_selector]
                    if round_ui is not None:
                        _controls.insert(1, round_ui)
                    controls = mo.hstack(_controls, justify="start", align="end", wrap=True, gap=0.35)
                    label_rows = build_notebook_baserender_label_rows(
                        labels_df,
                        record_id=str(baserender_record_selector.value),
                        round_value=int(round_ui.value) if round_ui is not None else None,
                    )
                    label_view = (
                        mo.ui.table(pl.DataFrame(label_rows), page_size=5)
                        if label_rows
                        else mo.md("No observed label is available for this record and round.")
                    )
                    if baserender_record_row is None:
                        visual = mo.md("No contract-valid record is available for this selection.")
                    else:
                        try:
                            payload = render_notebook_baserender_record(
                                baserender_record_row,
                                notebook_baserender_contract,
                            )
                            _round_text = (
                                f"round {int(round_ui.value)}"
                                if round_ui is not None
                                else "the selected round"
                            )
                            _label_state = "observed label available" if label_rows else "no observed label available"
                            visual = mo.image(
                                payload["image_bytes"],
                                alt=f"{payload['alt_text']} Selected scope: {_round_text}; {_label_state}.",
                                caption=str(payload["caption"]),
                                rounded=True,
                                style={
                                    "width": "auto",
                                    "max-height": "min(58vh, 520px)",
                                    "max-width": "100%",
                                    "height": "auto",
                                    "object-fit": "contain",
                                    "overflow": "auto",
                                    "background": "white",
                                    "display": "block",
                                    "margin": "0 auto",
                                },
                            )
                        except Exception as exc:
                            visual = mo.md(f"Record render failed: `{exc}`")
                    details = mo.accordion(
                        {
                            "Label": label_view,
                            "Render contract": mo.ui.table(
                                pl.DataFrame(build_notebook_baserender_contract_rows(notebook_baserender_contract)),
                                page_size=8,
                            ),
                            "Plot inventory": mo.ui.table(
                                pl.DataFrame(plot_inventory_rows),
                                page_size=12,
                            ),
                        },
                        multiple=True,
                    )
                else:
                    selected_plot_choice = next(
                        plot_choice for plot_choice in plot_choices if plot_choice["label"] == surface["plot_label"]
                    )
                    plot_scope_ui = plot_scope_controls.get(surface["plot_label"])
                    choice = select_notebook_plot_scope(
                        selected_plot_choice,
                        str(plot_scope_ui.value) if plot_scope_ui is not None else None,
                    )
                    _controls = [visual_surface_ui]
                    if plot_scope_ui is not None:
                        _controls.append(plot_scope_ui)
                    controls = mo.hstack(_controls, justify="start", align="end", wrap=True, gap=0.35)
                    visual = _plot_image(choice)
                    method_sections = build_notebook_plot_method_sections(choice)
                    details = mo.accordion(
                        {
                            **{label: mo.md(text) for label, text in method_sections.items()},
                            "Evidence": mo.ui.table(
                                pl.DataFrame(build_notebook_plot_card_rows(choice)),
                                page_size=12,
                            ),
                            "Plot inventory": mo.ui.table(
                                pl.DataFrame(plot_inventory_rows),
                                page_size=12,
                            ),
                        },
                        multiple=True,
                    )
                plot_panel = mo.vstack([mo.md(visual_surface_note), controls, visual, details], gap=0.45)
            return plot_panel
        """
    ).strip("\n")
