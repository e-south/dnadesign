from __future__ import annotations

from textwrap import dedent


def render_visual_surface_cells() -> str:
    """Render generated marimo cells for the OPAL visual command surface."""

    return dedent(
        """
        @app.cell
        def _(Path, build_notebook_visual_surface_model, notebook_view_model, plot_entries):
            visual_surface_model = build_notebook_visual_surface_model(
                notebook_view_model,
                plot_entries=plot_entries,
            )
            plots_dir = Path(visual_surface_model["plots_dir"])
            plot_choices = visual_surface_model["choices"]
            missing_outputs = visual_surface_model["missing_outputs"]
            stale_plot_artifacts = visual_surface_model["stale_artifacts"]
            return plots_dir, plot_choices, missing_outputs, stale_plot_artifacts


        @app.cell
        def _(
            mo,
            notebook_baserender_contract,
            plot_cfg_error,
            plot_choices,
            plots_dir,
            missing_outputs,
            stale_plot_artifacts,
        ):
            visual_surface_ui = None
            visual_surface_choices = []
            if plot_cfg_error:
                visual_surface_note = f"Plot config unavailable: `{plot_cfg_error}`"
            else:
                if notebook_baserender_contract.get("available"):
                    visual_surface_choices.append(
                        {
                            "label": "Record render",
                            "kind": "baserender",
                            "plot_label": None,
                        }
                    )
                visual_surface_choices.extend(
                    {
                        "label": plot_choice["title"],
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
                visual_surface_note = "\\n".join(_lines)
                if visual_surface_choices:
                    labels = [choice["label"] for choice in visual_surface_choices]
                    visual_surface_ui = mo.ui.dropdown(labels, value=labels[0], label="Visual")
            return visual_surface_ui, visual_surface_choices, visual_surface_note


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
            render_notebook_baserender_record,
            round_ui,
            visual_surface_choices,
            visual_surface_note,
            visual_surface_ui,
        ):
            if visual_surface_ui is None:
                plot_panel = mo.md(visual_surface_note)
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
                            "width": "100%",
                            "max-width": "100%",
                            "height": "auto",
                            "object-fit": "contain",
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
                            visual = mo.image(
                                payload["image_bytes"],
                                alt=str(payload["alt_text"]),
                                caption=str(payload["caption"]),
                                rounded=True,
                                style={
                                    "width": "100%",
                                    "max-width": "100%",
                                    "height": "auto",
                                    "object-fit": "contain",
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
                            )
                        },
                        multiple=True,
                    )
                else:
                    choice = next(
                        plot_choice for plot_choice in plot_choices if plot_choice["label"] == surface["plot_label"]
                    )
                    visual = _plot_image(choice)
                    method_sections = build_notebook_plot_method_sections(choice)
                    details = mo.accordion(
                        {
                            **{label: mo.md(text) for label, text in method_sections.items()},
                            "Evidence": mo.ui.table(
                                pl.DataFrame(build_notebook_plot_card_rows(choice)),
                                page_size=12,
                            ),
                        },
                        multiple=True,
                    )
                plot_panel = mo.vstack([mo.md(visual_surface_note), controls, visual, details], gap=0.45)
            return plot_panel
        """
    ).strip("\n")
