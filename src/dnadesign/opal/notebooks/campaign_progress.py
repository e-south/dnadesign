import marimo

__generated_with = "0.19.4"

app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from dnadesign.opal.notebooks.api.generated import (
        build_campaign_set_notebook_view_model,
        build_campaign_set_round_options,
        build_notebook_artifact_garden_rows,
        build_notebook_artifact_garden_summary_rows,
        build_notebook_at_a_glance_rows,
        build_notebook_campaign_header_lines,
        build_notebook_campaign_summary_row,
        build_notebook_change_rows,
        build_notebook_change_summary_rows,
        build_notebook_evidence_rows,
        build_notebook_metric_definition_rows,
        build_notebook_plot_card_rows,
        build_notebook_plot_inventory_rows,
        build_notebook_plot_method_sections,
        build_notebook_plot_scope_options,
        build_notebook_validity_rows,
        build_notebook_visual_surface_model,
        find_notebook_repo_root,
        list_notebook_campaign_paths,
        select_notebook_plot_scope,
    )

    return (
        Path,
        build_campaign_set_notebook_view_model,
        build_campaign_set_round_options,
        build_notebook_artifact_garden_rows,
        build_notebook_artifact_garden_summary_rows,
        build_notebook_at_a_glance_rows,
        build_notebook_campaign_header_lines,
        build_notebook_campaign_summary_row,
        build_notebook_change_rows,
        build_notebook_change_summary_rows,
        build_notebook_evidence_rows,
        build_notebook_metric_definition_rows,
        build_notebook_plot_card_rows,
        build_notebook_plot_inventory_rows,
        build_notebook_plot_method_sections,
        build_notebook_plot_scope_options,
        build_notebook_validity_rows,
        build_notebook_visual_surface_model,
        find_notebook_repo_root,
        list_notebook_campaign_paths,
        mo,
        pl,
        select_notebook_plot_scope,
    )


@app.cell
def _(Path, find_notebook_repo_root):
    notebook_path = Path(__file__).resolve()
    repo_root = find_notebook_repo_root(notebook_path)
    return (repo_root,)


@app.cell
def _(list_notebook_campaign_paths, repo_root):
    config_paths = list_notebook_campaign_paths(repo_root)
    return (config_paths,)


@app.cell
def _(build_campaign_set_round_options, config_paths, mo):
    if config_paths:
        round_options = build_campaign_set_round_options(config_paths)
    else:
        round_options = ["latest"]
    round_ui = mo.ui.dropdown(round_options, value=round_options[0], label="Round")
    return round_options, round_ui


@app.cell
def _(build_campaign_set_notebook_view_model, config_paths, round_ui):
    selected_round_selector = str(round_ui.value)
    if config_paths:
        campaign_set_view_model = build_campaign_set_notebook_view_model(
            config_paths,
            round_selector=selected_round_selector,
        )
        campaigns = campaign_set_view_model["campaigns"]
    else:
        campaign_set_view_model = {
            "schema_version": "opal.notebook_campaign_set_view_model.v1",
            "campaign_count": 0,
            "campaigns": [],
            "warnings": [],
        }
        campaigns = []
    return campaign_set_view_model, campaigns, selected_round_selector


@app.cell
def _(build_notebook_campaign_summary_row, campaign_set_view_model, campaigns, mo, pl, selected_round_selector):
    campaign_rows = [build_notebook_campaign_summary_row(campaign_model) for campaign_model in campaigns]
    if campaign_rows:
        campaign_labels = [f"{index + 1}. {row['label']}" for index, row in enumerate(campaign_rows)]
        campaign_ui = mo.ui.dropdown(campaign_labels, value=campaign_labels[0], label="Campaign")
        campaign_summary_df = pl.DataFrame(campaign_rows)
    else:
        campaign_labels = []
        campaign_ui = None
        campaign_summary_df = pl.DataFrame([])
    header_md = mo.md(
        "# Campaigns\n\n"
        f"`{campaign_set_view_model['campaign_count']}` campaigns. "
        f"Round selector: `{selected_round_selector}`."
    )
    return campaign_labels, campaign_summary_df, campaign_ui, header_md


@app.cell
def _(campaign_labels, campaign_ui, campaigns):
    if campaign_ui is None:
        selected_campaign_model = None
        selected_label = None
    else:
        selected_label = str(campaign_ui.value)
        selected_index = campaign_labels.index(selected_label)
        selected_campaign_model = campaigns[selected_index]
    return selected_campaign_model, selected_label


@app.cell
def _(
    build_notebook_at_a_glance_rows,
    build_notebook_campaign_header_lines,
    build_notebook_validity_rows,
    mo,
    pl,
    selected_campaign_model,
):
    if selected_campaign_model is None:
        selected_campaign_header_md = mo.md("## No campaign selected")
        selected_overview_panel = mo.md("No campaign configs were found.")
        selected_validity_md = mo.md("")
    else:
        selected_campaign_header_md = mo.md(
            "\n".join(build_notebook_campaign_header_lines(selected_campaign_model, heading_level=2))
        )
        selected_overview_panel = mo.ui.table(
            pl.DataFrame(build_notebook_at_a_glance_rows(selected_campaign_model)),
            page_size=14,
        )
        selected_validity_md = mo.ui.table(
            pl.DataFrame(build_notebook_validity_rows(selected_campaign_model)),
            page_size=14,
        )
    return selected_campaign_header_md, selected_overview_panel, selected_validity_md


@app.cell
def _(build_notebook_plot_inventory_rows, build_notebook_visual_surface_model, selected_campaign_model):
    if selected_campaign_model is None:
        visual_surface_model = {"choices": [], "inventory_status_counts": {}, "stale_artifacts": []}
        plot_choices = []
        plot_inventory_rows = []
        plot_inventory_counts = {}
    else:
        visual_surface_model = build_notebook_visual_surface_model(selected_campaign_model)
        plot_choices = visual_surface_model["choices"]
        plot_inventory_rows = build_notebook_plot_inventory_rows(visual_surface_model)
        plot_inventory_counts = visual_surface_model["inventory_status_counts"]
    return plot_choices, plot_inventory_rows, plot_inventory_counts


@app.cell
def _(mo, plot_choices):
    if plot_choices:
        plot_labels = [choice["label"] for choice in plot_choices]
        plot_ui = mo.ui.dropdown(plot_labels, value=plot_labels[0], label="Visual surface")
    else:
        plot_ui = None
    return plot_ui


@app.cell
def _(build_notebook_plot_scope_options, mo, plot_choices):
    plot_scope_controls = {}
    plot_scope_options_by_plot = {}
    for plot_choice in plot_choices:
        plot_scope_options = build_notebook_plot_scope_options(plot_choice)
        plot_scope_options_by_plot[plot_choice["label"]] = plot_scope_options
        if len(plot_scope_options) > 1:
            scope_labels = [option["label"] for option in plot_scope_options]
            plot_scope_controls[plot_choice["label"]] = mo.ui.dropdown(
                scope_labels,
                value=scope_labels[0],
                label="Plot scope",
            )
    return plot_scope_controls, plot_scope_options_by_plot


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
    plot_scope_controls,
    plot_ui,
    select_notebook_plot_scope,
):
    if plot_ui is None:
        lines = ["No written manifest-backed plot media are available for this campaign."]
        if plot_inventory_counts:
            parts = [f"{key}={value}" for key, value in sorted(plot_inventory_counts.items())]
            lines.append("Plot inventory: " + ", ".join(parts))
        items = [mo.md("\n".join(lines))]
        if plot_inventory_rows:
            items.append(mo.ui.table(pl.DataFrame(plot_inventory_rows), page_size=12))
        plot_panel = mo.vstack(items, gap=0.45)
    else:
        selected = str(plot_ui.value)
        selected_plot_choice = next(choice for choice in plot_choices if choice["label"] == selected)
        plot_scope_ui = plot_scope_controls.get(selected)
        choice = select_notebook_plot_scope(
            selected_plot_choice,
            str(plot_scope_ui.value) if plot_scope_ui is not None else None,
        )

        def plot_image(plot_choice):
            path = Path(plot_choice["path"])
            if not path.exists():
                return mo.md(f"Plot media missing: `{plot_choice['path_label']}`")
            return mo.image(
                path.read_bytes(),
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

        control_items = [plot_ui]
        if plot_scope_ui is not None:
            control_items.append(plot_scope_ui)
        _plot_controls = mo.hstack(control_items, justify="start", align="end", wrap=True, gap=0.35)
        method_sections = build_notebook_plot_method_sections(choice)
        plot_panel = mo.vstack(
            [
                _plot_controls,
                plot_image(choice),
                mo.accordion(
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
                ),
            ],
            gap=0.45,
        )
    return plot_panel


@app.cell
def _(build_notebook_evidence_rows, mo, pl, selected_campaign_model):
    if selected_campaign_model is None:
        evidence_panel = mo.md("")
    else:
        evidence_rows = build_notebook_evidence_rows(selected_campaign_model)
        if evidence_rows:
            evidence_panel = mo.ui.table(pl.DataFrame(evidence_rows), page_size=10)
        else:
            evidence_panel = mo.md("No warnings or stale artifacts reported for this campaign.")
    return evidence_panel


@app.cell
def _(
    build_notebook_artifact_garden_rows,
    build_notebook_artifact_garden_summary_rows,
    build_notebook_change_rows,
    build_notebook_change_summary_rows,
    build_notebook_metric_definition_rows,
    mo,
    pl,
    selected_campaign_model,
):
    if selected_campaign_model is None:
        metric_definitions_panel = mo.md("")
        changes_panel = mo.md("")
        artifact_garden_panel = mo.md("")
    else:
        metric_rows = build_notebook_metric_definition_rows(selected_campaign_model)
        metric_definitions_panel = (
            mo.ui.table(pl.DataFrame(metric_rows), page_size=10)
            if metric_rows
            else mo.md("No manifest-backed plot metric definitions are available.")
        )

        change_rows = build_notebook_change_rows(selected_campaign_model)
        changes_table = (
            mo.ui.table(pl.DataFrame(change_rows), page_size=10)
            if change_rows
            else mo.md("No round changes are available yet.")
        )
        changes_panel = mo.vstack(
            [
                mo.ui.table(
                    pl.DataFrame(build_notebook_change_summary_rows(selected_campaign_model)),
                    page_size=8,
                ),
                changes_table,
            ]
        )

        artifact_rows = build_notebook_artifact_garden_rows(selected_campaign_model)
        artifact_summary_rows = build_notebook_artifact_garden_summary_rows(selected_campaign_model)
        artifact_rows_panel = (
            mo.ui.table(pl.DataFrame(artifact_rows), page_size=10)
            if artifact_rows
            else mo.md("No artifact garden rows are available.")
        )
        artifact_garden_panel = mo.vstack(
            [
                mo.ui.table(pl.DataFrame(artifact_summary_rows), page_size=10),
                artifact_rows_panel,
            ]
        )
    return artifact_garden_panel, changes_panel, metric_definitions_panel


@app.cell
def _(
    artifact_garden_panel,
    campaign_summary_df,
    campaign_ui,
    changes_panel,
    evidence_panel,
    header_md,
    metric_definitions_panel,
    mo,
    plot_panel,
    round_ui,
    selected_campaign_header_md,
    selected_overview_panel,
    selected_validity_md,
):
    _controls = [round_ui]
    if campaign_ui is not None:
        _controls.append(campaign_ui)
    mo.vstack(
        [
            header_md,
            mo.hstack(_controls, justify="start", align="end", wrap=True, gap=0.35),
            selected_campaign_header_md,
            plot_panel,
            mo.accordion(
                {
                    "Campaigns at a glance": mo.ui.table(campaign_summary_df, page_size=12),
                    "Selected campaign": selected_overview_panel,
                    "Validity": selected_validity_md,
                    "Changes": changes_panel,
                    "Metric definitions": metric_definitions_panel,
                    "Artifacts": artifact_garden_panel,
                    "Warnings and stale artifacts": evidence_panel,
                },
                multiple=True,
                lazy=True,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
