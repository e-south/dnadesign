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
        build_notebook_artifact_garden_rows,
        build_notebook_artifact_garden_summary_rows,
        build_notebook_at_a_glance_rows,
        build_notebook_campaign_header_lines,
        build_notebook_campaign_summary_row,
        build_notebook_change_rows,
        build_notebook_change_summary_rows,
        build_notebook_collection_set_choices,
        build_notebook_collection_visual_card_rows,
        build_notebook_collection_visual_choices,
        build_notebook_evidence_rows,
        build_notebook_metric_definition_rows,
        build_notebook_no_plot_scope_rows,
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
        build_notebook_artifact_garden_rows,
        build_notebook_artifact_garden_summary_rows,
        build_notebook_at_a_glance_rows,
        build_notebook_campaign_header_lines,
        build_notebook_campaign_summary_row,
        build_notebook_change_rows,
        build_notebook_change_summary_rows,
        build_notebook_collection_set_choices,
        build_notebook_collection_visual_card_rows,
        build_notebook_collection_visual_choices,
        build_notebook_evidence_rows,
        build_notebook_metric_definition_rows,
        build_notebook_no_plot_scope_rows,
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
def _(build_campaign_set_notebook_view_model, config_paths):
    selected_round_selector = "all"
    if config_paths:
        campaign_set_view_model = build_campaign_set_notebook_view_model(
            config_paths,
            round_selector=selected_round_selector,
        )
        campaigns = campaign_set_view_model["campaigns"]
        collection = campaign_set_view_model.get("collection")
        collection_visuals = campaign_set_view_model.get("collection_visuals") or []
    else:
        campaign_set_view_model = {
            "schema_version": "opal.notebook_campaign_set_view_model.v1",
            "campaign_count": 0,
            "campaigns": [],
            "collection": None,
            "collection_visuals": [],
            "warnings": [],
        }
        campaigns = []
        collection = None
        collection_visuals = []
    return campaign_set_view_model, campaigns, collection, collection_visuals, selected_round_selector


@app.cell
def _(
    build_notebook_campaign_summary_row,
    campaign_set_view_model,
    campaigns,
    collection_visuals,
    mo,
    pl,
    selected_round_selector,
):
    campaign_rows = [build_notebook_campaign_summary_row(campaign_model) for campaign_model in campaigns]
    if campaign_rows:
        campaign_labels = [f"{index + 1}. {row['label']}" for index, row in enumerate(campaign_rows)]
        campaign_ui = mo.ui.dropdown(campaign_labels, value=campaign_labels[0], label="OPAL campaign")
        campaign_summary_df = pl.DataFrame(campaign_rows)
    else:
        campaign_labels = []
        campaign_ui = None
        campaign_summary_df = pl.DataFrame([])
    collection_set_count = len(
        {
            str(visual.get("comparison_set_key") or "")
            for visual in collection_visuals
            if str(visual.get("comparison_set_key") or "")
        }
    )
    collection_clause = (
        f" `{collection_set_count}` campaign sets and `{len(collection_visuals)}` collection visuals are available."
        if collection_visuals
        else ""
    )
    header_md = mo.md(
        "# OPAL Campaign Review\n\n"
        f"There are `{campaign_set_view_model['campaign_count']}` OPAL campaigns available for "
        f"review scope `{selected_round_selector}`.{collection_clause}"
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
            show_column_summaries=False,
        )
        selected_validity_md = mo.ui.table(
            pl.DataFrame(build_notebook_validity_rows(selected_campaign_model)),
            page_size=14,
            show_column_summaries=False,
        )
    return selected_campaign_header_md, selected_overview_panel, selected_validity_md


@app.cell
def _(
    build_notebook_plot_inventory_rows,
    build_notebook_visual_surface_model,
    selected_campaign_model,
):
    if selected_campaign_model is None:
        visual_surface_model = {"choices": [], "inventory_status_counts": {}, "stale_artifacts": []}
        campaign_plot_choices = []
        plot_inventory_rows = []
        plot_inventory_counts = {}
    else:
        visual_surface_model = build_notebook_visual_surface_model(selected_campaign_model)
        campaign_plot_choices = visual_surface_model["choices"]
        plot_inventory_rows = build_notebook_plot_inventory_rows(visual_surface_model)
        plot_inventory_counts = visual_surface_model["inventory_status_counts"]
    return campaign_plot_choices, plot_inventory_counts, plot_inventory_rows


@app.cell
def _(build_notebook_collection_set_choices, collection_visuals):
    collection_set_choices = build_notebook_collection_set_choices(collection_visuals)
    return collection_set_choices


@app.cell
def _(collection_set_choices, mo):
    view_mode_options = ["Campaign", "Campaign set"] if collection_set_choices else ["Campaign"]
    default_view_mode = "Campaign set" if collection_set_choices else "Campaign"
    view_mode_ui = mo.ui.radio(view_mode_options, value=default_view_mode, label="Review surface")
    return default_view_mode, view_mode_options, view_mode_ui


@app.cell
def _(view_mode_ui):
    active_view_mode = str(view_mode_ui.value)
    return active_view_mode


@app.cell
def _(mo):
    collection_set_key_memory, set_collection_set_key_memory = mo.state(None)
    return collection_set_key_memory, set_collection_set_key_memory


@app.cell
def _(active_view_mode, collection_set_choices, collection_set_key_memory, mo, set_collection_set_key_memory):
    if active_view_mode == "Campaign set" and collection_set_choices:
        _labels_by_key = {choice["key"]: choice["label"] for choice in collection_set_choices}
        _keys_by_label = {choice["label"]: choice["key"] for choice in collection_set_choices}
        _preferred_key = collection_set_key_memory()
        _preferred_label = (
            _labels_by_key[_preferred_key] if _preferred_key in _labels_by_key else collection_set_choices[0]["label"]
        )

        def _remember_collection_set(label):
            set_collection_set_key_memory(_keys_by_label[str(label)])

        collection_set_ui = mo.ui.dropdown(
            [choice["label"] for choice in collection_set_choices],
            value=_preferred_label,
            label="Campaign set",
            on_change=_remember_collection_set,
        )
    else:
        collection_set_ui = None
    return collection_set_ui


@app.cell
def _(collection_set_choices, collection_set_ui):
    if collection_set_ui is None:
        selected_collection_set_choice = collection_set_choices[0] if collection_set_choices else None
    else:
        _selected = str(collection_set_ui.value)
        selected_collection_set_choice = next(
            choice for choice in collection_set_choices if choice["label"] == _selected
        )
    return selected_collection_set_choice


@app.cell
def _(mo):
    visual_label_memory, set_visual_label_memory = mo.state(None)
    return set_visual_label_memory, visual_label_memory


@app.cell
def _(
    active_view_mode,
    build_notebook_collection_visual_choices,
    campaign_plot_choices,
    collection_visuals,
    selected_collection_set_choice,
):
    if active_view_mode == "Campaign set":
        _set_key = selected_collection_set_choice.get("key") if selected_collection_set_choice is not None else None
        visual_choices = build_notebook_collection_visual_choices(
            collection_visuals,
            comparison_set_key=_set_key,
        )
    else:
        visual_choices = campaign_plot_choices
    return visual_choices


@app.cell
def _(active_view_mode, mo, set_visual_label_memory, visual_choices, visual_label_memory):
    if visual_choices:
        _labels = [choice["label"] for choice in visual_choices]
        _preferred_visual_label = visual_label_memory()
        _preferred_visual_label = _preferred_visual_label if _preferred_visual_label in _labels else _labels[0]
        if active_view_mode == "Campaign set":
            plot_ui = mo.ui.dropdown(
                _labels,
                value=_preferred_visual_label,
                label="Collection visual",
                on_change=set_visual_label_memory,
            )
        else:
            plot_ui = mo.ui.dropdown(
                _labels,
                value=_preferred_visual_label,
                label="Visual surface",
                on_change=set_visual_label_memory,
            )
    else:
        plot_ui = None
    return plot_ui


@app.cell
def _(plot_ui, visual_choices):
    if plot_ui is None:
        selected_visual_choice = None
    else:
        selected_visual_choice = next(choice for choice in visual_choices if choice["label"] == str(plot_ui.value))
    return selected_visual_choice


@app.cell
def _(active_view_mode, build_notebook_plot_scope_options, mo, selected_visual_choice):
    if active_view_mode == "Campaign set" or selected_visual_choice is None:
        plot_scope_options = []
        plot_scope_ui = None
    else:
        plot_scope_options = build_notebook_plot_scope_options(selected_visual_choice)
        if len(plot_scope_options) > 1:
            _scope_labels = [option["label"] for option in plot_scope_options]
            _scope_control_label = str(plot_scope_options[0].get("control_label") or "Plot scope")
            plot_scope_ui = mo.ui.dropdown(
                _scope_labels,
                value=_scope_labels[0],
                label=_scope_control_label,
            )
        else:
            plot_scope_ui = None
    return plot_scope_options, plot_scope_ui


@app.cell
def _(
    Path,
    active_view_mode,
    build_notebook_collection_visual_card_rows,
    build_notebook_no_plot_scope_rows,
    build_notebook_plot_card_rows,
    build_notebook_plot_method_sections,
    mo,
    pl,
    plot_inventory_counts,
    plot_inventory_rows,
    plot_scope_ui,
    plot_ui,
    selected_campaign_model,
    select_notebook_plot_scope,
    selected_visual_choice,
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

    _control_items = []
    if plot_ui is not None:
        _control_items.append(plot_ui)
    if plot_scope_ui is not None:
        _control_items.append(plot_scope_ui)
    _controls = (
        mo.hstack(_control_items, justify="start", align="end", wrap=True, gap=0.35) if _control_items else mo.md("")
    )

    if selected_visual_choice is None:
        if active_view_mode == "Campaign set":
            _lines = ["No manifest-backed campaign-set comparison visuals are available."]
        else:
            _lines = ["No written manifest-backed plot media are available for this campaign."]
            if plot_inventory_counts:
                _parts = [f"{key}={value}" for key, value in sorted(plot_inventory_counts.items())]
                _lines.append("Plot inventory: " + ", ".join(_parts))
        _items = [_controls, mo.md("\n".join(_lines))]
        if active_view_mode != "Campaign set":
            _scope_rows = build_notebook_no_plot_scope_rows(selected_campaign_model)
            _scope_panel = mo.ui.table(
                pl.DataFrame(_scope_rows),
                page_size=12,
                show_column_summaries=False,
            )
            _items.append(mo.accordion({"Current campaign and plot evidence": _scope_panel}, multiple=True))
            if plot_inventory_rows:
                _items.append(
                    mo.ui.table(
                        pl.DataFrame(plot_inventory_rows),
                        page_size=12,
                        show_column_summaries=False,
                    )
                )
        plot_panel = mo.vstack(_items, gap=0.45)
    elif active_view_mode == "Campaign set":
        _visual = _image(selected_visual_choice)
        _details = {
            "Evidence": mo.ui.table(
                pl.DataFrame(build_notebook_collection_visual_card_rows(selected_visual_choice)),
                page_size=12,
                show_column_summaries=False,
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
            "Evidence": mo.ui.table(
                pl.DataFrame(build_notebook_plot_card_rows(_choice)),
                page_size=12,
                show_column_summaries=False,
            ),
            "Plot inventory": mo.ui.table(
                pl.DataFrame(plot_inventory_rows),
                page_size=12,
                show_column_summaries=False,
            ),
        }
        plot_panel = mo.vstack([_controls, _visual, mo.accordion(_details, multiple=True)], gap=0.45)
    return plot_panel


@app.cell
def _(build_notebook_evidence_rows, mo, pl, selected_campaign_model):
    if selected_campaign_model is None:
        evidence_panel = mo.md("")
    else:
        evidence_rows = build_notebook_evidence_rows(selected_campaign_model)
        if evidence_rows:
            evidence_panel = mo.ui.table(
                pl.DataFrame(evidence_rows),
                page_size=10,
                show_column_summaries=False,
            )
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
            mo.ui.table(
                pl.DataFrame(metric_rows),
                page_size=10,
                show_column_summaries=False,
            )
            if metric_rows
            else mo.md("No manifest-backed plot metric definitions are available.")
        )

        change_rows = build_notebook_change_rows(selected_campaign_model)
        changes_table = (
            mo.ui.table(
                pl.DataFrame(change_rows),
                page_size=10,
                show_column_summaries=False,
            )
            if change_rows
            else mo.md("No round changes are available yet.")
        )
        changes_panel = mo.vstack(
            [
                mo.ui.table(
                    pl.DataFrame(build_notebook_change_summary_rows(selected_campaign_model)),
                    page_size=8,
                    show_column_summaries=False,
                ),
                changes_table,
            ]
        )

        artifact_rows = build_notebook_artifact_garden_rows(selected_campaign_model)
        artifact_summary_rows = build_notebook_artifact_garden_summary_rows(selected_campaign_model)
        artifact_rows_panel = (
            mo.ui.table(
                pl.DataFrame(artifact_rows),
                page_size=10,
                show_column_summaries=False,
            )
            if artifact_rows
            else mo.md("No artifact garden rows are available.")
        )
        artifact_garden_panel = mo.vstack(
            [
                mo.ui.table(
                    pl.DataFrame(artifact_summary_rows),
                    page_size=10,
                    show_column_summaries=False,
                ),
                artifact_rows_panel,
            ]
        )
    return artifact_garden_panel, changes_panel, metric_definitions_panel


@app.cell
def _(
    active_view_mode,
    artifact_garden_panel,
    campaign_summary_df,
    campaign_ui,
    collection_set_ui,
    changes_panel,
    evidence_panel,
    header_md,
    metric_definitions_panel,
    mo,
    plot_panel,
    selected_campaign_header_md,
    selected_overview_panel,
    selected_validity_md,
    view_mode_ui,
):
    _items = [header_md]
    _top_control_items = [view_mode_ui]
    if active_view_mode != "Campaign set" and campaign_ui is not None:
        _top_control_items.append(campaign_ui)
    elif collection_set_ui is not None:
        _top_control_items.append(collection_set_ui)
    _items.append(mo.vstack(_top_control_items, gap=0.20))
    if active_view_mode != "Campaign set":
        _items.append(selected_campaign_header_md)
    _accordion_items = {
        "OPAL campaigns at a glance": mo.ui.table(
            campaign_summary_df,
            page_size=12,
            show_column_summaries=False,
        ),
    }
    if active_view_mode != "Campaign set":
        _accordion_items.update(
            {
                "Selected OPAL campaign": selected_overview_panel,
                "Validity": selected_validity_md,
                "Changes": changes_panel,
                "Metric definitions": metric_definitions_panel,
                "Artifacts": artifact_garden_panel,
                "Warnings and stale artifacts": evidence_panel,
            }
        )
    _items.extend(
        [
            plot_panel,
            mo.accordion(
                _accordion_items,
                multiple=True,
                lazy=True,
            ),
        ]
    )
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
