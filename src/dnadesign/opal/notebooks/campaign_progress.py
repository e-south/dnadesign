import marimo

__generated_with = "0.19.4"

app = marimo.App(width="medium")


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
        build_notebook_collection_visual_description,
        build_notebook_evidence_rows,
        build_notebook_metric_definition_rows,
        build_notebook_plot_card_rows,
        build_notebook_plot_method_sections,
        build_notebook_plot_scope_options,
        build_notebook_validity_rows,
        build_notebook_visual_surface_model,
        find_notebook_repo_root,
        list_notebook_campaign_paths,
        render_notebook_plot_choice_image,
        render_notebook_reader_evidence_artifact_control,
        render_notebook_reader_evidence_artifact_visual,
        render_notebook_reader_evidence_panel,
        render_notebook_reader_evidence_plot_type_control,
        render_notebook_reader_evidence_time_control,
        render_notebook_visual_panel,
        select_notebook_plot_scope,
    )

    def opal_table(data, *, page_size):
        return mo.ui.table(data, page_size=page_size, show_column_summaries=False)

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
        build_notebook_collection_visual_description,
        build_notebook_evidence_rows,
        build_notebook_metric_definition_rows,
        build_notebook_plot_card_rows,
        build_notebook_plot_method_sections,
        build_notebook_plot_scope_options,
        build_notebook_validity_rows,
        build_notebook_visual_surface_model,
        find_notebook_repo_root,
        list_notebook_campaign_paths,
        mo,
        opal_table,
        pl,
        render_notebook_plot_choice_image,
        render_notebook_reader_evidence_artifact_control,
        render_notebook_reader_evidence_artifact_visual,
        render_notebook_reader_evidence_panel,
        render_notebook_reader_evidence_plot_type_control,
        render_notebook_reader_evidence_time_control,
        render_notebook_visual_panel,
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
    campaigns,
    mo,
    pl,
):
    campaign_rows = [build_notebook_campaign_summary_row(campaign_model) for campaign_model in campaigns]
    if campaign_rows:
        campaign_labels = [f"{index + 1}. {row['label']}" for index, row in enumerate(campaign_rows)]
        campaign_ui = mo.ui.dropdown(campaign_labels, value=campaign_labels[0], label="Campaign")
        campaign_summary_df = pl.DataFrame(campaign_rows)
    else:
        campaign_labels = []
        campaign_ui = None
        campaign_summary_df = pl.DataFrame([])
    header_md = mo.md("# OPAL Review Notebook")
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
        selected_campaign_brief_md = mo.md("")
        selected_overview_panel = mo.md("No campaign configs were found.")
        selected_validity_md = mo.md("")
    else:
        _header_lines = build_notebook_campaign_header_lines(selected_campaign_model, heading_level=2)
        selected_campaign_brief_md = mo.md(_header_lines[2] if len(_header_lines) > 2 else "")
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
    return selected_campaign_brief_md, selected_overview_panel, selected_validity_md


@app.cell
def _(
    build_notebook_visual_surface_model,
    selected_campaign_model,
):
    if selected_campaign_model is None:
        visual_surface_model = {"choices": [], "inventory_status_counts": {}, "stale_artifacts": []}
        campaign_plot_choices = []
    else:
        visual_surface_model = build_notebook_visual_surface_model(selected_campaign_model)
        campaign_plot_choices = visual_surface_model["choices"]
    return campaign_plot_choices


@app.cell
def _(build_notebook_collection_set_choices, collection_visuals):
    collection_set_choices = build_notebook_collection_set_choices(collection_visuals)
    return collection_set_choices


@app.cell
def _(collection_set_choices, mo):
    if collection_set_choices:
        view_mode_ui = mo.ui.dropdown(["Campaign", "Campaign set"], value="Campaign", label="View")
    else:
        view_mode_ui = None
    return (view_mode_ui,)


@app.cell
def _(view_mode_ui):
    active_view_mode = str(view_mode_ui.value) if view_mode_ui is not None else "Campaign"
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
                label="Collection plot",
                on_change=set_visual_label_memory,
            )
        else:
            plot_ui = mo.ui.dropdown(
                _labels,
                value=_preferred_visual_label,
                label="Plot deliverable",
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
    active_view_mode,
    build_notebook_collection_visual_card_rows,
    build_notebook_collection_visual_description,
    build_notebook_plot_card_rows,
    build_notebook_plot_method_sections,
    mo,
    opal_table,
    pl,
    plot_scope_ui,
    plot_ui,
    render_notebook_plot_choice_image,
    render_notebook_visual_panel,
    select_notebook_plot_scope,
    selected_visual_choice,
):
    plot_panel = render_notebook_visual_panel(
        active_view_mode=active_view_mode,
        build_notebook_collection_visual_card_rows=build_notebook_collection_visual_card_rows,
        build_notebook_plot_card_rows=build_notebook_plot_card_rows,
        build_notebook_plot_method_sections=build_notebook_plot_method_sections,
        collection_visual_description=build_notebook_collection_visual_description,
        mo=mo,
        opal_table=opal_table,
        pl=pl,
        plot_scope_ui=plot_scope_ui,
        plot_ui=plot_ui,
        render_notebook_plot_choice_image=render_notebook_plot_choice_image,
        selected_visual_choice=selected_visual_choice,
        select_notebook_plot_scope=select_notebook_plot_scope,
    )
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
            else mo.md("No plot metric definitions are available.")
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
def _(mo, pl, render_notebook_reader_evidence_panel, selected_campaign_model):
    if selected_campaign_model is None:
        reader_evidence_panel = mo.md("")
        reader_evidence_surface = {"rows": [], "artifact_rows": [], "media_rows": [], "media_plot_type_labels": []}
    else:

        def _opal_table(data, *, page_size):
            return mo.ui.table(data, page_size=page_size, show_column_summaries=False)

        _reader_evidence = render_notebook_reader_evidence_panel(
            selected_campaign_model,
            mo=mo,
            opal_table=_opal_table,
            pl=pl,
        )
        reader_evidence_panel = _reader_evidence["panel"]
        reader_evidence_surface = _reader_evidence["surface"]
    return reader_evidence_panel, reader_evidence_surface


@app.cell
def _(mo, reader_evidence_surface, render_notebook_reader_evidence_plot_type_control):
    reader_evidence_plot_type_ui = render_notebook_reader_evidence_plot_type_control(
        reader_evidence_surface,
        mo=mo,
    )
    return reader_evidence_plot_type_ui


@app.cell
def _(
    mo,
    reader_evidence_plot_type_ui,
    reader_evidence_surface,
    render_notebook_reader_evidence_artifact_control,
):
    selected_reader_evidence_plot_type_label = (
        str(reader_evidence_plot_type_ui.value) if reader_evidence_plot_type_ui is not None else None
    )
    reader_evidence_artifact_ui = render_notebook_reader_evidence_artifact_control(
        reader_evidence_surface,
        selected_plot_type_label=selected_reader_evidence_plot_type_label,
        mo=mo,
    )
    return reader_evidence_artifact_ui, selected_reader_evidence_plot_type_label


@app.cell
def _(
    mo,
    reader_evidence_artifact_ui,
    reader_evidence_surface,
    render_notebook_reader_evidence_time_control,
    selected_reader_evidence_plot_type_label,
):
    _selected_artifact_label = None if reader_evidence_artifact_ui is None else str(reader_evidence_artifact_ui.value)
    reader_evidence_time_ui = render_notebook_reader_evidence_time_control(
        reader_evidence_surface,
        selected_plot_type_label=selected_reader_evidence_plot_type_label,
        selected_artifact_label=_selected_artifact_label,
        mo=mo,
    )
    return reader_evidence_time_ui


@app.cell
def _(
    mo,
    reader_evidence_artifact_ui,
    reader_evidence_surface,
    reader_evidence_time_ui,
    render_notebook_reader_evidence_artifact_visual,
    selected_reader_evidence_plot_type_label,
):
    _selected_artifact_label = None if reader_evidence_artifact_ui is None else str(reader_evidence_artifact_ui.value)
    _selected_time_h = None
    if reader_evidence_time_ui is not None and hasattr(reader_evidence_time_ui, "value"):
        _selected_time_h = float(reader_evidence_time_ui.value)
    reader_evidence_visual = render_notebook_reader_evidence_artifact_visual(
        reader_evidence_surface,
        selected_plot_type_label=selected_reader_evidence_plot_type_label,
        selected_artifact_label=_selected_artifact_label,
        mo=mo,
        selected_time_h=_selected_time_h,
    )
    return reader_evidence_visual


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
    reader_evidence_artifact_ui,
    reader_evidence_panel,
    reader_evidence_plot_type_ui,
    reader_evidence_time_ui,
    reader_evidence_visual,
    selected_campaign_brief_md,
    selected_visual_choice,
    selected_overview_panel,
    selected_validity_md,
    view_mode_ui,
):
    _items = [header_md]
    if active_view_mode != "Campaign set" and campaign_ui is not None:
        _top_control_items = [campaign_ui]
        if view_mode_ui is not None:
            _top_control_items.append(view_mode_ui)
    elif collection_set_ui is not None:
        _top_control_items = [view_mode_ui, collection_set_ui] if view_mode_ui is not None else [collection_set_ui]
    else:
        _top_control_items = [item for item in [view_mode_ui] if item is not None]
    if _top_control_items:
        _items.append(mo.hstack(_top_control_items, justify="start", align="end", wrap=True, gap=0.35))
    if active_view_mode != "Campaign set":
        _items.append(selected_campaign_brief_md)
    _reader_plot_panel = None
    if reader_evidence_plot_type_ui is not None:
        _reader_controls = [reader_evidence_plot_type_ui]
        if reader_evidence_artifact_ui is not None:
            _reader_controls.append(reader_evidence_artifact_ui)
        if reader_evidence_time_ui is not None:
            _reader_controls.append(reader_evidence_time_ui)
        _reader_plot_panel = mo.vstack(
            [
                mo.hstack(_reader_controls, justify="start", align="end", wrap=True, gap=0.35),
                reader_evidence_visual,
            ],
            gap=0.35,
        )
    _plot_items = []
    if selected_visual_choice is not None or _reader_plot_panel is None:
        _plot_items.append(plot_panel)
    if _reader_plot_panel is not None:
        _plot_items.append(_reader_plot_panel)
    _items.append(mo.vstack(_plot_items, gap=0.55))
    _accordion_items = {
        "Campaigns at a glance": mo.ui.table(
            campaign_summary_df,
            page_size=12,
            show_column_summaries=False,
        ),
    }
    if active_view_mode != "Campaign set":
        _status_panel = mo.vstack(
            [selected_overview_panel, selected_validity_md, changes_panel, evidence_panel], gap=0.35
        )
        _data_panel = mo.vstack([reader_evidence_panel, metric_definitions_panel, artifact_garden_panel], gap=0.35)
        _accordion_items.update(
            {
                "Campaign status": _status_panel,
                "Data and evidence records": _data_panel,
            }
        )
    _items.extend(
        [
            mo.accordion(
                _accordion_items,
                multiple=True,
            ),
        ]
    )
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
