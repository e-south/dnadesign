# ABOUTME: Marimo notebook for promoter OPAL dashboard exploration.
# ABOUTME: Provides interactive exploration of campaign datasets and scoring.
import marimo

__generated_with = "0.19.4"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _():
    from pathlib import Path
    from textwrap import dedent

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from dnadesign.opal.src.analysis.dashboard.theme import setup_altair_theme, with_title

    setup_altair_theme()

    return Path, alt, dedent, mo, np, pl, with_title


@app.cell(hide_code=True)
def _():
    # Notebook polish summary:
    # - Performance: fast list->numpy conversion for vector columns.
    # - Legends: theme-level legend/title colors for visibility on white.
    # - Plots: consistent titles/subtitles and RF/cluster UX improvements.
    # - Clusters: stable numeric ordering for distribution views.
    from dnadesign.opal.src.analysis.dashboard import datasets as dash_datasets
    from dnadesign.opal.src.analysis.dashboard import diagnostics as dash_diagnostics
    from dnadesign.opal.src.analysis.dashboard import hues as dash_hues
    from dnadesign.opal.src.analysis.dashboard import labels as dash_labels
    from dnadesign.opal.src.analysis.dashboard import models as dash_models
    from dnadesign.opal.src.analysis.dashboard import scores as dash_scores
    from dnadesign.opal.src.analysis.dashboard import selection as dash_selection
    from dnadesign.opal.src.analysis.dashboard import transient as dash_transient
    from dnadesign.opal.src.analysis.dashboard import ui as dash_ui
    from dnadesign.opal.src.analysis.dashboard import util as dash_util
    from dnadesign.opal.src.analysis.dashboard.charts import diagnostics_guidance as dash_diag_guidance
    from dnadesign.opal.src.analysis.dashboard.charts import plots as dash_plots
    from dnadesign.opal.src.analysis.dashboard.views import sfxi as dash_sfxi
    from dnadesign.opal.src.analysis.sfxi import gates as sfxi_gates
    from dnadesign.opal.src.analysis.sfxi import support as sfxi_support
    from dnadesign.opal.src.analysis.sfxi import uncertainty as sfxi_uncertainty
    from dnadesign.opal.src.objectives import sfxi_math

    build_mode_view = dash_scores.build_mode_view
    build_cluster_chart = dash_plots.build_cluster_chart
    build_umap_explorer_chart = dash_plots.build_umap_explorer_chart
    build_label_events = dash_labels.build_label_events
    build_pred_events = dash_labels.build_pred_events
    build_round_options_from_label_hist = dash_labels.build_round_options_from_label_hist
    build_label_sfxi_view = dash_sfxi.build_label_sfxi_view
    build_pred_sfxi_view = dash_sfxi.build_pred_sfxi_view
    build_umap_controls = dash_ui.build_umap_controls
    build_diagnostics_panels = dash_diag_guidance.build_diagnostics_panels
    campaign_label_from_path = dash_datasets.campaign_label_from_path
    build_explorer_hue_registry = dash_hues.build_explorer_hue_registry
    build_hue_registry = dash_hues.build_hue_registry
    build_sfxi_hue_registry = dash_hues.build_sfxi_hue_registry
    default_view_hues = dash_hues.default_view_hues
    compute_sfxi_params = dash_sfxi.compute_sfxi_params
    build_score_histogram = dash_transient.build_score_histogram
    build_feature_importance_chart = dash_transient.build_feature_importance_chart
    compute_transient_overlay = dash_transient.compute_transient_overlay
    coerce_selection_dataframe = dash_selection.coerce_selection_dataframe
    Diagnostics = dash_diagnostics.Diagnostics
    diagnostics_to_lines = dash_diagnostics.diagnostics_to_lines
    attach_namespace_columns = dash_util.attach_namespace_columns
    choose_axis_defaults = dash_util.choose_axis_defaults
    choose_dropdown_value = dash_util.choose_dropdown_value
    list_series_to_numpy = dash_util.list_series_to_numpy
    state_value_changed = dash_util.state_value_changed
    dedupe_exprs = dash_util.dedupe_exprs
    dedup_latest_labels = dash_labels.dedup_latest_labels
    find_repo_root = dash_datasets.find_repo_root
    is_altair_undefined = dash_util.is_altair_undefined
    list_campaign_paths = dash_datasets.list_campaign_paths
    load_campaign_selection = dash_datasets.load_campaign_selection
    load_round_ctx_from_dir = dash_models.load_round_ctx_from_dir
    resolve_artifact_state = dash_models.resolve_artifact_state
    get_feature_importances = dash_models.get_feature_importances
    load_feature_importances_from_artifact = dash_models.load_feature_importances_from_artifact
    missingness_summary = dash_util.missingness_summary
    namespace_summary = dash_util.namespace_summary
    resolve_brush_selection = dash_selection.resolve_brush_selection
    resolve_sfxi_readiness = dash_sfxi.resolve_sfxi_readiness
    safe_is_numeric = dash_util.safe_is_numeric
    return (
        Diagnostics,
        attach_namespace_columns,
        build_cluster_chart,
        build_diagnostics_panels,
        build_explorer_hue_registry,
        build_feature_importance_chart,
        build_hue_registry,
        build_sfxi_hue_registry,
        build_label_events,
        build_label_sfxi_view,
        build_pred_sfxi_view,
        build_mode_view,
        build_pred_events,
        build_round_options_from_label_hist,
        build_score_histogram,
        build_umap_controls,
        build_umap_explorer_chart,
        campaign_label_from_path,
        choose_axis_defaults,
        choose_dropdown_value,
        coerce_selection_dataframe,
        compute_sfxi_params,
        compute_transient_overlay,
        dedup_latest_labels,
        dedupe_exprs,
        default_view_hues,
        diagnostics_to_lines,
        find_repo_root,
        get_feature_importances,
        list_series_to_numpy,
        load_feature_importances_from_artifact,
        load_round_ctx_from_dir,
        is_altair_undefined,
        list_campaign_paths,
        load_campaign_selection,
        missingness_summary,
        namespace_summary,
        resolve_artifact_state,
        resolve_brush_selection,
        resolve_sfxi_readiness,
        safe_is_numeric,
        state_value_changed,
        sfxi_gates,
        sfxi_math,
        sfxi_support,
        sfxi_uncertainty,
    )


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
    ## Promoter Dashboard for Exploratory Data Analysis

    `records.parquet` is the record store for `DenseGen`-generated promoter designs,
    including TFBS annotations, Evo2-derived logits, `OPAL` artifacts, etc. This dashboard helps you
    explore and validate those records, defaulting to a dataset of 50K 60 bp bacterial promoters containing
    binding sites for CpxR and LexA.

    Given a collection of transcription factor binding sites, `DenseGen` generates dense arrays + TFBS
    annotations + `densegen__visual`. `Infer` then generates sequence logits (which come from passing the dense
    arrays through the 7B-param genomic language model Evo2). After pooling these logits in the sequence
    dimension (resulting in a [1,512] vector per sequence), we can calculate Leiden clusters on them with
    `Cluster` + generate UMAP coordinates.

    #### Open question

    Among all these dense array-derived promoters, which ones (when cloned into *E. coli*) exhibit a targeted
    condition-dependent expression specification? I.e., if we define that we want an AND/OR gate based on the
    presence of either EtOH or ciprofloxacin, can we navigate this sequence space to find hits? This is where
    active learning, the `SFXI` metric, and `OPAL` come in.

    Suggested workflow
    1. Select an OPAL campaign (configs/campaign.yaml).
    2. Explore the dataset schema + missingness for that campaign’s records.parquet.
    3. Inspect a record: view its sequence and TFBS placement.
    4. Explore UMAP neighborhoods and brush select a working subset.
    5. Switch between Canonical vs Overlay scoring; tweak SFXI params in Overlay mode to explore sensitivity.
    6. (Optional) Export a tabular subset for downstream analysis.
    """
        )
    )
    return


@app.cell
def _(Path, find_repo_root):
    notebook_path = Path(__file__).resolve()
    repo_root = find_repo_root(notebook_path)
    return (repo_root,)


@app.cell
def _(campaign_label_from_path, list_campaign_paths, repo_root):
    campaign_paths = list_campaign_paths(repo_root)
    campaign_labels = [campaign_label_from_path(path, repo_root) for path in campaign_paths]
    campaign_path_map = dict(zip(campaign_labels, campaign_paths))
    default_campaign_label = None
    for label in campaign_labels:
        if "opal/campaigns/demo" in label.replace("\\", "/"):
            default_campaign_label = label
            break
    if default_campaign_label is None and campaign_labels:
        default_campaign_label = campaign_labels[0]
    if not campaign_labels:
        campaign_labels = ["(no campaigns found)"]
        default_campaign_label = campaign_labels[0]
    return campaign_labels, campaign_path_map, default_campaign_label


@app.cell
def _(mo):
    dataset_override_input = mo.ui.text(
        value="",
        label="Advanced override (records.parquet path, optional)",
        full_width=True,
        placeholder="Absolute or repo-root-relative path",
    )
    return (dataset_override_input,)


@app.cell
def _(
    campaign_path_map,
    diagnostics_to_lines,
    load_campaign_selection,
    mo,
    opal_panel_prefix_default,
    opal_panel_prefix_dropdown,
    repo_root,
):
    campaign_label = opal_panel_prefix_dropdown.value if opal_panel_prefix_dropdown is not None else None
    if campaign_label is None:
        campaign_label = opal_panel_prefix_default
    campaign_path = campaign_path_map.get(campaign_label) if campaign_label else None
    campaign_selection = load_campaign_selection(campaign_path=campaign_path, repo_root=repo_root)
    selection_lines = diagnostics_to_lines(campaign_selection.diagnostics)
    selection_notice_md = mo.md("\n".join(selection_lines)) if selection_lines else mo.md("")
    return campaign_selection, selection_notice_md


@app.cell
def _(campaign_selection):
    dashboard_defaults = {}
    if campaign_selection.info is not None and campaign_selection.info.dashboard:
        explorer_defaults = campaign_selection.info.dashboard.get("explorer_defaults")
        if isinstance(explorer_defaults, dict):
            dashboard_defaults = dict(explorer_defaults)
    return (dashboard_defaults,)


@app.cell
def _(Path, campaign_selection, dataset_override_input, repo_root):
    override = dataset_override_input.value.strip()
    dataset_mode = "campaign"
    dataset_path = campaign_selection.records_path
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_absolute() and repo_root is not None:
            override_path = (repo_root / override_path).resolve()
        dataset_path = override_path
        dataset_mode = "override"
    return dataset_mode, dataset_path


@app.cell
def _(mo):
    load_button = None
    return (load_button,)


@app.cell
def _(campaign_selection, dataset_mode, dataset_path, load_button, mo, pl):
    _unused = load_button
    df_raw = pl.DataFrame()
    dataset_name = None
    status_lines = []
    if campaign_selection.label:
        status_lines.append(f"Campaign: `{campaign_selection.label}`")
    if campaign_selection.info is not None:
        status_lines.append(f"Slug: `{campaign_selection.info.slug}`")
    if campaign_selection.workdir is not None:
        status_lines.append(f"Workdir: `{campaign_selection.workdir}`")
    if dataset_path is None:
        status_lines.append("Records path: **missing**")
    else:
        status_lines.append(f"Records path: `{dataset_path}`")
        status_lines.append(f"Mode: `{dataset_mode}`")
    if dataset_path is not None and dataset_path.exists():
        try:
            df_raw = pl.read_parquet(dataset_path)
            status_lines.append(f"Rows: `{df_raw.height}`")
            status_lines.append(f"Columns: `{len(df_raw.columns)}`")
            dataset_name = dataset_path.parent.name if dataset_path.name == "records.parquet" else dataset_path.stem
        except Exception as exc:
            status_lines.append(f"Error: failed to read parquet ({exc})")
    elif dataset_path is not None:
        status_lines.append("Error: records.parquet not found.")
    dataset_status_md = mo.md("\n".join(status_lines)) if status_lines else mo.md("")
    return dataset_name, dataset_status_md, df_raw


@app.cell
def _(df_raw):
    if "__row_id" in df_raw.columns:
        df_prelim = df_raw
    else:
        df_prelim = df_raw.with_row_index("__row_id")
    if "id" in df_prelim.columns:
        df_prelim = df_prelim.with_columns(df_prelim["id"].cast(str))
    return (df_prelim,)


@app.cell
def _(df_prelim, mo):
    required_cols = ["id", "cluster__ldn_v1__umap_x", "cluster__ldn_v1__umap_y"]
    missing = [col for col in required_cols if col not in df_prelim.columns]
    data_ready = True
    data_ready_note_md = mo.md("")
    if missing:
        data_ready = False
        data_ready_note_md = mo.md(
            "\n".join(
                [
                    "**Campaign/dataset incompatible**",
                    "Missing required columns:",
                    ", ".join(f"`{col}`" for col in missing),
                ]
            )
        )
    return data_ready, data_ready_note_md


@app.cell
def _(df_prelim, mo, namespace_summary):
    namespace_summary_df = namespace_summary(df_prelim.columns)
    namespace_table = mo.ui.table(namespace_summary_df)
    return (namespace_table,)


@app.cell
def _(df_prelim, missingness_summary, mo):
    missingness_df = missingness_summary(df_prelim)
    missingness_table = mo.ui.table(missingness_df, page_size=5)
    return (missingness_table,)


@app.cell
def _(df_prelim, mo):
    dataset_table = mo.ui.table(df_prelim, page_size=5)
    return (dataset_table,)


@app.cell
def _(mo):
    dataset_explorer_plot_type_state, set_dataset_explorer_plot_type_state = mo.state(None)
    dataset_explorer_x_state, set_dataset_explorer_x_state = mo.state(None)
    dataset_explorer_y_state, set_dataset_explorer_y_state = mo.state(None)
    dataset_explorer_color_state, set_dataset_explorer_color_state = mo.state(None)
    return (
        dataset_explorer_color_state,
        dataset_explorer_plot_type_state,
        dataset_explorer_x_state,
        dataset_explorer_y_state,
        set_dataset_explorer_color_state,
        set_dataset_explorer_plot_type_state,
        set_dataset_explorer_x_state,
        set_dataset_explorer_y_state,
    )


@app.cell
def _(
    build_explorer_hue_registry,
    choose_axis_defaults,
    choose_dropdown_value,
    data_ready,
    dashboard_defaults,
    default_view_hues,
    dataset_explorer_color_state,
    dataset_explorer_plot_type_state,
    dataset_explorer_x_state,
    dataset_explorer_y_state,
    df_view,
    mo,
    safe_is_numeric,
):
    df_explorer = df_view if data_ready else df_view.head(0)
    numeric_cols = [name for name, dtype in df_explorer.schema.items() if safe_is_numeric(dtype)]
    defaults = dashboard_defaults or {}
    _default_x = defaults.get("x")
    _default_y = defaults.get("y")
    _preferred_x = "opal__view__score"
    _preferred_y = "opal__view__effect_scaled"
    _x_default, _y_default = choose_axis_defaults(
        numeric_cols=numeric_cols,
        default_x=_default_x,
        default_y=_default_y,
        preferred_x=_preferred_x,
        preferred_y=_preferred_y,
    )

    plot_type_options = ["scatter", "histogram"]
    plot_type_default = (
        choose_dropdown_value(plot_type_options, current=dataset_explorer_plot_type_state(), preferred="scatter")
        or "scatter"
    )
    plot_type_dropdown = mo.ui.dropdown(
        options=plot_type_options,
        value=plot_type_default,
        label="Plot type",
    )
    x_options = numeric_cols or ["(none)"]
    x_default = (
        choose_dropdown_value(x_options, current=dataset_explorer_x_state(), preferred=_x_default) or x_options[0]
    )
    x_dropdown = mo.ui.dropdown(
        options=x_options,
        value=x_default,
        label="X column",
        full_width=True,
    )
    y_options = numeric_cols or ["(none)"]
    y_default = (
        choose_dropdown_value(y_options, current=dataset_explorer_y_state(), preferred=_y_default) or y_options[0]
    )
    y_dropdown = mo.ui.dropdown(
        options=y_options,
        value=y_default,
        label="Y column (scatter only)",
        full_width=True,
    )
    hue_registry = build_explorer_hue_registry(
        df_explorer,
        preferred=default_view_hues(),
        include_columns=True,
        denylist={"__row_id", "id", "id_", "id__"},
    )
    color_options = ["(none)"] + hue_registry.labels()
    _default_color = defaults.get("color")
    if _default_color in color_options:
        preferred_color = _default_color
    elif "Score" in hue_registry.labels():
        preferred_color = "Score"
    else:
        preferred_color = "(none)"
    color_default = (
        choose_dropdown_value(
            color_options,
            current=dataset_explorer_color_state(),
            preferred=preferred_color,
        )
        or "(none)"
    )
    dataset_explorer_color_dropdown = mo.ui.dropdown(
        options=color_options,
        value=color_default,
        label="Color by (scatter only)",
        full_width=True,
    )
    bins_slider = mo.ui.slider(5, 200, value=30, label="Histogram bins")
    return (
        bins_slider,
        dataset_explorer_color_dropdown,
        hue_registry,
        plot_type_dropdown,
        x_dropdown,
        y_dropdown,
    )


@app.cell
def _(
    alt,
    bins_slider,
    dataset_explorer_color_dropdown,
    dataset_name,
    dedupe_exprs,
    df_view,
    hue_registry,
    mo,
    pl,
    plot_type_dropdown,
    safe_is_numeric,
    with_title,
    x_dropdown,
    y_dropdown,
):
    plot_type = plot_type_dropdown.value
    _x_col = x_dropdown.value
    _y_col = y_dropdown.value
    _note_lines: list[str] = []
    if not hasattr(alt, "_DNAD_PLOT_SIZE"):
        alt._DNAD_PLOT_SIZE = 420
    _plot_size = alt._DNAD_PLOT_SIZE
    _df_explorer_source = df_view
    _id_col = "id" if "id" in _df_explorer_source.columns else "__row_id"

    _dataset_label = dataset_name or "dataset"
    _color_label = dataset_explorer_color_dropdown.value
    _hue = None if _color_label == "(none)" else hue_registry.get(_color_label)
    _color_encoding = alt.Undefined
    _color_title = None
    _color_tooltip = None
    _size_encoding = alt.Undefined
    _order_encoding = alt.Undefined
    _okabe_ito = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ]
    _fallback_scheme = "tableau20"

    if plot_type == "scatter":
        if (
            _x_col not in _df_explorer_source.columns
            or _y_col not in _df_explorer_source.columns
            or not (
                safe_is_numeric(_df_explorer_source.schema[_x_col])
                and safe_is_numeric(_df_explorer_source.schema[_y_col])
            )
        ):
            _note_lines.append("Select numeric X and Y columns for scatter.")
            df_plot = pl.DataFrame(
                schema={
                    "__row_id": pl.Int64,
                    "id": pl.Utf8,
                    "x": pl.Float64,
                    "y": pl.Float64,
                }
            )
            chart = (
                alt.Chart(df_plot)
                .mark_circle(opacity=0.7, stroke=None, strokeWidth=0)
                .encode(
                    x=alt.X("x"),
                    y=alt.Y("y"),
                    tooltip=[c for c in ["id", "__row_id", "x", "y"] if c in df_plot.columns],
                )
            )
        else:
            if _y_col == _x_col:
                y_col_plot = f"{_y_col}__y"
                _select_exprs = [pl.col("__row_id")]
                if _id_col != "__row_id":
                    _select_exprs.append(pl.col(_id_col))
                _select_exprs.extend([pl.col(_x_col), pl.col(_y_col).alias(y_col_plot)])
            else:
                y_col_plot = _y_col
                _select_exprs = [pl.col("__row_id")]
                if _id_col != "__row_id":
                    _select_exprs.append(pl.col(_id_col))
                _select_exprs.extend([pl.col(_x_col), pl.col(_y_col)])
            if _hue is not None and _hue.key in _df_explorer_source.columns and _hue.key not in {_x_col, _y_col}:
                _select_exprs.append(pl.col(_hue.key))
            df_plot = _df_explorer_source.select(dedupe_exprs(_select_exprs))

            if _hue is not None and _hue.key in df_plot.columns:
                _non_null_count = df_plot.select(pl.col(_hue.key).count()).item() if df_plot.height else 0
                if _non_null_count == 0:
                    _note_lines.append(f"Color `{_hue.key}` has no non-null values; rendering without color.")
                elif _hue.kind == "categorical" and _hue.category_labels:
                    _label_col = f"{_hue.key}__label"
                    _layer_col = f"{_hue.key}__layer"
                    _yes_label, _no_label = _hue.category_labels
                    _bool_mask = pl.col(_hue.key).fill_null(False)
                    df_plot = df_plot.with_columns(
                        pl.when(_bool_mask).then(pl.lit(_yes_label)).otherwise(pl.lit(_no_label)).alias(_label_col),
                        pl.when(_bool_mask).then(pl.lit(1)).otherwise(pl.lit(0)).alias(_layer_col),
                    )
                    _color_title = _hue.label
                    _color_tooltip = _label_col
                    if "observed" in _hue.key:
                        _highlight_color = _okabe_ito[5]
                    elif "top_k" in _hue.key:
                        _highlight_color = _okabe_ito[2]
                    else:
                        _highlight_color = _okabe_ito[0]
                    _color_scale = alt.Scale(domain=[_yes_label, _no_label], range=[_highlight_color, "#B0B0B0"])
                    _color_encoding = alt.Color(
                        f"{_label_col}:N",
                        title=_color_title,
                        scale=_color_scale,
                        legend=alt.Legend(title=_color_title),
                    )
                    _size_encoding = alt.Size(
                        f"{_layer_col}:Q",
                        legend=None,
                        scale=alt.Scale(domain=[0, 1], range=[30, 60]),
                    )
                    _order_encoding = alt.Order(f"{_layer_col}:Q", sort="ascending")
                elif _hue.kind == "numeric":
                    _color_title = _hue.label
                    _color_tooltip = _hue.key
                    _color_encoding = alt.Color(
                        f"{_hue.key}:Q",
                        title=_color_title,
                        legend=alt.Legend(title=_color_title, format=".2f", tickCount=5),
                    )
                else:
                    _n_unique = df_plot.select(pl.col(_hue.key).n_unique()).item() if df_plot.height else 0
                    _color_scale = (
                        alt.Scale(range=_okabe_ito)
                        if _n_unique <= len(_okabe_ito)
                        else alt.Scale(scheme=_fallback_scheme)
                    )
                    _color_title = _hue.label
                    _color_tooltip = _hue.key
                    _color_encoding = alt.Color(
                        f"{_hue.key}:N",
                        title=_color_title,
                        scale=_color_scale,
                        legend=alt.Legend(title=_color_title),
                    )

            _brush = alt.selection_interval(name="explorer_brush", encodings=["x", "y"])
            _tooltip_cols = [c for c in [_id_col, "__row_id", _x_col, y_col_plot] if c in df_plot.columns]
            _tooltip_cols = list(dict.fromkeys(_tooltip_cols))
            if _color_tooltip and _color_tooltip in df_plot.columns and _color_tooltip not in _tooltip_cols:
                _tooltip_cols.append(_color_tooltip)
            chart = (
                alt.Chart(df_plot)
                .mark_circle(opacity=0.7, stroke=None, strokeWidth=0)
                .encode(
                    x=alt.X(_x_col, title=_x_col),
                    y=alt.Y(y_col_plot, title=_y_col),
                    color=_color_encoding,
                    size=_size_encoding,
                    order=_order_encoding,
                    tooltip=_tooltip_cols,
                )
                .add_params(_brush)
            )
    else:
        if _x_col not in _df_explorer_source.columns or not safe_is_numeric(_df_explorer_source.schema[_x_col]):
            _note_lines.append("Select a numeric column for the histogram.")
            df_plot = pl.DataFrame(
                schema={
                    "__row_id": pl.Int64,
                    "id": pl.Utf8,
                    "x": pl.Float64,
                }
            )
            chart = alt.Chart(df_plot).mark_bar().encode(x=alt.X("x", bin=alt.Bin(maxbins=10)), y=alt.Y("count()"))
        else:
            _select_exprs = [pl.col("__row_id")]
            if _id_col != "__row_id":
                _select_exprs.append(pl.col(_id_col))
            _select_exprs.append(pl.col(_x_col))
            df_plot = _df_explorer_source.select(dedupe_exprs(_select_exprs))
            chart = (
                alt.Chart(df_plot)
                .mark_bar()
                .encode(
                    x=alt.X(
                        _x_col,
                        bin=alt.Bin(maxbins=int(bins_slider.value)),
                        title=_x_col,
                    ),
                    y=alt.Y("count()", title="count"),
                )
            )

    _rows_shown = int(df_plot.height)
    if plot_type == "scatter":
        title = "Dataset explorer scatter"
        subtitle = f"{_dataset_label} · n={_rows_shown}"
    else:
        title = "Dataset explorer histogram"
        subtitle = f"{_dataset_label} · n={_rows_shown}"
    chart = with_title(chart, title, subtitle).properties(width=_plot_size, height=_plot_size)
    dataset_explorer_chart_ui = mo.ui.altair_chart(chart)
    note_md = mo.md("\n".join(_note_lines)) if _note_lines else mo.md("")
    controls_row = mo.hstack([x_dropdown, y_dropdown, dataset_explorer_color_dropdown])
    bins_row = bins_slider if plot_type == "histogram" else mo.md("")
    dataset_explorer = mo.vstack(
        [
            plot_type_dropdown,
            controls_row,
            bins_row,
            note_md,
            dataset_explorer_chart_ui,
        ]
    )
    return dataset_explorer, dataset_explorer_chart_ui


@app.cell
def _(
    dataset_explorer_color_dropdown,
    dataset_explorer_color_state,
    dataset_explorer_plot_type_state,
    dataset_explorer_x_state,
    dataset_explorer_y_state,
    plot_type_dropdown,
    set_dataset_explorer_color_state,
    set_dataset_explorer_plot_type_state,
    set_dataset_explorer_x_state,
    set_dataset_explorer_y_state,
    state_value_changed,
    x_dropdown,
    y_dropdown,
):
    if plot_type_dropdown is not None:
        _next_value = plot_type_dropdown.value
        if state_value_changed(dataset_explorer_plot_type_state(), _next_value):
            set_dataset_explorer_plot_type_state(_next_value)
    if x_dropdown is not None:
        _next_value = x_dropdown.value
        if state_value_changed(dataset_explorer_x_state(), _next_value):
            set_dataset_explorer_x_state(_next_value)
    if y_dropdown is not None:
        _next_value = y_dropdown.value
        if state_value_changed(dataset_explorer_y_state(), _next_value):
            set_dataset_explorer_y_state(_next_value)
    if dataset_explorer_color_dropdown is not None:
        _next_value = dataset_explorer_color_dropdown.value
        if state_value_changed(dataset_explorer_color_state(), _next_value):
            set_dataset_explorer_color_state(_next_value)
    return ()


@app.cell
def _(
    coerce_selection_dataframe,
    data_ready,
    dataset_explorer_chart_ui,
    df_view,
    is_altair_undefined,
    mo,
    pl,
    x_dropdown,
    y_dropdown,
):
    _unused = (x_dropdown, y_dropdown)
    _df_explorer_source = df_view if data_ready else df_view.head(0)
    _selected_raw = dataset_explorer_chart_ui.value

    if _selected_raw is None or is_altair_undefined(_selected_raw):
        df_explorer_selected = _df_explorer_source.head(0)
        explorer_selection_note_md = mo.md("No points selected.")
    else:
        _selected_df = coerce_selection_dataframe(_selected_raw)

        if _selected_df is None or "__row_id" not in _selected_df.columns:
            df_explorer_selected = _df_explorer_source.head(0)
            explorer_selection_note_md = mo.md("Selection missing row ids.")
        else:
            _selected_ids = _selected_df.select(pl.col("__row_id").drop_nulls().unique()).to_series().to_list()
            if not _selected_ids:
                df_explorer_selected = _df_explorer_source.head(0)
                explorer_selection_note_md = mo.md("No points selected.")
            else:
                df_explorer_selected = _df_explorer_source.filter(pl.col("__row_id").is_in(_selected_ids))
                explorer_selection_note_md = mo.md(f"Selected rows: `{df_explorer_selected.height}`")

    explorer_selected_table = mo.ui.table(df_explorer_selected)
    return explorer_selected_table, explorer_selection_note_md


@app.cell
def _(data_ready, df_prelim):
    df_active = df_prelim if data_ready else df_prelim.head(0)
    return (df_active,)


@app.cell
def _(df_active, mo):
    df_active_note_md = mo.md(f"Active rows: `{df_active.height}`. Summaries reflect the loaded dataset.")
    return (df_active_note_md,)


@app.cell
def _(mo):
    cluster_metric_state, set_cluster_metric_state = mo.state(None)
    cluster_hue_state, set_cluster_hue_state = mo.state(None)
    return (
        cluster_hue_state,
        cluster_metric_state,
        set_cluster_hue_state,
        set_cluster_metric_state,
    )


@app.cell
def _(choose_dropdown_value, cluster_hue_state, cluster_metric_state, hue_registry, mo):
    metric_options = [opt.label for opt in hue_registry.options if opt.kind == "numeric"]
    if not metric_options:
        metric_options = ["(none)"]
    default_metric = "Score" if "Score" in metric_options else metric_options[0]
    metric_default = (
        choose_dropdown_value(
            metric_options,
            current=cluster_metric_state(),
            preferred=default_metric,
        )
        or metric_options[0]
    )
    transient_cluster_metric_dropdown = mo.ui.dropdown(
        options=metric_options,
        value=metric_default,
        label="Cluster plot metric",
        full_width=True,
    )
    hue_options = ["(none)"] + hue_registry.labels()
    default_hue = "(none)"
    for candidate in ["Top-K", "Observed (labeled)"]:
        if candidate in hue_registry.labels():
            default_hue = candidate
            break
    hue_default = (
        choose_dropdown_value(
            hue_options,
            current=cluster_hue_state(),
            preferred=default_hue,
        )
        or "(none)"
    )
    transient_cluster_hue_dropdown = mo.ui.dropdown(
        options=hue_options,
        value=hue_default,
        label="Color by (hue)",
        full_width=True,
    )
    return transient_cluster_hue_dropdown, transient_cluster_metric_dropdown


@app.cell
def _(
    cluster_hue_state,
    cluster_metric_state,
    set_cluster_hue_state,
    set_cluster_metric_state,
    state_value_changed,
    transient_cluster_hue_dropdown,
    transient_cluster_metric_dropdown,
):
    if transient_cluster_metric_dropdown is not None:
        _next_value = transient_cluster_metric_dropdown.value
        if state_value_changed(cluster_metric_state(), _next_value):
            set_cluster_metric_state(_next_value)
    if transient_cluster_hue_dropdown is not None:
        _next_value = transient_cluster_hue_dropdown.value
        if state_value_changed(cluster_hue_state(), _next_value):
            set_cluster_hue_state(_next_value)
    return ()


@app.cell(column=1)
def _(
    data_ready_note_md,
    dataset_override_input,
    dataset_status_md,
    mo,
    opal_panel_prefix_dropdown,
    selection_notice_md,
):
    header = mo.md("## Campaign & dataset")
    intro = mo.md(
        "Select an OPAL campaign. The dashboard resolves `records.parquet` from the campaign config and "
        "uses it as the single source of truth."
    )
    mo.vstack(
        [
            header,
            intro,
            opal_panel_prefix_dropdown,
            selection_notice_md,
            dataset_override_input,
            dataset_status_md,
            data_ready_note_md,
        ]
    )
    return


@app.cell
def _(missingness_table, mo, namespace_table):
    mo.vstack(
        [
            mo.md(
                "**Column namespaces**  \n"
                "A quick provenance map: prefixes before `__` indicate which tool or pipeline "
                "contributed each field."
            ),
            namespace_table,
            mo.md("**Missing data**  \nScan for gaps before plotting or modeling. Emptiest columns appear first."),
            missingness_table,
        ]
    )
    return


@app.cell
def _(mo):
    umap_color_state, set_umap_color_state = mo.state(None)
    return set_umap_color_state, umap_color_state


@app.cell
def _(build_umap_controls, df_view, hue_registry, mo, umap_color_state):
    controls = build_umap_controls(
        mo=mo,
        df_active=df_view,
        hue_registry=hue_registry,
        default_hue_key="opal__view__score",
        current_color=umap_color_state(),
    )
    umap_color_dropdown = controls.umap_color_dropdown
    umap_x_input = controls.umap_x_input
    umap_y_input = controls.umap_y_input
    return umap_color_dropdown, umap_x_input, umap_y_input


@app.cell
def _(alt, pl):
    _unused = (pl,)
    if not hasattr(alt, "_DNAD_PLOT_SIZE"):
        alt._DNAD_PLOT_SIZE = 420
    return


@app.cell
def _(mo):
    from io import BytesIO

    def fig_to_image(fig, *, dpi: int = 150):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=int(dpi), bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        return mo.image(buf.getvalue())

    return (fig_to_image,)


@app.cell
def _(
    alt,
    build_umap_explorer_chart,
    dataset_name,
    df_view,
    hue_registry,
    mo,
    umap_color_dropdown,
    umap_x_input,
    umap_y_input,
):
    _x_col = umap_x_input.value.strip()
    _y_col = umap_y_input.value.strip()
    if not hasattr(alt, "_DNAD_PLOT_SIZE"):
        alt._DNAD_PLOT_SIZE = 420
    _plot_size = alt._DNAD_PLOT_SIZE

    _color_label = umap_color_dropdown.value
    _hue = None
    if _color_label != "(none)":
        _hue = next((opt for opt in hue_registry.options if opt.key == _color_label), None)

    _result = build_umap_explorer_chart(
        df=df_view,
        x_col=_x_col,
        y_col=_y_col,
        hue=_hue,
        point_size=20,
        opacity=0.55,
        plot_size=_plot_size,
        dataset_name=dataset_name,
    )
    df_umap_plot = _result.df_plot
    umap_valid = _result.valid
    umap_chart_note_md = mo.md(_result.note) if _result.note else mo.md("")
    umap_chart_ui = mo.ui.altair_chart(_result.chart)
    return df_umap_plot, umap_chart_note_md, umap_chart_ui, umap_valid


@app.cell
def _(set_umap_color_state, state_value_changed, umap_color_dropdown, umap_color_state):
    if umap_color_dropdown is not None:
        _next_value = umap_color_dropdown.value
        if state_value_changed(umap_color_state(), _next_value):
            set_umap_color_state(_next_value)
    return ()


@app.cell
def _(
    df_umap_plot,
    mo,
    resolve_brush_selection,
    umap_chart_ui,
    umap_valid,
    umap_x_input,
    umap_y_input,
):
    _unused = (umap_x_input, umap_y_input)
    _selected_raw = umap_chart_ui.value
    df_umap_selected, selection_note = resolve_brush_selection(
        df_plot=df_umap_plot,
        selected_raw=_selected_raw,
        selection_enabled=bool(umap_valid),
        id_col="__row_id",
    )
    umap_selection_note_md = mo.md(selection_note)

    umap_selected_explorer = mo.ui.table(df_umap_selected)
    return df_umap_selected, umap_selected_explorer, umap_selection_note_md


@app.cell(column=2)
def _(
    dataset_explorer,
    dataset_table,
    df_active_note_md,
    explorer_selected_table,
    explorer_selection_note_md,
    mo,
):
    mo.vstack(
        [
            mo.md("## Exploratory data analysis"),
            mo.md(
                "Inspect the table, then pivot into targeted plots. Selections here "
                "propagate to other panels as a shared record pool."
            ),
            dataset_table,
            mo.md("### Ad-hoc plotting using a Polars-backed chart explorer."),
            dataset_explorer,
            mo.md("#### Selected rows (from explorer brush)"),
            explorer_selection_note_md,
            explorer_selected_table,
            df_active_note_md,
        ]
    )
    return


@app.cell(column=3)
def _(
    baserender_error_md,
    id_dropdown,
    id_nav_buttons,
    id_override_input,
    inspect_pool_dropdown,
    inspect_view_mode_dropdown,
    mo,
    render_element,
    save_pdf_button,
    save_pdf_path_input,
    save_pdf_status_md,
    save_pdf_target_md,
    sequence_md,
    summary_md,
    tfbs_note_md,
    tfbs_table_ui,
):
    mo.vstack(
        [
            mo.md("## Inspect a dense array-derived promoter"),
            mo.md("View TFBS placements/orientations. Use the pool selectors to constrain which IDs are available."),
            inspect_pool_dropdown,
            id_dropdown,
            id_override_input,
            summary_md,
            sequence_md,
            inspect_view_mode_dropdown,
            id_nav_buttons,
            baserender_error_md,
            render_element,
            tfbs_note_md,
            tfbs_table_ui,
            save_pdf_path_input,
            save_pdf_target_md,
            save_pdf_button,
            save_pdf_status_md,
        ]
    )
    return


@app.cell
def _(mo):
    export_source_label_map = {
        "Full dataset (all rows)": "df_active",
        "UMAP brush selection": "df_umap_selected",
        "SFXI scored labels (current view)": "df_sfxi_selected",
        "Selected score Top-K": "df_score_top_k_pool",
        "Overlay Top-K": "df_overlay_top_k_pool",
        "Current inspected record (single row)": "active_record",
    }
    export_source_dropdown = mo.ui.dropdown(
        options=list(export_source_label_map.keys()),
        value="Full dataset (all rows)",
        label="Export source (tabular data)",
        full_width=True,
    )
    export_format_dropdown = mo.ui.dropdown(
        options=["csv", "parquet"],
        value="parquet",
        label="Format",
    )
    export_button = mo.ui.run_button(label="Export")
    return (
        export_button,
        export_format_dropdown,
        export_source_dropdown,
        export_source_label_map,
    )


@app.cell
def _(
    active_record,
    dataset_name,
    df_active,
    df_overlay_top_k_pool,
    df_pred_selected,
    df_score_top_k_pool,
    df_sfxi_selected,
    df_umap_selected,
    export_button,
    export_format_dropdown,
    export_source_dropdown,
    export_source_label_map,
    mo,
    mode_dropdown,
    opal_campaign_info,
    opal_pred_selected_round,
    opal_pred_selected_run_id,
    overlay_result,
    pl,
    repo_root,
    sfxi_params,
):
    export_status_md = None
    export_note_md = mo.md("")

    if export_button.value:
        source_label = export_source_dropdown.value
        source = export_source_label_map.get(source_label, source_label)
        if source == "df_umap_selected":
            df_export = df_umap_selected
        elif source == "df_sfxi_selected":
            df_export = df_sfxi_selected
        elif source == "df_score_top_k_pool":
            df_export = df_score_top_k_pool
        elif source == "df_overlay_top_k_pool":
            df_export = df_overlay_top_k_pool
        elif source == "active_record":
            df_export = active_record if active_record is not None else df_active.head(0)
        else:
            df_export = df_active

        if df_export is None or df_export.is_empty():
            export_status_md = mo.md("Nothing to export.")
        else:
            from datetime import datetime

            _export_out_dir = (
                repo_root / "src" / "dnadesign" / "opal" / "notebooks" / "_outputs" / "promoter_eda_exports"
            )
            _export_out_dir.mkdir(parents=True, exist_ok=True)
            suffix = export_format_dropdown.value

            def _slugify(value: str | None) -> str:
                import re

                if not value:
                    return "dataset"
                slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
                return slug or "dataset"

            dataset_slug = _slugify(dataset_name)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            overlay_ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            out_path = _export_out_dir / f"promoter_eda_{dataset_slug}_{ts}.{suffix}"
            can_write = True

            def _prefix_objective_columns(df_export: pl.DataFrame) -> pl.DataFrame:
                mapping = {
                    "score": "obj__score",
                    "logic_fidelity": "obj__logic_fidelity",
                    "effect_scaled": "obj__effect_scaled",
                    "effect_raw": "obj__effect_raw",
                }
                rename_map = {
                    src: dst
                    for src, dst in mapping.items()
                    if src in df_export.columns and dst not in df_export.columns
                }
                if rename_map:
                    df_export = df_export.rename(rename_map)
                drop_cols = [
                    src for src, dst in mapping.items() if src in df_export.columns and dst in df_export.columns
                ]
                if drop_cols:
                    df_export = df_export.drop(drop_cols)
                return df_export

            df_export = _prefix_objective_columns(df_export)
            if mode_dropdown.value == "Overlay":
                overlay_ready = True
                if (
                    overlay_result is None
                    or overlay_result.df_pred_scored is None
                    or overlay_result.df_pred_scored.is_empty()
                ):
                    export_status_md = mo.md(
                        "Overlay export unavailable: no overlay scores computed for the selected round/run."
                    )
                    overlay_ready = False
                if overlay_ready and (
                    df_pred_selected is None
                    or df_pred_selected.is_empty()
                    or "pred_y_hat" not in df_pred_selected.columns
                ):
                    export_status_md = mo.md(
                        "Overlay export unavailable: stored prediction vectors (y_hat) missing for the selected "
                        "round/run."
                    )
                    overlay_ready = False
                if not overlay_ready:
                    can_write = False

                if overlay_ready:
                    objective_name = opal_campaign_info.objective_name if opal_campaign_info is not None else "sfxi_v1"
                    overlay_metrics = None
                    if overlay_result is not None and overlay_result.df_pred_scored is not None:
                        overlay_metrics = overlay_result.df_pred_scored.select(
                            [
                                col
                                for col in [
                                    "id",
                                    "opal__overlay__score",
                                    "opal__overlay__logic_fidelity",
                                    "opal__overlay__effect_scaled",
                                    "opal__overlay__rank",
                                    "opal__overlay__top_k",
                                ]
                                if col in overlay_result.df_pred_scored.columns
                            ]
                        )
                    df_export_overlay = df_export
                    if overlay_metrics is not None and not overlay_metrics.is_empty() and "id" in df_export.columns:
                        df_export_overlay = df_export_overlay.join(overlay_metrics, on="id", how="left")
                    else:
                        for _col in [
                            "opal__overlay__score",
                            "opal__overlay__logic_fidelity",
                            "opal__overlay__effect_scaled",
                            "opal__overlay__rank",
                            "opal__overlay__top_k",
                        ]:
                            if _col not in df_export_overlay.columns:
                                df_export_overlay = df_export_overlay.with_columns(pl.lit(None).alias(_col))
                    df_export_overlay = df_export_overlay.join(
                        df_pred_selected.select(["id", "pred_y_hat"]), on="id", how="left"
                    )
                    df_export_overlay = df_export_overlay.with_columns(
                        [
                            pl.col("opal__overlay__rank").cast(pl.Int64, strict=False),
                            pl.col("opal__overlay__top_k").cast(pl.Boolean, strict=False),
                        ]
                    )
                    for _col in [
                        "opal__overlay__score",
                        "opal__overlay__logic_fidelity",
                        "opal__overlay__effect_scaled",
                        "opal__overlay__rank",
                        "opal__overlay__top_k",
                        "pred_y_hat",
                    ]:
                        if _col not in df_export_overlay.columns:
                            df_export_overlay = df_export_overlay.with_columns(pl.lit(None).alias(_col))

                    def _coerce_y_hat(value):
                        if value is None:
                            return None
                        try:
                            return [float(v) for v in value]
                        except Exception:
                            return None

                    df_export_overlay = df_export_overlay.with_columns(
                        pl.col("pred_y_hat")
                        .map_elements(_coerce_y_hat, return_dtype=pl.List(pl.Float64))
                        .alias("pred_y_hat")
                    )
                    objective_params_struct = pl.struct(
                        [
                            pl.lit(list(sfxi_params.setpoint)).alias("setpoint"),
                            pl.lit(list(sfxi_params.weights)).alias("weights"),
                            pl.lit(float(sfxi_params.d)).alias("d"),
                            pl.lit(float(sfxi_params.beta)).alias("beta"),
                            pl.lit(float(sfxi_params.gamma)).alias("gamma"),
                            pl.lit(float(sfxi_params.delta)).alias("delta"),
                            pl.lit(float(sfxi_params.p)).alias("p"),
                            pl.lit(float(sfxi_params.fallback_p)).alias("fallback_p"),
                            pl.lit(int(sfxi_params.min_n)).alias("min_n"),
                            pl.lit(float(sfxi_params.eps)).alias("eps"),
                        ]
                    ).alias("params")
                    objective_struct = pl.struct(
                        [
                            pl.lit(objective_name).alias("name"),
                            objective_params_struct,
                        ]
                    ).alias("objective")
                    metrics_struct = pl.struct(
                        [
                            pl.col("opal__overlay__score").alias("score"),
                            pl.col("opal__overlay__logic_fidelity").alias("logic_fidelity"),
                            pl.col("opal__overlay__effect_scaled").alias("effect_scaled"),
                        ]
                    ).alias("metrics")
                    selection_struct = pl.struct(
                        [
                            pl.col("opal__overlay__rank").alias("rank"),
                            pl.col("opal__overlay__top_k").alias("top_k"),
                        ]
                    ).alias("selection")
                    entry_struct = pl.struct(
                        [
                            pl.lit("pred").alias("kind"),
                            pl.lit("dashboard_overlay").alias("src"),
                            pl.lit(opal_campaign_info.slug if opal_campaign_info is not None else None).alias(
                                "campaign_slug"
                            ),
                            pl.lit(opal_pred_selected_round).alias("as_of_round"),
                            pl.lit(opal_pred_selected_run_id).alias("run_id"),
                            pl.lit(overlay_ts).alias("ts"),
                            pl.col("pred_y_hat").alias("y_hat"),
                            objective_struct,
                            metrics_struct,
                            selection_struct,
                        ]
                    )
                    df_export = df_export_overlay.with_columns(
                        pl.concat_list(entry_struct).alias("opal_dashboard__label_hist")
                    )
                    export_note_md = mo.md("Added `opal_dashboard__label_hist` with dashboard overlay provenance.")

            if can_write:
                if suffix == "csv":
                    df_export.write_csv(out_path)
                else:
                    df_export.write_parquet(out_path)

                status_line = (
                    f"Saved `{out_path}` from `{source_label}` "
                    f"({df_export.height} rows × {len(df_export.columns)} cols)."
                )
                export_status_md = mo.md(status_line)
    return export_note_md, export_status_md


@app.cell(column=4)
def _(
    mo,
    umap_chart_note_md,
    umap_chart_ui,
    umap_color_dropdown,
    umap_selected_explorer,
    umap_selection_note_md,
    umap_x_input,
    umap_y_input,
):
    mo.vstack(
        [
            mo.md("## UMAP explorer (Evo2 logits mean-pooled in the sequence dimension)"),
            mo.md(
                "Evo2 outputs are pooled across sequence length into a fixed-width vector [1,512]. Running a UMAP then "
                "projects these high-dimensional sequence representations into 2D. Color by metadata or "
                "model outputs and brush to define a working subset."
            ),
            mo.md("See `src/dnadesign/infer/README.md`."),
            mo.hstack([umap_x_input, umap_y_input, umap_color_dropdown]),
            umap_chart_note_md,
            umap_chart_ui,
            umap_selection_note_md,
            umap_selected_explorer,
        ]
    )
    return


@app.cell
def _(campaign_labels, default_campaign_label, mo):
    opal_panel_prefix_dropdown = mo.ui.dropdown(
        options=campaign_labels,
        value=default_campaign_label,
        label="Campaign config",
        full_width=True,
    )
    opal_panel_prefix_default = default_campaign_label
    return opal_panel_prefix_default, opal_panel_prefix_dropdown


@app.cell
def _(build_label_events, campaign_selection, df_active, mo):
    opal_campaign_info = None
    opal_label_events_df = df_active.head(0)
    opal_label_diag = None
    opal_label_hist_col = None
    opal_campaign_path = None
    opal_workdir = None
    opal_campaign_notice_md = mo.md("")

    opal_campaign_info = campaign_selection.info if campaign_selection is not None else None
    opal_campaign_path = campaign_selection.path if campaign_selection is not None else None
    opal_workdir = campaign_selection.workdir if campaign_selection is not None else None
    if opal_campaign_info is None:
        opal_campaign_notice_md = mo.md("Select a compatible campaign to enable OPAL views.")
    else:
        opal_label_hist_col = f"opal__{opal_campaign_info.slug}__label_hist"
        _label_events = build_label_events(
            df=df_active,
            label_hist_col=opal_label_hist_col,
            y_col_name=opal_campaign_info.y_column,
            campaign_slug=opal_campaign_info.slug,
            source_kind="records",
        )
        opal_label_events_df = _label_events.df
        opal_label_diag = _label_events.diag
    return (
        opal_campaign_info,
        opal_campaign_notice_md,
        opal_campaign_path,
        opal_label_diag,
        opal_label_events_df,
        opal_label_hist_col,
        opal_workdir,
    )


@app.cell
def _(diagnostics_to_lines, mo, opal_label_diag):
    label_notice_md = mo.md("")
    if opal_label_diag is not None and getattr(opal_label_diag, "diagnostics", None) is not None:
        lines = diagnostics_to_lines(opal_label_diag.diagnostics)
        if opal_label_diag.message:
            lines.insert(0, str(opal_label_diag.message))
        if lines:
            label_notice_md = mo.md("\n".join(lines))
    return (label_notice_md,)


@app.cell
def _(
    build_round_options_from_label_hist,
    opal_label_events_df,
    opal_pred_events_df,
):
    round_options = build_round_options_from_label_hist(
        label_events_df=opal_label_events_df,
        pred_events_df=opal_pred_events_df,
    )
    return (round_options,)


@app.cell
def _(
    build_pred_events,
    df_active,
    diagnostics_to_lines,
    mo,
    opal_label_hist_col,
):
    opal_pred_events_df = df_active.head(0)
    opal_pred_diag = None
    pred_diag_md = mo.md("")
    if opal_label_hist_col:
        _pred_events = build_pred_events(
            df=df_active,
            label_hist_col=opal_label_hist_col,
        )
        opal_pred_events_df = _pred_events.df
        opal_pred_diag = _pred_events.diag
        _lines = []
        if opal_pred_diag is not None and getattr(opal_pred_diag, "message", None):
            _lines.append(str(opal_pred_diag.message))
        if opal_pred_diag is not None and getattr(opal_pred_diag, "diagnostics", None) is not None:
            _lines.extend(diagnostics_to_lines(opal_pred_diag.diagnostics))
        pred_diag_md = mo.md("\n".join(_lines)) if _lines else mo.md("")
    return opal_pred_events_df, pred_diag_md


@app.cell
def _(diagnostics_to_lines, mo, round_options):
    opal_round_dropdown = None
    opal_round_value_map = {}
    rounds = sorted({int(r) for r in (round_options.rounds or [])})
    if rounds:
        opal_round_value_map = {f"R={_r}": _r for _r in rounds}
        _default_label = f"R={rounds[-1]}"
        opal_round_dropdown = mo.ui.dropdown(
            options=list(opal_round_value_map.keys()),
            value=_default_label,
            label="As-of round",
            full_width=True,
        )
    else:
        _empty_label = "(no rounds available)"
        opal_round_dropdown = mo.ui.dropdown(
            options=[_empty_label],
            value=_empty_label,
            label="As-of round",
            full_width=True,
        )
    round_notice_lines = diagnostics_to_lines(round_options.diagnostics)
    round_notice_md = mo.md("\n".join(round_notice_lines)) if round_notice_lines else mo.md("")
    return opal_round_dropdown, opal_round_value_map, round_notice_md


@app.cell
def _(opal_round_dropdown, opal_round_value_map):
    opal_pred_selected_round = None
    if opal_round_dropdown is not None and opal_round_value_map:
        opal_pred_selected_round = opal_round_value_map.get(opal_round_dropdown.value)
    return (opal_pred_selected_round,)


@app.cell
def _(mo, opal_pred_selected_round, round_options):
    opal_run_id_dropdown = None
    opal_pred_run_ids = []
    opal_run_id_prompt = None
    if opal_pred_selected_round is not None and round_options is not None:
        opal_pred_run_ids = round_options.run_ids_by_round.get(int(opal_pred_selected_round), [])
    opal_pred_run_ids = sorted({str(r) for r in opal_pred_run_ids if r is not None})
    if len(opal_pred_run_ids) > 1:
        prompt = "(select run_id)"
        opal_run_id_prompt = prompt
        opal_run_id_dropdown = mo.ui.dropdown(
            options=[prompt] + opal_pred_run_ids,
            value=prompt,
            label="Run ID",
            full_width=True,
        )
    return opal_pred_run_ids, opal_run_id_dropdown, opal_run_id_prompt


@app.cell
def _(opal_pred_run_ids, opal_run_id_dropdown, opal_run_id_prompt):
    opal_pred_selected_run_id = None
    if opal_run_id_dropdown is not None:
        selected = opal_run_id_dropdown.value
        opal_pred_selected_run_id = None if selected == opal_run_id_prompt else selected
    elif len(opal_pred_run_ids) == 1:
        opal_pred_selected_run_id = opal_pred_run_ids[0]
    return (opal_pred_selected_run_id,)


@app.cell
def _(mo):
    mode_dropdown = mo.ui.dropdown(
        options=["Canonical", "Overlay"],
        value="Canonical",
        label="Mode",
        full_width=True,
    )
    return (mode_dropdown,)


@app.cell
def _(mo):
    sfxi_source_dropdown = mo.ui.dropdown(
        options=["Observed", "Predicted", "Both"],
        value="Observed",
        label="SFXI source",
        full_width=True,
    )
    return (sfxi_source_dropdown,)


@app.cell
def _(
    mo,
    opal_pred_events_df,
    opal_pred_run_ids,
    opal_pred_selected_round,
    opal_pred_selected_run_id,
    pl,
):
    pred_notice_md = mo.md("")
    df_pred_selected = opal_pred_events_df.head(0)
    if opal_pred_selected_round is None:
        pred_notice_md = mo.md("Select a round to view stored predictions.")
    else:
        df_pred_selected = opal_pred_events_df.filter(pl.col("as_of_round") == int(opal_pred_selected_round))
        if opal_pred_selected_run_id:
            df_pred_selected = df_pred_selected.filter(pl.col("run_id") == str(opal_pred_selected_run_id))
        elif opal_pred_run_ids and len(opal_pred_run_ids) > 1:
            pred_notice_md = mo.md("Multiple run_ids exist for this round; select one to avoid mixing reruns.")
            df_pred_selected = opal_pred_events_df.head(0)
        if df_pred_selected.is_empty():
            pred_notice_md = mo.md("No stored predictions found for the selected round/run.")
    return df_pred_selected, pred_notice_md


@app.cell
def _(mo, opal_label_events_df, pl):
    label_src_multiselect = None
    label_src_values = []
    if (
        opal_label_events_df is not None
        and not opal_label_events_df.is_empty()
        and "label_src" in opal_label_events_df.columns
    ):
        label_src_values = opal_label_events_df.select(pl.col("label_src").drop_nulls().unique()).to_series().to_list()
    label_src_values = sorted({str(x) for x in label_src_values if x is not None})
    if label_src_values:
        label_src_multiselect = mo.ui.multiselect(
            options=label_src_values,
            value=label_src_values,
            label="Label sources (filter; default ALL)",
            full_width=True,
        )
    return (label_src_multiselect,)


@app.cell
def _(mo, opal_pred_selected_round):
    opal_selected_round = opal_pred_selected_round
    view_options = [
        "all_events",
        "as_of_round (dedup latest per id)",
        "current_round_only",
    ]
    view_default = "as_of_round (dedup latest per id)" if opal_selected_round is not None else "all_events"
    opal_label_view_dropdown = mo.ui.dropdown(
        options=view_options,
        value=view_default,
        label="Label view",
        full_width=True,
    )
    return opal_label_view_dropdown, opal_selected_round


@app.cell
def _(opal_label_view_dropdown):
    label_view_choice = opal_label_view_dropdown.value if opal_label_view_dropdown is not None else "all_events"
    return (label_view_choice,)


@app.cell
def _(
    dedup_latest_labels,
    label_src_multiselect,
    label_view_choice,
    mo,
    opal_campaign_info,
    opal_label_events_df,
    opal_selected_round,
    pl,
):
    opal_labels_view_df = opal_label_events_df if opal_label_events_df is not None else pl.DataFrame()
    df_labels_filtered = opal_labels_view_df
    if (
        df_labels_filtered is not None
        and not df_labels_filtered.is_empty()
        and label_src_multiselect is not None
        and "label_src" in df_labels_filtered.columns
    ):
        selected_sources = label_src_multiselect.value
        if selected_sources:
            df_labels_filtered = df_labels_filtered.filter(pl.col("label_src").is_in(selected_sources))

    opal_labels_current_df = df_labels_filtered
    opal_labels_asof_df = df_labels_filtered
    opal_round_notice_md = mo.md("")

    if opal_campaign_info is None or df_labels_filtered is None or df_labels_filtered.is_empty():
        pass
    else:
        df_round_scope = df_labels_filtered
        if opal_selected_round is not None and "observed_round" in df_labels_filtered.columns:
            df_round_scope = df_labels_filtered.filter(pl.col("observed_round") <= int(opal_selected_round))
            opal_labels_current_df = df_labels_filtered.filter(pl.col("observed_round") == int(opal_selected_round))
        else:
            opal_labels_current_df = df_round_scope

        cumulative = bool(opal_campaign_info.training_policy.get("cumulative_training", True))
        opal_labels_asof_df = df_round_scope if cumulative else opal_labels_current_df
        if opal_selected_round is not None:
            policy = str(
                opal_campaign_info.training_policy.get(
                    "label_cross_round_deduplication_policy",
                    "latest_only",
                )
            )
            if policy == "latest_only":
                opal_labels_asof_df = dedup_latest_labels(
                    opal_labels_asof_df,
                    id_col="id",
                    round_col="observed_round",
                )

        view_choice = label_view_choice
        if view_choice.startswith("as_of_round"):
            if opal_selected_round is None:
                opal_round_notice_md = mo.md("Select a round to apply the as_of_round view; showing all events.")
                opal_labels_view_df = df_round_scope
            else:
                opal_labels_view_df = opal_labels_asof_df
        elif view_choice == "current_round_only":
            if opal_selected_round is None:
                opal_round_notice_md = mo.md("Select a round to apply the current_round_only view; showing all events.")
                opal_labels_view_df = df_round_scope
            else:
                opal_labels_view_df = opal_labels_current_df
        else:
            opal_labels_view_df = df_round_scope

        sort_cols = [col for col in ["id", "observed_round", "label_ts"] if col in opal_labels_view_df.columns]
        if sort_cols:
            opal_labels_view_df = opal_labels_view_df.sort(sort_cols)

    return (
        opal_labels_asof_df,
        opal_labels_current_df,
        opal_labels_view_df,
        opal_round_notice_md,
    )


@app.cell
def _(
    mo,
    opal_campaign_info,
    opal_campaign_path,
    opal_pred_selected_round,
    opal_pred_selected_run_id,
    opal_workdir,
):
    summary_lines = ["### Provenance (summary)"]
    if opal_campaign_info is not None:
        summary_lines.append(f"- Campaign: `{opal_campaign_info.slug}`")
    if opal_pred_selected_round is not None:
        summary_lines.append(f"- As-of round: `R={opal_pred_selected_round}`")
    if opal_pred_selected_run_id is not None:
        summary_lines.append(f"- Run ID: `{opal_pred_selected_run_id}`")
    _summary_md = mo.md("\n".join(summary_lines))

    details_lines = ["### Provenance details"]
    if opal_campaign_path is not None:
        details_lines.append(f"- Config: `{opal_campaign_path}`")
    if opal_workdir is not None:
        details_lines.append(f"- Workdir: `{opal_workdir}`")
    if opal_pred_selected_round is not None:
        details_lines.append(f"- As-of round: `R={opal_pred_selected_round}`")
    if opal_pred_selected_run_id is not None:
        details_lines.append(f"- Run ID: `{opal_pred_selected_run_id}`")

    details_md = mo.md("\n".join(details_lines))
    opal_provenance_md = mo.ui.tabs({"Summary": _summary_md, "Details": details_md})
    return (opal_provenance_md,)


@app.cell
def _(mo, opal_campaign_info):
    objective_params = opal_campaign_info.objective_params if opal_campaign_info is not None else {}
    _setpoint_vec = objective_params.get("setpoint_vector") or [0.0, 0.0, 0.0, 1.0]
    if len(_setpoint_vec) != 4:
        _setpoint_vec = [0.0, 0.0, 0.0, 1.0]
    p00, p10, p01, p11 = (float(x) for x in _setpoint_vec)
    beta_default = float(objective_params.get("logic_exponent_beta", 1.0))
    gamma_default = float(objective_params.get("intensity_exponent_gamma", 1.0))
    delta_default = float(objective_params.get("intensity_log2_offset_delta", 0.0))
    exponent_options = [0.25, 0.5, 1.0, 2.0, 4.0]
    beta_value = beta_default if beta_default in exponent_options else 1.0
    gamma_value = gamma_default if gamma_default in exponent_options else 1.0
    scaling = dict(objective_params.get("scaling") or {})
    p_default = float(scaling.get("percentile", 95.0))
    min_n_default = int(scaling.get("min_n", 5))
    eps_default = float(scaling.get("eps", 1.0e-8))
    fallback_default = float(scaling.get("fallback_percentile", p_default))
    sfxi_default_values = {
        "p00": p00,
        "p10": p10,
        "p01": p01,
        "p11": p11,
        "beta": beta_value,
        "gamma": gamma_value,
    }

    sfxi_p00_slider = mo.ui.slider(0.0, 1.0, value=p00, label="p00", step=0.01)
    sfxi_p10_slider = mo.ui.slider(0.0, 1.0, value=p10, label="p10", step=0.01)
    sfxi_p01_slider = mo.ui.slider(0.0, 1.0, value=p01, label="p01", step=0.01)
    sfxi_p11_slider = mo.ui.slider(0.0, 1.0, value=p11, label="p11", step=0.01)
    sfxi_beta_input = mo.ui.dropdown(options=exponent_options, value=beta_value, label="beta")
    sfxi_gamma_input = mo.ui.dropdown(options=exponent_options, value=gamma_value, label="gamma")
    sfxi_fixed_params = {
        "delta": delta_default,
        "percentile": p_default,
        "fallback_percentile": fallback_default,
        "min_n": min_n_default,
        "eps": eps_default,
    }
    return (
        sfxi_beta_input,
        sfxi_default_values,
        sfxi_fixed_params,
        sfxi_gamma_input,
        sfxi_p00_slider,
        sfxi_p01_slider,
        sfxi_p10_slider,
        sfxi_p11_slider,
    )


@app.cell
def _(mo, sfxi_params):
    _p00, _p10, _p01, _p11 = (float(v) for v in sfxi_params.setpoint)
    _setpoint_text = ", ".join([f"{v:.3f}" for v in (_p00, _p10, _p01, _p11)])
    sfxi_setpoint_md = mo.md(f"Current setpoint vector (state order: 00, 10, 01, 11): `p = [{_setpoint_text}]`")
    return (sfxi_setpoint_md,)


@app.cell
def _(mo):
    sfxi_color_state, set_sfxi_color_state = mo.state(None)
    return sfxi_color_state, set_sfxi_color_state


@app.cell
def _(build_sfxi_hue_registry, choose_dropdown_value, df_sfxi_scatter, mo, sfxi_color_state):
    sfxi_hue_registry = build_sfxi_hue_registry(
        df_sfxi_scatter,
        preferred=[],
        include_columns=True,
        denylist={"__row_id", "id", "id_"},
    )
    hue_labels = sfxi_hue_registry.labels()
    if not hue_labels:
        options = ["score"] if "score" in df_sfxi_scatter.columns else ["(none)"]
    else:
        options = hue_labels
    preferred = "score" if "score" in options else None
    default = choose_dropdown_value(options, current=sfxi_color_state(), preferred=preferred) or options[0]
    sfxi_color_dropdown = mo.ui.dropdown(
        options=options,
        value=default,
        label="Color by",
        full_width=True,
    )
    return sfxi_color_dropdown, sfxi_hue_registry


@app.cell
def _(mo):
    support_y_state, set_support_y_state = mo.state(None)
    support_color_state, set_support_color_state = mo.state(None)
    uncertainty_color_state, set_uncertainty_color_state = mo.state(None)
    uncertainty_kind_state, set_uncertainty_kind_state = mo.state(None)
    uncertainty_components_state, set_uncertainty_components_state = mo.state(None)
    uncertainty_reduction_state, set_uncertainty_reduction_state = mo.state(None)
    return (
        set_support_color_state,
        set_support_y_state,
        set_uncertainty_color_state,
        set_uncertainty_components_state,
        set_uncertainty_kind_state,
        set_uncertainty_reduction_state,
        support_color_state,
        support_y_state,
        uncertainty_color_state,
        uncertainty_components_state,
        uncertainty_kind_state,
        uncertainty_reduction_state,
    )


@app.cell
def _(
    choose_dropdown_value,
    mo,
    support_y_state,
    uncertainty_components_state,
    uncertainty_kind_state,
    uncertainty_reduction_state,
):
    diagnostics_sample_slider = mo.ui.slider(0, 5000, value=1500, step=100, label="Diagnostics sample (n)")
    diagnostics_seed_slider = mo.ui.slider(0, 10000, value=0, step=1, label="Diagnostics seed")
    support_y_options = ["Score", "Logic fidelity"]
    support_y_default = (
        choose_dropdown_value(support_y_options, current=support_y_state(), preferred="Score") or "Score"
    )
    support_y_dropdown = mo.ui.dropdown(
        options=support_y_options,
        value=support_y_default,
        label="Support Y-axis",
        full_width=True,
    )
    uncertainty_kind_options = ["score", "y_hat"]
    uncertainty_kind_default = (
        choose_dropdown_value(uncertainty_kind_options, current=uncertainty_kind_state(), preferred="score") or "score"
    )
    uncertainty_kind_dropdown = mo.ui.dropdown(
        options=uncertainty_kind_options,
        value=uncertainty_kind_default,
        label="Uncertainty kind",
        full_width=True,
    )
    uncertainty_components_options = ["all", "logic", "intensity"]
    uncertainty_components_default = (
        choose_dropdown_value(
            uncertainty_components_options,
            current=uncertainty_components_state(),
            preferred="all",
        )
        or "all"
    )
    uncertainty_components_dropdown = mo.ui.dropdown(
        options=uncertainty_components_options,
        value=uncertainty_components_default,
        label="Y-hat components",
        full_width=True,
    )
    uncertainty_reduction_options = ["mean", "max"]
    uncertainty_reduction_default = (
        choose_dropdown_value(
            uncertainty_reduction_options,
            current=uncertainty_reduction_state(),
            preferred="mean",
        )
        or "mean"
    )
    uncertainty_reduction_dropdown = mo.ui.dropdown(
        options=uncertainty_reduction_options,
        value=uncertainty_reduction_default,
        label="Y-hat reduction",
        full_width=True,
    )
    return (
        diagnostics_sample_slider,
        diagnostics_seed_slider,
        support_y_dropdown,
        uncertainty_components_dropdown,
        uncertainty_kind_dropdown,
        uncertainty_reduction_dropdown,
    )


@app.cell
def _(choose_dropdown_value, hue_registry, mo, support_color_state, uncertainty_color_state):
    _diag_hue_options = ["(none)"] + hue_registry.labels()
    support_color_preferred = "Effect scaled" if "Effect scaled" in _diag_hue_options else "Score"
    support_color_default = (
        choose_dropdown_value(_diag_hue_options, current=support_color_state(), preferred=support_color_preferred)
        or "(none)"
    )
    support_color_dropdown = mo.ui.dropdown(
        options=_diag_hue_options,
        value=support_color_default,
        label="Support color",
        full_width=True,
    )
    uncertainty_color_preferred = "Logic fidelity" if "Logic fidelity" in _diag_hue_options else "Score"
    uncertainty_color_default = (
        choose_dropdown_value(
            _diag_hue_options, current=uncertainty_color_state(), preferred=uncertainty_color_preferred
        )
        or "(none)"
    )
    uncertainty_color_dropdown = mo.ui.dropdown(
        options=_diag_hue_options,
        value=uncertainty_color_default,
        label="Uncertainty color",
        full_width=True,
    )
    return support_color_dropdown, uncertainty_color_dropdown


@app.cell
def _(
    set_support_color_state,
    set_support_y_state,
    set_uncertainty_color_state,
    set_uncertainty_components_state,
    set_uncertainty_kind_state,
    set_uncertainty_reduction_state,
    state_value_changed,
    support_color_dropdown,
    support_color_state,
    support_y_dropdown,
    support_y_state,
    uncertainty_color_dropdown,
    uncertainty_color_state,
    uncertainty_components_dropdown,
    uncertainty_components_state,
    uncertainty_kind_dropdown,
    uncertainty_kind_state,
    uncertainty_reduction_dropdown,
    uncertainty_reduction_state,
):
    if support_y_dropdown is not None:
        _next_value = support_y_dropdown.value
        if state_value_changed(support_y_state(), _next_value):
            set_support_y_state(_next_value)
    if support_color_dropdown is not None:
        _next_value = support_color_dropdown.value
        if state_value_changed(support_color_state(), _next_value):
            set_support_color_state(_next_value)
    if uncertainty_color_dropdown is not None:
        _next_value = uncertainty_color_dropdown.value
        if state_value_changed(uncertainty_color_state(), _next_value):
            set_uncertainty_color_state(_next_value)
    if uncertainty_kind_dropdown is not None:
        _next_value = uncertainty_kind_dropdown.value
        if state_value_changed(uncertainty_kind_state(), _next_value):
            set_uncertainty_kind_state(_next_value)
    if uncertainty_components_dropdown is not None:
        _next_value = uncertainty_components_dropdown.value
        if state_value_changed(uncertainty_components_state(), _next_value):
            set_uncertainty_components_state(_next_value)
    if uncertainty_reduction_dropdown is not None:
        _next_value = uncertainty_reduction_dropdown.value
        if state_value_changed(uncertainty_reduction_state(), _next_value):
            set_uncertainty_reduction_state(_next_value)
    return ()


@app.cell
def _(set_sfxi_color_state, sfxi_color_dropdown, sfxi_color_state, state_value_changed):
    if sfxi_color_dropdown is not None:
        _next_value = sfxi_color_dropdown.value
        if state_value_changed(sfxi_color_state(), _next_value):
            set_sfxi_color_state(_next_value)
    return ()


@app.cell
def _(
    alt,
    dataset_name,
    df_sfxi_scatter,
    mo,
    opal_campaign_info,
    pl,
    sfxi_color_dropdown,
    sfxi_hue_registry,
    with_title,
):
    if not hasattr(alt, "_DNAD_PLOT_SIZE"):
        alt._DNAD_PLOT_SIZE = 420
    _plot_size = alt._DNAD_PLOT_SIZE
    _sfxi_explain_text = "No SFXI data to plot."
    _slug = opal_campaign_info.slug if opal_campaign_info is not None else "campaign"
    _base_subtitle = f"{dataset_name or 'dataset'} · {_slug}"
    _color_label = sfxi_color_dropdown.value
    _hue = None if _color_label == "(none)" else sfxi_hue_registry.get(_color_label)
    _okabe_ito = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ]
    _fallback_scheme = "tableau20"
    _sfxi_brush = alt.selection_interval(name="sfxi_brush", encodings=["x", "y"])
    if df_sfxi_scatter.is_empty():
        empty_df = pl.DataFrame(
            schema={
                "__row_id": pl.Int64,
                "id": pl.Utf8,
                "logic_fidelity": pl.Float64,
                "effect_scaled": pl.Float64,
                "score": pl.Float64,
            }
        )
        tooltip_cols = [
            c
            for c in [
                "id",
                "__row_id",
                "observed_round",
                "label_src",
                "label_ts",
                "logic_fidelity",
                "effect_scaled",
                "score",
            ]
            if c in empty_df.columns
        ]
        _chart = (
            alt.Chart(empty_df)
            .mark_circle(opacity=0.7, stroke=None, strokeWidth=0, size=90)
            .encode(
                x=alt.X(
                    "logic_fidelity",
                    title="Logic fidelity",
                    scale=alt.Scale(domain=[0, 1.02]),
                ),
                y=alt.Y(
                    "effect_scaled",
                    title="Effect (scaled)",
                    scale=alt.Scale(domain=[0, 1.02]),
                ),
                color=alt.Color(
                    "score:Q",
                    title="SFXI score",
                    legend=alt.Legend(title="SFXI score", format=".2f", tickCount=5),
                ),
                tooltip=tooltip_cols,
            )
            .add_params(_sfxi_brush)
        )
        _chart = with_title(
            _chart,
            "SFXI components: logic fidelity vs scaled effect",
            _base_subtitle,
        ).properties(width=_plot_size, height=_plot_size)
        sfxi_chart_note_md = mo.md(_sfxi_explain_text)
    else:
        plot_cols = [
            col
            for col in ["__row_id", "id", "logic_fidelity", "effect_scaled", "score"]
            if col in df_sfxi_scatter.columns
        ]
        if _hue is not None and _hue.key in df_sfxi_scatter.columns and _hue.key not in plot_cols:
            plot_cols.append(_hue.key)
        df_sfxi_plot = df_sfxi_scatter.select(plot_cols)
        tooltip_cols = [
            c
            for c in [
                "id",
                "__row_id",
                "observed_round",
                "label_src",
                "label_ts",
                "logic_fidelity",
                "effect_scaled",
                "score",
            ]
            if c in df_sfxi_plot.columns
        ]
        _color_note = None
        _color_title = None
        _color_tooltip = None
        _color_encoding = alt.Undefined
        if _hue is not None and _hue.key in df_sfxi_plot.columns:
            _non_null_count = df_sfxi_plot.select(pl.col(_hue.key).count()).item() if df_sfxi_plot.height else 0
            if _non_null_count == 0:
                _color_note = f"Color `{_hue.key}` has no non-null values; rendering without color."
            elif _hue.kind == "categorical" and _hue.category_labels:
                _label_col = f"{_hue.key}__label"
                _yes_label, _no_label = _hue.category_labels
                df_sfxi_plot = df_sfxi_plot.with_columns(
                    pl.when(pl.col(_hue.key)).then(pl.lit(_yes_label)).otherwise(pl.lit(_no_label)).alias(_label_col)
                )
                _color_title = _hue.label
                _color_tooltip = _label_col
                _color_scale = (
                    alt.Scale(domain=[_yes_label, _no_label], range=[_okabe_ito[5], "#B0B0B0"])
                    if "top_k" in _hue.key
                    else alt.Scale(domain=[_yes_label, _no_label], range=[_okabe_ito[2], "#B0B0B0"])
                )
                _color_encoding = alt.Color(
                    f"{_label_col}:N",
                    title=_color_title,
                    scale=_color_scale,
                    legend=alt.Legend(title=_color_title),
                )
            elif _hue.kind == "numeric":
                _color_title = _hue.label
                _color_tooltip = _hue.key
                _color_encoding = alt.Color(
                    f"{_hue.key}:Q",
                    title=_color_title,
                    legend=alt.Legend(title=_color_title, format=".2f", tickCount=5),
                )
            else:
                _n_unique = df_sfxi_plot.select(pl.col(_hue.key).n_unique()).item() if df_sfxi_plot.height else 0
                _color_scale = (
                    alt.Scale(range=_okabe_ito) if _n_unique <= len(_okabe_ito) else alt.Scale(scheme=_fallback_scheme)
                )
                _color_title = _hue.label
                _color_tooltip = _hue.key
                _color_encoding = alt.Color(
                    f"{_hue.key}:N",
                    title=_color_title,
                    scale=_color_scale,
                    legend=alt.Legend(title=_color_title),
                )
        elif _color_label and _color_label != "(none)":
            _color_note = f"Color `{_color_label}` unavailable; rendering without color."

        if _color_tooltip and _color_tooltip in df_sfxi_plot.columns and _color_tooltip not in tooltip_cols:
            tooltip_cols.append(_color_tooltip)
        _chart = (
            alt.Chart(df_sfxi_plot)
            .mark_circle(opacity=0.7, stroke=None, strokeWidth=0, size=90)
            .encode(
                x=alt.X(
                    "logic_fidelity",
                    title="Logic fidelity",
                    scale=alt.Scale(domain=[0, 1.02]),
                ),
                y=alt.Y(
                    "effect_scaled",
                    title="Effect (scaled)",
                    scale=alt.Scale(domain=[0, 1.02]),
                ),
                color=_color_encoding,
                tooltip=tooltip_cols,
            )
            .add_params(_sfxi_brush)
        )
        _color_subtitle = _base_subtitle
        if _color_title:
            _color_subtitle = f"{_color_subtitle} · color={_color_title}"
        _chart = with_title(
            _chart,
            "SFXI components: logic fidelity vs scaled effect",
            f"{_color_subtitle} · n={df_sfxi_plot.height}",
        ).properties(width=_plot_size, height=_plot_size)
        sfxi_chart_note_md = mo.md(_color_note) if _color_note else mo.md("")

    sfxi_chart_ui = mo.ui.altair_chart(_chart)
    return sfxi_chart_note_md, sfxi_chart_ui


@app.cell
def _(df_sfxi_scatter, resolve_brush_selection, sfxi_chart_ui):
    df_sfxi_brush_selected, _ = resolve_brush_selection(
        df_plot=df_sfxi_scatter,
        selected_raw=sfxi_chart_ui.value,
        selection_enabled=not df_sfxi_scatter.is_empty(),
        id_col="__row_id",
    )
    df_sfxi_selected = df_sfxi_scatter
    return df_sfxi_brush_selected, df_sfxi_selected


@app.cell(column=5)
def _(
    label_notice_md,
    label_src_multiselect,
    mo,
    mode_dropdown,
    opal_campaign_notice_md,
    opal_label_view_dropdown,
    opal_labels_table_note_md,
    opal_labels_table_ui,
    opal_provenance_md,
    opal_round_dropdown,
    opal_round_notice_md,
    opal_run_id_dropdown,
    pred_diag_md,
    pred_notice_md,
    round_notice_md,
    sfxi_source_dropdown,
    view_notice_md,
):
    _opal_references_md = mo.Html(
        """
        <div>References (local files):</div>
        <ul style="margin-top: 0; margin-bottom: 0; padding-left: 1.1rem;">
          <li><code>src/dnadesign/opal/README.md</code></li>
          <li><code>src/dnadesign/opal/src/cli/README.md</code></li>
          <li><code>src/dnadesign/opal/docs/setpoint_fidelity_x_intensity.md</code></li>
        </ul>
        """
    )

    selection_controls = [
        c for c in [mode_dropdown, sfxi_source_dropdown, opal_round_dropdown, opal_run_id_dropdown] if c is not None
    ]
    selection_row = mo.hstack(selection_controls) if selection_controls else mo.md("")

    label_controls = [c for c in [label_src_multiselect, opal_label_view_dropdown] if c is not None]
    label_controls_row = mo.hstack(label_controls) if label_controls else mo.md("")

    sfxi_panel = mo.vstack(
        [
            mo.md("## Characterizing promoters by their SFXI score"),
            mo.md(
                "SFXI (setpoint fidelity x intensity) compresses a 2-factor logic response into a single "
                "scalar that balances logic correctness and brightness. OPAL is the active-learning harness "
                "that tries to predict an 8-vector response and reduces it via an objective to rank candidates."
            ),
            _opal_references_md,
            opal_campaign_notice_md,
            mo.md("### Canonical selection"),
            selection_row,
            round_notice_md,
            pred_notice_md,
            pred_diag_md,
            view_notice_md,
            opal_provenance_md,
            mo.md("### Labels (observed events)"),
            mo.md(
                "These rows reflect experimental label events ingested for the selected campaign "
                "(e.g., via `opal ingest-y`). The as-of round defines the training cutoff for active learning."
            ),
            label_controls_row,
            label_notice_md,
            opal_labels_table_note_md,
            opal_round_notice_md,
            opal_labels_table_ui,
        ]
    )
    sfxi_panel
    return


@app.cell(column=6)
def _(
    mo,
    mode_dropdown,
    sfxi_beta_input,
    sfxi_chart_note_md,
    sfxi_chart_ui,
    sfxi_color_dropdown,
    sfxi_gamma_input,
    sfxi_meta_md,
    sfxi_notice_md,
    sfxi_source_notice_md,
    sfxi_p00_slider,
    sfxi_p01_slider,
    sfxi_p10_slider,
    sfxi_p11_slider,
    sfxi_setpoint_md,
):
    setpoint_row_top = mo.hstack([sfxi_p00_slider, sfxi_p10_slider, sfxi_p01_slider, sfxi_p11_slider])
    exponent_row = mo.hstack([sfxi_beta_input, sfxi_gamma_input])
    sfxi_controls = mo.vstack([setpoint_row_top, exponent_row])
    sfxi_defs_md = mo.md(
        "- **logic_fidelity**: proximity to the setpoint logic vector.\n"
        "- **effect_scaled**: intensity normalized by the current scaling denom.\n"
        "- **score** = (logic_fidelity ** beta) * (effect_scaled ** gamma)"
    )
    controls_block = (
        sfxi_controls if mode_dropdown.value == "Overlay" else mo.md("Switch to **Overlay** to edit SFXI parameters.")
    )
    mo.vstack(
        [
            mo.md("## SFXI: setpoint + exponents"),
            mo.md(
                "The **setpoint** defines the desired logic profile `p ∈ [0,1]^4` in state order "
                "`[00, 10, 01, 11]`. Adjusting `p` changes what “good” logic looks like and how intensity is "
                "weighted across states. Exponents β/γ tune the trade-off between logic fidelity and intensity."
            ),
            mo.md(
                "**Setpoint changes update SFXI scoring/ranking only; the artifact model is fixed and "
                "no canonical files are modified.**"
            ),
            controls_block,
            sfxi_notice_md,
            sfxi_source_notice_md,
            sfxi_setpoint_md,
            sfxi_defs_md,
            sfxi_color_dropdown,
            sfxi_chart_note_md,
            sfxi_chart_ui,
            sfxi_meta_md,
        ]
    )
    return


@app.cell(column=7)
def _(
    mo,
    mode_dropdown,
    rf_model_source_note_md,
    transient_cluster_chart,
    transient_cluster_hue_dropdown,
    transient_cluster_metric_dropdown,
    transient_feature_chart,
    transient_hist_chart,
    transient_md,
):
    rf_header = "## OPAL artifact model (campaign run)"
    rf_note = (
        "**Overlay mode** recomputes SFXI scores from stored predictions using your current parameters. "
        "Artifacts are optional and used only for feature importance summaries."
    )

    feature_panel = (
        transient_feature_chart if transient_feature_chart is not None else mo.md("Feature importance unavailable.")
    )
    hist_panel = transient_hist_chart if transient_hist_chart is not None else mo.md("Score histogram unavailable.")
    cluster_panel = (
        transient_cluster_chart
        if transient_cluster_chart is not None
        else mo.md("Leiden cluster score plot unavailable.")
    )
    cluster_controls = mo.hstack([transient_cluster_metric_dropdown, transient_cluster_hue_dropdown])
    mode_note = mo.md("")
    if mode_dropdown.value != "Overlay":
        mode_note = mo.md("Switch to **Overlay** mode to recompute SFXI metrics from stored predictions.")
    rf_blocks = [
        mo.md(rf_header),
        rf_model_source_note_md,
        mo.md(
            "Label history is the canonical per-record source for observed/predicted values. "
            "Artifacts in `outputs/rounds/round_<r>` are optional and used only for feature importance."
        ),
        transient_md if transient_md is not None else mo.md(""),
        mo.md(rf_note),
        mode_note,
        feature_panel,
        mo.md(
            "Feature importance highlights which input dimensions the artifact model relies on most for its "
            "predictions."
        ),
        hist_panel,
        mo.md("The histogram shows the distribution of predicted SFXI scores across the full candidate pool."),
        cluster_controls,
        cluster_panel,
        mo.md(
            "The cluster plot compares the selected metric across Leiden clusters, ordered numerically for stability."
        ),
    ]
    overlay_panel = mo.vstack(rf_blocks)
    overlay_panel
    return


@app.cell(column=8)
def _(
    active_record,
    active_record_id,
    build_diagnostics_panels,
    diagnostics_sample_slider,
    diagnostics_seed_slider,
    derived_metrics_note_md,
    df_pred_selected,
    df_view,
    fig_to_image,
    hue_registry,
    mo,
    opal_campaign_info,
    opal_labels_current_df,
    sfxi_params,
    support_color_dropdown,
    support_y_dropdown,
    uncertainty_available,
    uncertainty_color_dropdown,
):
    diag_sample_n = int(diagnostics_sample_slider.value) if diagnostics_sample_slider is not None else 0
    diag_seed = int(diagnostics_seed_slider.value) if diagnostics_seed_slider is not None else 0

    support_y_map = {
        "Score": "opal__view__score",
        "Logic fidelity": "opal__view__logic_fidelity",
    }
    support_y_col = support_y_map.get(support_y_dropdown.value, "opal__view__score")
    support_color = hue_registry.get(support_color_dropdown.value)
    uncertainty_color = hue_registry.get(uncertainty_color_dropdown.value)

    panels = build_diagnostics_panels(
        df_pred_selected=df_pred_selected,
        df_view=df_view,
        active_record=active_record,
        active_record_id=active_record_id,
        opal_campaign_info=opal_campaign_info,
        opal_labels_current_df=opal_labels_current_df,
        sfxi_params=sfxi_params,
        support_y_col=support_y_col,
        support_color=support_color,
        uncertainty_color=uncertainty_color,
        uncertainty_available=uncertainty_available,
        sample_n=diag_sample_n,
        seed=diag_seed,
    )

    def _render_panel(panel, *, default_note: str):
        if panel.chart is None:
            return mo.md(panel.note or default_note)
        element = fig_to_image(panel.chart) if panel.kind != "altair" else mo.ui.altair_chart(panel.chart)
        if panel.note:
            return mo.vstack([mo.md(panel.note), element])
        return element

    sample_note = mo.md(panels.sample_note) if panels.sample_note else mo.md("")
    factorial_panel = _render_panel(panels.factorial, default_note="Factorial effects unavailable.")
    decomp_panel = _render_panel(panels.decomposition, default_note="Setpoint decomposition unavailable.")
    support_panel = _render_panel(panels.support, default_note="Support diagnostics unavailable.")
    uncertainty_panel = _render_panel(panels.uncertainty, default_note="Uncertainty plot unavailable.")
    sweep_panel = _render_panel(panels.sweep, default_note="Setpoint sweep unavailable.")
    intensity_panel = _render_panel(panels.intensity, default_note="Intensity scaling unavailable.")

    mo.vstack(
        [
            mo.md("## Diagnostics / AL Guidance"),
            derived_metrics_note_md,
            sample_note,
            mo.md("### Factorial-effects map"),
            factorial_panel,
            mo.md("### Setpoint decomposition (active record)"),
            decomp_panel,
            mo.md("### Logic support diagnostics"),
            mo.vstack([support_y_dropdown, support_color_dropdown]),
            support_panel,
            mo.md("### Uncertainty diagnostics"),
            uncertainty_color_dropdown,
            uncertainty_panel,
            mo.md("### Setpoint sweep (objective landscape)"),
            sweep_panel,
            mo.md("### Intensity scaling diagnostics"),
            intensity_panel,
        ]
    )
    return


@app.cell
def _(df_view, mode_dropdown, pl):
    df_overlay_top_k_pool = df_view.head(0)
    df_score_top_k_pool = df_view.head(0)
    if "opal__view__top_k" in df_view.columns:
        df_score_top_k_pool = df_view.filter(pl.col("opal__view__top_k").fill_null(False))
        if mode_dropdown.value == "Overlay":
            df_overlay_top_k_pool = df_score_top_k_pool
    return df_overlay_top_k_pool, df_score_top_k_pool


@app.cell
def _(mo):
    inspect_pool_label_map = {
        "Full dataset (all rows)": "df_active",
        "UMAP brush selection": "df_umap_selected",
        "SFXI brush selection": "df_sfxi_brush_selected",
        "SFXI scored labels (current view)": "df_sfxi_selected",
        "Selected score Top-K": "df_score_top_k_pool",
        "Overlay Top-K": "df_overlay_top_k_pool",
    }
    inspect_pool_dropdown = mo.ui.dropdown(
        options=list(inspect_pool_label_map.keys()),
        value="Full dataset (all rows)",
        label="Record pool",
        full_width=True,
    )
    return inspect_pool_dropdown, inspect_pool_label_map


@app.cell
def _(
    df_active,
    df_overlay_top_k_pool,
    df_score_top_k_pool,
    df_sfxi_brush_selected,
    df_sfxi_selected,
    df_umap_selected,
    inspect_pool_dropdown,
    inspect_pool_label_map,
):
    pool_choice = inspect_pool_label_map.get(inspect_pool_dropdown.value, inspect_pool_dropdown.value)
    if pool_choice == "df_umap_selected":
        df_pool = df_umap_selected
    elif pool_choice == "df_sfxi_brush_selected":
        df_pool = df_sfxi_brush_selected
    elif pool_choice == "df_sfxi_selected":
        df_pool = df_sfxi_selected
    elif pool_choice == "df_overlay_top_k_pool":
        df_pool = df_overlay_top_k_pool
    elif pool_choice == "df_score_top_k_pool":
        df_pool = df_score_top_k_pool
    else:
        df_pool = df_active
    return (df_pool,)


@app.cell
def _(mo):
    active_id_state, set_active_id_state = mo.state(None)
    return active_id_state, set_active_id_state


@app.cell
def _(active_id_state, df_pool, mo):
    id_override_input = mo.ui.text(
        value="",
        label="ID override (exact or substring)",
        full_width=True,
    )
    prev_button = mo.ui.run_button(label="Prev")
    next_button = mo.ui.run_button(label="Next")
    id_nav_buttons = mo.hstack([prev_button, next_button])

    _id_values = []
    if "id" in df_pool.columns:
        _id_values = df_pool.select("id").drop_nulls().to_series().to_list()

    _max_ids = 1000
    _truncated = len(_id_values) > _max_ids
    _id_values = _id_values[:_max_ids]

    _active_id_value = active_id_state()
    if _active_id_value and _active_id_value not in _id_values:
        _id_values = [_active_id_value, *_id_values]

    _default_id = _active_id_value if _active_id_value in _id_values else None
    if _default_id is None and _id_values:
        _default_id = _id_values[0]
    if not _id_values:
        _id_values = ["(no ids)"]
        _default_id = "(no ids)"

    id_dropdown = mo.ui.dropdown(
        options=_id_values,
        value=_default_id,
        label="Record id",
        full_width=True,
    )
    return (
        id_dropdown,
        id_nav_buttons,
        id_override_input,
        next_button,
        prev_button,
    )


@app.cell
def _(
    active_id_state,
    df_pool,
    id_dropdown,
    id_override_input,
    next_button,
    pl,
    prev_button,
    set_active_id_state,
):
    active_record_id = None
    active_record = None

    _id_values = []
    if not df_pool.is_empty() and "id" in df_pool.columns:
        _id_values = df_pool.select("id").drop_nulls().to_series().to_list()
    _max_ids = 1000
    _id_values = _id_values[:_max_ids]

    _override_text = str(id_override_input.value or "").strip()
    _prev_clicked = bool(prev_button.value)
    _next_clicked = bool(next_button.value)
    _selected_id = id_dropdown.value if id_dropdown is not None else None
    if _selected_id == "(no ids)":
        _selected_id = None
    _active_id_value = active_id_state()

    _candidate_id = None

    if _override_text:
        _exact = df_pool.filter(pl.col("id") == _override_text)
        if _exact.height:
            active_record_id = _exact["id"][0]
        else:
            _matches = df_pool.filter(pl.col("id").cast(pl.Utf8).str.contains(_override_text))
            if _matches.height:
                active_record_id = _matches["id"][0]
    else:
        _candidate_id = (
            _selected_id
            if _selected_id in _id_values
            else (_active_id_value if _active_id_value in _id_values else None)
        )
        if _id_values and (_prev_clicked or _next_clicked):
            if _candidate_id not in _id_values:
                _candidate_id = _id_values[0]
            _current_index = _id_values.index(_candidate_id)
            if _prev_clicked and not _next_clicked:
                _candidate_id = _id_values[(_current_index - 1) % len(_id_values)]
            elif _next_clicked and not _prev_clicked:
                _candidate_id = _id_values[(_current_index + 1) % len(_id_values)]

        if _candidate_id is None and _id_values:
            _candidate_id = _id_values[0]

        if _candidate_id is not None:
            active_record_id = _candidate_id

    if active_record_id != _active_id_value:
        set_active_id_state(active_record_id)

    if active_record_id is not None:
        active_record = df_pool.filter(pl.col("id") == active_record_id).head(1)
        if active_record.is_empty():
            active_record = None
    return active_record, active_record_id


@app.cell
def _(active_record, active_record_id, mo, show_metadata_checkbox):
    _unused = show_metadata_checkbox
    if active_record is None or active_record.is_empty():
        summary_md = mo.md("No active record to display.")
        sequence_md = mo.md("")
    else:
        summary_md = mo.md(
            "\n".join(
                [
                    f"Record id: `{active_record_id}`",
                ]
            )
        )

        seq_text = ""
        if "sequence" in active_record.columns:
            seq_text = active_record["sequence"][0]
        sequence_md = mo.md(f"Sequence: `{seq_text}`" if seq_text else "Sequence column missing.")
    return sequence_md, summary_md


@app.cell
def _(active_record, mo, pl):
    tfbs_note_md = mo.md("")
    tfbs_table_ui = mo.md("")
    if active_record is None or active_record.is_empty():
        tfbs_note_md = mo.md("No active record.")
    elif "densegen__used_tfbs_detail" not in active_record.columns:
        tfbs_note_md = mo.md("No TFBS placements available.")
    else:
        tfbs_value = active_record["densegen__used_tfbs_detail"][0]
        if tfbs_value is None:
            tfbs_note_md = mo.md("No TFBS placements available.")
        else:
            try:
                tfbs_table = pl.DataFrame(tfbs_value)
                tfbs_table_ui = mo.ui.table(tfbs_table)
            except Exception:
                tfbs_note_md = mo.md("No TFBS placements available.")
    return tfbs_note_md, tfbs_table_ui


@app.cell
def _(mo):
    inspect_view_mode_dropdown = mo.ui.dropdown(
        options=["BaseRender", "DenseGen visual"],
        value="BaseRender",
        label="View mode",
        full_width=True,
    )
    show_metadata_checkbox = mo.ui.checkbox(
        label="Show metadata (advanced)",
        value=False,
    )
    return inspect_view_mode_dropdown, show_metadata_checkbox


@app.cell
def _(active_record, dataset_path, inspect_view_mode_dropdown, mo):
    baserender_error_md = mo.md("")
    render_element = mo.md("")
    view_mode = inspect_view_mode_dropdown.value

    if active_record is None or active_record.is_empty():
        baserender_error_md = mo.md("No active record to render.")
    else:
        if view_mode == "BaseRender":
            try:
                from dnadesign.baserender.src import api as _baserender_api
                from dnadesign.baserender.src.io.parquet import (
                    read_parquet_records_by_ids as _read_parquet_records_by_ids,
                )

                _active_id = active_record["id"][0] if "id" in active_record.columns else None
                if _active_id is None:
                    raise ValueError("BaseRender requires a non-null id column.")

                _records_iter = _read_parquet_records_by_ids(
                    path=dataset_path,
                    ids=[str(_active_id)],
                    id_col="id",
                    sequence_col="sequence",
                    annotations_col="densegen__used_tfbs_detail",
                )
                _record = next(iter(_records_iter), None)
                if _record is None:
                    raise ValueError("Record not found in dataset for BaseRender.")

                _fig = _baserender_api.render_image(_record, fmt="png")
                _fig.patch.set_facecolor("white")
                _fig.patch.set_alpha(1.0)
                render_element = _fig
            except Exception as exc:
                baserender_error_md = mo.md(
                    "\n".join(
                        [
                            "BaseRender failed; falling back to DenseGen visual.",
                            f"Error: {exc}",
                            "CLI alternative: `uv run baserender --help`",
                        ]
                    )
                )
                view_mode = "DenseGen visual"

        if view_mode == "DenseGen visual":
            if "densegen__visual" in active_record.columns:
                visual_text = active_record["densegen__visual"][0]
                if isinstance(visual_text, str):
                    visual_text = visual_text.replace("\r\n", "\n").replace("\r", "\n")
                    visual_text = visual_text.replace("-->", "\n-->").replace("<--", "\n<--")
                    visual_text = "\n".join(line for line in visual_text.split("\n") if line.strip())
                    visual_text = visual_text.strip()
                    render_element = mo.md(f"```\n{visual_text}\n```")
                else:
                    render_element = mo.md("densegen__visual not available.")
            else:
                render_element = mo.md("densegen__visual not available.")
    return baserender_error_md, render_element


@app.cell
def _(active_record_id, dataset_name, mo, repo_root):
    _active_id = active_record_id if active_record_id is not None else "unknown"
    _default_path = (
        repo_root
        / "src"
        / "dnadesign"
        / "opal"
        / "notebooks"
        / "_outputs"
        / "baserender"
        / f"{dataset_name}__{_active_id}__baserender.pdf"
    )
    save_pdf_path_input = mo.ui.text(
        value=str(_default_path),
        label="PDF output path (absolute or repo-root-relative)",
        full_width=True,
    )
    save_pdf_button = mo.ui.run_button(label="Save still (PDF)")
    return save_pdf_button, save_pdf_path_input


@app.cell
def _(
    Path,
    active_record,
    dataset_path,
    mo,
    repo_root,
    save_pdf_button,
    save_pdf_path_input,
):
    save_pdf_status_md = None
    save_pdf_path = None
    save_pdf_target_md = mo.md("")

    raw_path = save_pdf_path_input.value.strip()
    suffix_adjusted = False
    if raw_path:
        _target_path = Path(raw_path).expanduser()
        if not _target_path.is_absolute():
            _target_path = (repo_root / _target_path).resolve()
        if _target_path.suffix.lower() != ".pdf":
            _target_path = _target_path.with_suffix(".pdf")
            suffix_adjusted = True
        save_pdf_path = _target_path
        suffix_note = " (pdf suffix enforced)" if suffix_adjusted else ""
        save_pdf_target_md = mo.md(f"Target PDF path: `{save_pdf_path}`{suffix_note}")
    else:
        save_pdf_target_md = mo.md("Target PDF path: *(enter a path to enable saving)*")

    if save_pdf_button.value:
        if active_record is None or active_record.is_empty():
            save_pdf_status_md = mo.md("No active record to save.")
        elif save_pdf_path is None:
            save_pdf_status_md = mo.md("No output path provided.")
        else:
            try:
                from dnadesign.baserender.src import api as _baserender_api
                from dnadesign.baserender.src.io.parquet import (
                    read_parquet_records_by_ids as _read_parquet_records_by_ids,
                )

                _active_id = active_record["id"][0] if "id" in active_record.columns else None
                if _active_id is None:
                    raise ValueError("BaseRender requires a non-null id column.")

                _records_iter = _read_parquet_records_by_ids(
                    path=dataset_path,
                    ids=[str(_active_id)],
                    id_col="id",
                    sequence_col="sequence",
                    annotations_col="densegen__used_tfbs_detail",
                )
                _record = next(iter(_records_iter), None)
                if _record is None:
                    raise ValueError("Record not found in dataset for BaseRender.")

                save_pdf_path.parent.mkdir(parents=True, exist_ok=True)

                _fig = _baserender_api.render_image(_record, fmt="pdf")
                _fig.patch.set_facecolor("white")
                _fig.patch.set_alpha(1.0)
                _fig.savefig(
                    save_pdf_path,
                    format="pdf",
                    bbox_inches=None,
                    pad_inches=0.0,
                    facecolor="white",
                )
                from dnadesign.opal.src.plots._mpl_utils import ensure_mpl_config_dir

                ensure_mpl_config_dir()
                import matplotlib.pyplot as plt

                plt.close(_fig)
                save_pdf_status_md = mo.md(f"Saved PDF: `{save_pdf_path}`")
            except Exception as exc:
                save_pdf_status_md = mo.md(f"Failed to save PDF at `{save_pdf_path}`: {exc}")
    return save_pdf_status_md, save_pdf_target_md


@app.cell(column=9)
def _(
    export_button,
    export_format_dropdown,
    export_note_md,
    export_source_dropdown,
    export_status_md,
    mo,
):
    _export_label = export_source_dropdown.value
    export_header_md = mo.md(
        "## Export a dataframe\n"
        "Destination: `src/dnadesign/opal/notebooks/_outputs/"
        "promoter_eda_exports/promoter_eda_<dataset>_<timestamp>.<format>`\n"
        "Overlay mode adds `opal_dashboard__label_hist` for dashboard provenance."
    )
    mo.vstack(
        [
            export_header_md,
            export_note_md if export_note_md is not None else mo.md(""),
            export_source_dropdown,
            export_format_dropdown,
            export_button,
            export_status_md,
        ]
    )
    return


@app.cell
def _(
    mo,
    opal_campaign_info,
    opal_pred_selected_round,
    opal_pred_selected_run_id,
    opal_selected_round,
    resolve_artifact_state,
):
    rf_model_source_note_md = mo.md("")
    artifact_result = resolve_artifact_state(
        campaign_info=opal_campaign_info,
        as_of_round=opal_pred_selected_round,
        run_id=opal_pred_selected_run_id,
    )
    artifact_ready = artifact_result.use_artifact

    model_lines = ["**Model status**"]
    if artifact_ready:
        run_id_note = f" (run_id=`{opal_pred_selected_run_id}`)" if opal_pred_selected_run_id is not None else ""
        model_lines.append(f"- Model: OPAL artifact{run_id_note}")
        if artifact_result.model_path is not None:
            model_lines.append(f"- Path: `{artifact_result.model_path}`")
    else:
        model_lines.append("- Model: OPAL artifact unavailable (feature importance disabled).")
        model_lines.append(f"- Details: {artifact_result.note}")

    if opal_pred_selected_round is None:
        model_lines.append("- As-of round: **missing** (select a round for canonical/overlay views)")
    else:
        model_lines.append(f"- As-of round: `R={opal_pred_selected_round}`")
    if opal_selected_round is None:
        model_lines.append("- Labels view round: `all` (no round filter)")
    else:
        model_lines.append(f"- Labels view round: `R={opal_selected_round}`")
    model_lines.append("- Overlay uses stored predictions; artifacts are optional for feature importance.")

    rf_model_source_note_md = mo.md("\n".join(model_lines))
    return artifact_result, rf_model_source_note_md


@app.cell
def _(
    build_label_sfxi_view,
    compute_sfxi_params,
    mo,
    mode_dropdown,
    opal_campaign_info,
    opal_labels_current_df,
    opal_labels_view_df,
    opal_selected_round,
    resolve_sfxi_readiness,
    sfxi_beta_input,
    sfxi_default_values,
    sfxi_fixed_params,
    sfxi_gamma_input,
    sfxi_math,
    sfxi_p00_slider,
    sfxi_p01_slider,
    sfxi_p10_slider,
    sfxi_p11_slider,
):
    df_sfxi = opal_labels_view_df.head(0)
    sfxi_meta_md = mo.md("")
    sfxi_notice_md = mo.md("")

    readiness = resolve_sfxi_readiness(opal_campaign_info)
    if readiness.notice:
        sfxi_notice_md = mo.md(readiness.notice)

    delta = float(sfxi_fixed_params.get("delta", 0.0))
    p = float(sfxi_fixed_params.get("percentile", 95.0))
    fallback_p = float(sfxi_fixed_params.get("fallback_percentile", p))
    min_n = int(sfxi_fixed_params.get("min_n", 5))
    eps = float(sfxi_fixed_params.get("eps", 1.0e-8))

    if mode_dropdown.value == "Overlay":
        _setpoint = [
            sfxi_p00_slider.value,
            sfxi_p10_slider.value,
            sfxi_p01_slider.value,
            sfxi_p11_slider.value,
        ]
        beta = sfxi_beta_input.value
        gamma = sfxi_gamma_input.value
    else:
        _setpoint = [
            sfxi_default_values.get("p00", 0.0),
            sfxi_default_values.get("p10", 0.0),
            sfxi_default_values.get("p01", 0.0),
            sfxi_default_values.get("p11", 1.0),
        ]
        beta = sfxi_default_values.get("beta", 1.0)
        gamma = sfxi_default_values.get("gamma", 1.0)

    sfxi_params = compute_sfxi_params(
        setpoint=_setpoint,
        beta=beta,
        gamma=gamma,
        delta=delta,
        p=p,
        fallback_p=fallback_p,
        min_n=min_n,
        eps=eps,
        state_order=sfxi_math.STATE_ORDER,
    )

    label_view = build_label_sfxi_view(
        readiness=readiness,
        selected_round=opal_selected_round,
        labels_view_df=opal_labels_view_df,
        labels_current_df=opal_labels_current_df,
        params=sfxi_params,
    )
    if label_view.notice:
        sfxi_notice_md = mo.md(label_view.notice)
    df_sfxi = label_view.df
    if not df_sfxi.is_empty():
        sfxi_meta_md = mo.md("Label-level SFXI metrics are shown in the Labels table.")
    return df_sfxi, readiness, sfxi_meta_md, sfxi_notice_md, sfxi_params


@app.cell
def _(
    build_pred_sfxi_view,
    df_pred_selected,
    mode_dropdown,
    opal_labels_current_df,
    readiness,
    sfxi_params,
):
    pred_view = build_pred_sfxi_view(
        pred_df=df_pred_selected,
        labels_current_df=opal_labels_current_df,
        y_col=readiness.y_col if readiness is not None else None,
        params=sfxi_params,
        mode=mode_dropdown.value,
    )
    df_sfxi_pred = pred_view.df
    sfxi_pred_notice = pred_view.notice
    return df_sfxi_pred, sfxi_pred_notice


@app.cell
def _(
    attach_namespace_columns,
    df_sfxi,
    df_sfxi_pred,
    df_view,
    mo,
    pl,
    sfxi_pred_notice,
    sfxi_source_dropdown,
):
    df_obs = df_sfxi.with_columns(pl.lit("Observed").alias("sfxi_source")) if df_sfxi is not None else pl.DataFrame()
    df_pred = (
        df_sfxi_pred.with_columns(pl.lit("Predicted").alias("sfxi_source"))
        if df_sfxi_pred is not None
        else pl.DataFrame()
    )
    _sfxi_source = sfxi_source_dropdown.value if sfxi_source_dropdown is not None else "Observed"
    if _sfxi_source == "Predicted":
        df_sfxi_scatter = df_pred
    elif _sfxi_source == "Both":
        if df_obs.is_empty():
            df_sfxi_scatter = df_pred
        elif df_pred.is_empty():
            df_sfxi_scatter = df_obs
        else:
            df_sfxi_scatter = pl.concat([df_obs, df_pred], how="vertical_relaxed")
    else:
        df_sfxi_scatter = df_obs
    df_sfxi_scatter = attach_namespace_columns(
        df=df_sfxi_scatter,
        df_base=df_view,
        prefixes=("cluster__", "densegen__", "opal__sfxi__"),
    )

    sfxi_source_notice_md = mo.md("")
    if _sfxi_source in {"Predicted", "Both"} and df_pred.is_empty():
        sfxi_source_notice_md = (
            mo.md(sfxi_pred_notice) if sfxi_pred_notice else mo.md("No predicted SFXI data available.")
        )
    return df_sfxi_scatter, sfxi_source_notice_md


@app.cell
def _(df_sfxi, mo, opal_campaign_info, opal_labels_view_df, pl):
    opal_labels_table_note_md = mo.md("")
    df_labels = opal_labels_view_df if opal_labels_view_df is not None else pl.DataFrame()
    metrics_cols = [col for col in ["logic_fidelity", "effect_raw", "effect_scaled", "score"] if col in df_sfxi.columns]
    opal_labels_table_ui = mo.ui.table(df_labels, page_size=5)

    if not df_labels.is_empty():
        join_key = "__row_id" if "__row_id" in df_labels.columns and "__row_id" in df_sfxi.columns else None
        if not metrics_cols:
            metrics_cols = ["logic_fidelity", "effect_raw", "effect_scaled", "score"]

        df_table = df_labels
        if df_sfxi.is_empty():
            df_table = df_labels.with_columns([pl.lit(None).alias(col) for col in metrics_cols])
            opal_labels_table_note_md = mo.md(
                "SFXI metrics missing for labels; check that label vectors are present and valid for this round."
            )
        elif join_key is None:
            df_table = df_labels.with_columns([pl.lit(None).alias(col) for col in metrics_cols])
            opal_labels_table_note_md = mo.md(
                "SFXI metrics unavailable: missing `__row_id` for label rows (cannot align metrics)."
            )
        else:
            _label_metrics_df = df_sfxi.select([join_key, *[col for col in metrics_cols if col in df_sfxi.columns]])
            df_table = df_labels.join(_label_metrics_df, on=join_key, how="left")
            if "score" in df_table.columns:
                missing_scores = df_table.select(pl.col("score").is_null().sum()).item()
                if missing_scores:
                    opal_labels_table_note_md = mo.md(
                        f"SFXI metrics missing for {missing_scores} label rows; check label vectors for length/values."
                    )

        y_col_name = opal_campaign_info.y_column if opal_campaign_info is not None else "y_obs"
        y_display_col = y_col_name
        if y_col_name and y_col_name in df_table.columns:
            preview_col = f"{y_col_name}_preview"

            def _preview_vec(val) -> str | None:
                if val is None:
                    return None
                text = str(val)
                if len(text) > 120:
                    return text[:117] + "..."
                return text

            df_table = df_table.with_columns(
                pl.col(y_col_name).map_elements(_preview_vec, return_dtype=pl.Utf8).alias(preview_col)
            )
            y_display_col = preview_col

        display_cols = [
            col
            for col in [
                "__row_id",
                "id",
                "observed_round",
                "label_src",
                "label_ts",
                "logic_fidelity",
                "effect_raw",
                "effect_scaled",
                "score",
                y_display_col,
            ]
            if col in df_table.columns
        ]
        if display_cols:
            opal_labels_table_ui = mo.ui.table(df_table.select(display_cols), page_size=5)
        else:
            opal_labels_table_ui = mo.ui.table(df_table, page_size=5)
    return opal_labels_table_note_md, opal_labels_table_ui


@app.cell
def _(
    Diagnostics,
    artifact_result,
    build_feature_importance_chart,
    build_score_histogram,
    compute_transient_overlay,
    dataset_name,
    df_active,
    df_pred_selected,
    df_sfxi,
    get_feature_importances,
    load_feature_importances_from_artifact,
    mo,
    mode_dropdown,
    opal_campaign_info,
    opal_labels_current_df,
    opal_pred_selected_round,
    opal_pred_selected_run_id,
    readiness,
    sfxi_params,
):
    overlay_result = None
    transient_diag = Diagnostics()
    transient_feature_chart = None
    transient_hist_chart = None
    transient_hist_note_md = None

    selection_params = dict(opal_campaign_info.selection_params or {}) if opal_campaign_info is not None else {}
    model_params = opal_campaign_info.model_params if opal_campaign_info is not None else None
    feature_importances = None
    if artifact_result is None or not artifact_result.use_artifact:
        if opal_campaign_info is not None:
            transient_diag = transient_diag.add_error("Artifact model unavailable; feature importance disabled.")
    else:
        feature_importances = get_feature_importances(artifact_result.model)
        if feature_importances is None:
            feature_importances, fi_note = load_feature_importances_from_artifact(artifact_result.round_dir)
            if feature_importances is not None:
                transient_diag = transient_diag.add_warning("Feature importance loaded from artifact CSV.")
            elif fi_note:
                transient_diag = transient_diag.add_error(fi_note)

    if feature_importances is not None:
        feature_chart = build_feature_importance_chart(
            feature_importances=feature_importances,
            dataset_name=dataset_name,
            selected_round=opal_pred_selected_round,
            n_labels=int(opal_labels_current_df.height) if opal_labels_current_df is not None else 0,
            x_dim=None,
            model_params=model_params or {},
        )
        if feature_chart is not None:
            transient_feature_chart = mo.ui.altair_chart(feature_chart)

    if mode_dropdown.value == "Overlay":
        overlay_result = compute_transient_overlay(
            df_base=df_active,
            pred_df=df_pred_selected,
            labels_current_df=opal_labels_current_df,
            df_sfxi=df_sfxi,
            y_col=readiness.y_col if readiness is not None else None,
            sfxi_params=sfxi_params,
            selection_params=selection_params,
            dataset_name=dataset_name,
            as_of_round=opal_pred_selected_round,
            run_id=opal_pred_selected_run_id,
            x_dim=None,
        )
        transient_diag = transient_diag.merge(overlay_result.diagnostics)
        if overlay_result.hist_chart is not None:
            hist_element = mo.ui.altair_chart(overlay_result.hist_chart)
            if overlay_result.hist_note:
                transient_diag = transient_diag.add_error(overlay_result.hist_note)
                transient_hist_note_md = mo.md(overlay_result.hist_note)
                transient_hist_chart = mo.vstack([hist_element, transient_hist_note_md])
            else:
                transient_hist_chart = hist_element
        elif overlay_result.hist_note:
            transient_diag = transient_diag.add_error(overlay_result.hist_note)
            transient_hist_chart = mo.md(overlay_result.hist_note)
        elif overlay_result.diagnostics.errors:
            transient_hist_chart = mo.md(f"Overlay histogram unavailable: {overlay_result.diagnostics.errors[0]}")
    else:
        df_train = opal_labels_current_df if opal_labels_current_df is not None else df_sfxi.head(0)
        hist_chart, hist_note = build_score_histogram(
            df_pred_scored=df_pred_selected,
            score_col="pred_score",
            df_sfxi=df_sfxi,
            df_train=df_train,
            dataset_name=dataset_name,
            context_label="stored",
        )
        if hist_note:
            transient_hist_note_md = mo.md(hist_note)
            transient_diag = transient_diag.add_error(hist_note)
        if hist_chart is not None:
            hist_element = mo.ui.altair_chart(hist_chart)
            transient_hist_chart = (
                mo.vstack([hist_element, transient_hist_note_md])
                if transient_hist_note_md is not None
                else hist_element
            )
        elif hist_note:
            transient_hist_chart = mo.md(hist_note)
    return (
        overlay_result,
        transient_diag,
        transient_feature_chart,
        transient_hist_chart,
    )


@app.cell
def _(diagnostics_to_lines, mo, transient_diag):
    transient_lines = diagnostics_to_lines(transient_diag)
    transient_md = mo.md("\n".join(transient_lines)) if transient_lines else mo.md("")
    return (transient_md,)


@app.cell
def _(
    build_mode_view,
    df_active,
    df_pred_selected,
    df_sfxi,
    diagnostics_sample_slider,
    diagnostics_seed_slider,
    diagnostics_to_lines,
    list_series_to_numpy,
    load_round_ctx_from_dir,
    mo,
    mode_dropdown,
    np,
    opal_campaign_info,
    opal_labels_asof_df,
    opal_labels_current_df,
    opal_pred_selected_round,
    opal_pred_selected_run_id,
    overlay_result,
    pl,
    resolve_artifact_state,
    sfxi_gates,
    sfxi_math,
    sfxi_params,
    sfxi_support,
    sfxi_uncertainty,
    uncertainty_components_dropdown,
    uncertainty_kind_dropdown,
    uncertainty_reduction_dropdown,
):
    observed_ids = set()
    if opal_labels_asof_df is not None and not opal_labels_asof_df.is_empty() and "id" in opal_labels_asof_df.columns:
        observed_ids = set(
            opal_labels_asof_df.select(pl.col("id").cast(pl.Utf8).drop_nulls().unique()).to_series().to_list()
        )

    if mode_dropdown.value == "Overlay":
        metrics_df = overlay_result.df_pred_scored if overlay_result is not None else None
        score_col = "opal__overlay__score"
        logic_col = "opal__overlay__logic_fidelity"
        effect_col = "opal__overlay__effect_scaled"
        rank_col = "opal__overlay__rank"
        top_k_col = "opal__overlay__top_k"
    else:
        metrics_df = df_pred_selected
        score_col = "pred_score"
        logic_col = "pred_logic_fidelity"
        effect_col = "pred_effect_scaled"
        rank_col = "pred_rank"
        top_k_col = "pred_top_k"

    view_bundle = build_mode_view(
        df_base=df_active,
        metrics_df=metrics_df,
        id_col="id" if "id" in df_active.columns else "__row_id",
        observed_ids=observed_ids,
        observed_scores_df=df_sfxi,
        score_col=score_col,
        logic_col=logic_col,
        effect_col=effect_col,
        rank_col=rank_col,
        top_k_col=top_k_col,
    )
    df_view = view_bundle.df
    derived_notes = []
    uncertainty_available = False
    uncertainty_sample_info = None

    pred_join_key = None
    if df_pred_selected is not None:
        if "id" in df_pred_selected.columns and "id" in df_view.columns:
            pred_join_key = "id"
        elif "__row_id" in df_pred_selected.columns and "__row_id" in df_view.columns:
            pred_join_key = "__row_id"

    if pred_join_key is None or df_pred_selected is None or df_pred_selected.is_empty():
        derived_notes.append("Nearest gate unavailable (missing predictions).")
    elif "pred_y_hat" not in df_pred_selected.columns:
        derived_notes.append("Nearest gate unavailable (missing pred_y_hat).")
    else:
        pred_vec8 = list_series_to_numpy(df_pred_selected.get_column("pred_y_hat"), expected_len=8)
        if pred_vec8 is None:
            derived_notes.append("Nearest gate unavailable (invalid pred_y_hat vectors).")
        else:
            gate_cls, gate_dist = sfxi_gates.nearest_gate(
                pred_vec8[:, 0:4],
                state_order=sfxi_math.STATE_ORDER,
            )
            gate_df = df_pred_selected.select([pred_join_key]).with_columns(
                [
                    pl.Series("opal__sfxi__nearest_gate_class", gate_cls),
                    pl.Series("opal__sfxi__nearest_gate_dist", gate_dist),
                ]
            )
            df_view = df_view.join(gate_df, on=pred_join_key, how="left")

    label_logic = None
    label_ids = []
    if opal_labels_asof_df is None or opal_labels_asof_df.is_empty():
        derived_notes.append("dist_to_labeled_logic unavailable (no labels-as-of).")
    else:
        label_ids = (
            opal_labels_asof_df.select(pl.col("id").cast(pl.Utf8).drop_nulls().unique()).to_series().to_list()
            if "id" in opal_labels_asof_df.columns
            else []
        )
        labels_y_col = None
        if opal_campaign_info is not None and opal_campaign_info.y_column in opal_labels_asof_df.columns:
            labels_y_col = opal_campaign_info.y_column
        elif "y_obs" in opal_labels_asof_df.columns:
            labels_y_col = "y_obs"
            if opal_campaign_info is not None and opal_campaign_info.y_column not in opal_labels_asof_df.columns:
                derived_notes.append("Using y_obs for labels (campaign y_column missing).")
        if labels_y_col is None:
            derived_notes.append("dist_to_labeled_logic unavailable (label vectors missing).")
        else:
            label_vec8 = list_series_to_numpy(opal_labels_asof_df.get_column(labels_y_col), expected_len=8)
            if label_vec8 is None:
                derived_notes.append("dist_to_labeled_logic unavailable (invalid label vectors).")
            else:
                label_logic = label_vec8[:, 0:4]

    if label_logic is not None:
        if pred_join_key is None or df_pred_selected is None or df_pred_selected.is_empty():
            derived_notes.append("dist_to_labeled_logic unavailable (no predictions).")
        elif "pred_y_hat" not in df_pred_selected.columns:
            derived_notes.append("dist_to_labeled_logic unavailable (missing pred_y_hat).")
        else:
            pred_vec = list_series_to_numpy(df_pred_selected.get_column("pred_y_hat"), expected_len=8)
            if pred_vec is None:
                derived_notes.append("dist_to_labeled_logic unavailable (invalid pred_y_hat vectors).")
            else:
                dists = sfxi_support.dist_to_labeled_logic(
                    pred_vec[:, 0:4],
                    label_logic,
                    state_order=sfxi_math.STATE_ORDER,
                )
                dist_df = df_pred_selected.select([pred_join_key]).with_columns(
                    pl.Series("opal__sfxi__dist_to_labeled_logic", dists)
                )
                df_view = df_view.join(dist_df, on=pred_join_key, how="left")

    umap_x = "cluster__ldn_v1__umap_x" if "cluster__ldn_v1__umap_x" in df_view.columns else None
    umap_y = "cluster__ldn_v1__umap_y" if "cluster__ldn_v1__umap_y" in df_view.columns else None
    if umap_x is None:
        umap_x = next((col for col in df_view.columns if col.endswith("__umap_x")), None)
    if umap_y is None:
        umap_y = next((col for col in df_view.columns if col.endswith("__umap_y")), None)
    if umap_x is None or umap_y is None:
        derived_notes.append("dist_to_labeled_x unavailable (UMAP columns missing).")
    elif not label_ids:
        derived_notes.append("dist_to_labeled_x unavailable (no label ids).")
    else:
        label_coords_df = df_view.filter(pl.col("id").cast(pl.Utf8).is_in(label_ids))
        if label_coords_df.is_empty():
            derived_notes.append("dist_to_labeled_x unavailable (no label coords in view).")
        else:
            cand = df_view.select(
                [pl.col(umap_x).cast(pl.Float64, strict=False), pl.col(umap_y).cast(pl.Float64, strict=False)]
            ).to_numpy()
            lab = label_coords_df.select(
                [pl.col(umap_x).cast(pl.Float64, strict=False), pl.col(umap_y).cast(pl.Float64, strict=False)]
            ).to_numpy()
            if not np.all(np.isfinite(cand)) or not np.all(np.isfinite(lab)):
                derived_notes.append("dist_to_labeled_x unavailable (non-finite UMAP coords).")
            else:
                dist_x = sfxi_support.dist_to_labeled_x(cand, lab)
                df_view = df_view.with_columns(pl.Series("opal__sfxi__dist_to_labeled_x", dist_x))

    uncertainty_kind = uncertainty_kind_dropdown.value if uncertainty_kind_dropdown is not None else "score"
    uncertainty_components = (
        uncertainty_components_dropdown.value if uncertainty_components_dropdown is not None else "all"
    )
    uncertainty_reduction = (
        uncertainty_reduction_dropdown.value if uncertainty_reduction_dropdown is not None else "mean"
    )
    uncertainty_sample_n = diagnostics_sample_slider.value if diagnostics_sample_slider is not None else 0
    uncertainty_seed = diagnostics_seed_slider.value if diagnostics_seed_slider is not None else 0

    if opal_campaign_info is None or not opal_campaign_info.x_column:
        derived_notes.append("uncertainty unavailable (x_column missing).")
    elif opal_pred_selected_round is None:
        derived_notes.append("uncertainty unavailable (no selected round).")
    elif opal_campaign_info.x_column not in df_view.columns:
        derived_notes.append("uncertainty unavailable (X column missing in view).")
    else:
        uncertainty_artifact = resolve_artifact_state(
            campaign_info=opal_campaign_info,
            as_of_round=opal_pred_selected_round,
            run_id=opal_pred_selected_run_id,
        )
        if uncertainty_artifact is None or not uncertainty_artifact.use_artifact or uncertainty_artifact.model is None:
            derived_notes.append("uncertainty unavailable (artifact model missing).")
        elif not sfxi_uncertainty.supports_uncertainty(model=uncertainty_artifact.model):
            derived_notes.append("uncertainty unavailable (model lacks per-tree predictions).")
        else:
            uncertainty_df = df_view
            if uncertainty_sample_n and df_view.height > int(uncertainty_sample_n):
                uncertainty_df = df_view.sample(
                    n=int(uncertainty_sample_n),
                    seed=int(uncertainty_seed),
                    shuffle=True,
                )
                uncertainty_sample_info = f"sampled {uncertainty_df.height}/{df_view.height}"
            X = list_series_to_numpy(uncertainty_df.get_column(opal_campaign_info.x_column), expected_len=None)
            if X is None:
                derived_notes.append("uncertainty unavailable (invalid X vectors).")
            else:
                y_ops = opal_campaign_info.y_ops or []
                round_ctx = None
                if y_ops:
                    if uncertainty_artifact.round_dir is None:
                        derived_notes.append("uncertainty unavailable (round_ctx missing).")
                    else:
                        round_ctx, ctx_err = load_round_ctx_from_dir(uncertainty_artifact.round_dir)
                        if round_ctx is None:
                            derived_notes.append(f"uncertainty unavailable ({ctx_err}).")
                denom = None
                if opal_labels_current_df is not None and not opal_labels_current_df.is_empty():
                    labels_y_col = opal_campaign_info.y_column
                    if labels_y_col in opal_labels_current_df.columns:
                        labels_vec8 = list_series_to_numpy(
                            opal_labels_current_df.get_column(labels_y_col), expected_len=8
                        )
                        if labels_vec8 is not None:
                            try:
                                denom = sfxi_math.denom_from_labels(
                                    labels_vec8[:, 4:8],
                                    np.array(sfxi_params.setpoint, dtype=float),
                                    delta=float(sfxi_params.delta),
                                    percentile=int(sfxi_params.p),
                                    min_n=int(sfxi_params.min_n),
                                    eps=float(sfxi_params.eps),
                                    state_order=sfxi_math.STATE_ORDER,
                                )
                            except Exception as exc:
                                derived_notes.append(f"uncertainty denom failed: {exc}")
                try:
                    ctx = sfxi_uncertainty.UncertaintyContext(
                        setpoint=np.array(sfxi_params.setpoint, dtype=float),
                        beta=float(sfxi_params.beta),
                        gamma=float(sfxi_params.gamma),
                        delta=float(sfxi_params.delta),
                        denom=denom,
                        y_ops=y_ops,
                        round_ctx=round_ctx,
                    )
                    result = sfxi_uncertainty.compute_uncertainty(
                        uncertainty_artifact.model,
                        X,
                        kind=uncertainty_kind,
                        ctx=ctx,
                        components=str(uncertainty_components),
                        reduction=str(uncertainty_reduction),
                    )
                except Exception as exc:
                    derived_notes.append(f"uncertainty unavailable ({exc})")
                else:
                    uncertainty_available = True
                    uncertainty_join_key = "id" if "id" in uncertainty_df.columns else "__row_id"
                    uncertainty_values_df = uncertainty_df.select([uncertainty_join_key]).with_columns(
                        pl.Series("opal__sfxi__uncertainty", result.values)
                    )
                    df_view = df_view.join(uncertainty_values_df, on=uncertainty_join_key, how="left")
    view_notice_lines = diagnostics_to_lines(view_bundle.diagnostics)
    view_notice_md = mo.md("\n".join(view_notice_lines)) if view_notice_lines else mo.md("")
    derived_metrics_note_md = mo.md("\n".join(derived_notes)) if derived_notes else mo.md("")
    return df_view, derived_metrics_note_md, uncertainty_available, uncertainty_sample_info, view_notice_md


@app.cell
def _(
    build_cluster_chart,
    dataset_name,
    df_view,
    hue_registry,
    mo,
    transient_cluster_hue_dropdown,
    transient_cluster_metric_dropdown,
):
    _transient_cluster_chart = None
    metric_label = transient_cluster_metric_dropdown.value
    metric_option = hue_registry.get(metric_label)
    hue_label = transient_cluster_hue_dropdown.value
    _hue = None if hue_label == "(none)" else hue_registry.get(hue_label)
    if metric_option is not None and "cluster__ldn_v1" in df_view.columns and metric_option.key in df_view.columns:
        _id_col = "id" if "id" in df_view.columns else "__row_id"
        _cluster_chart = build_cluster_chart(
            df=df_view,
            cluster_col="cluster__ldn_v1",
            metric_col=metric_option.key,
            metric_label=metric_option.label,
            hue=_hue,
            dataset_name=dataset_name,
            id_col=_id_col,
            title="Metric by Leiden cluster",
        )
        if _cluster_chart is not None:
            _transient_cluster_chart = mo.ui.altair_chart(_cluster_chart)

    transient_cluster_chart = _transient_cluster_chart
    return (transient_cluster_chart,)


if __name__ == "__main__":
    app.run()
