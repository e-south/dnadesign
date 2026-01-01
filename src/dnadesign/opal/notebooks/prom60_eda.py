import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _():
    import math
    import os
    from pathlib import Path
    from textwrap import dedent

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from dnadesign.opal.src.models.random_forest import RandomForestModel

    alt.data_transformers.disable_max_rows()

    @alt.theme.register("dnad_white", enable=True)
    def _dnad_white_theme():
        return alt.theme.ThemeConfig(
            {
                "config": {
                    "background": "white",
                    "legend": {
                        "labelColor": "#111111",
                        "titleColor": "#111111",
                        "labelFontSize": 11,
                        "titleFontSize": 12,
                        "fillColor": "transparent",
                        "fillOpacity": 0.0,
                        "strokeColor": None,
                    },
                    "title": {
                        "color": "#111111",
                        "subtitleColor": "#333333",
                        "fontSize": 14,
                        "subtitleFontSize": 11,
                    },
                    "axis": {
                        "domain": True,
                        "domainColor": "#111111",
                        "domainWidth": 1,
                        "grid": True,
                        "gridColor": "#e6e6e6",
                        "gridOpacity": 0.35,
                        "ticks": True,
                        "tickColor": "#111111",
                        "tickSize": 5,
                        "labels": True,
                        "labelFontSize": 11,
                        "labelColor": "#111111",
                        "titleFontSize": 12,
                        "titleColor": "#111111",
                        "labelPadding": 2,
                        "titlePadding": 4,
                    },
                    "axisX": {"domain": True},
                    "axisY": {"domain": True},
                    "view": {"stroke": None},
                }
            }
        )

    try:
        alt.theme.enable("dnad_white")
    except Exception:
        alt.themes.enable("dnad_white")

    def chart_title(text: str, subtitle: str | None = None) -> alt.TitleParams:
        if subtitle:
            return alt.TitleParams(text=text, subtitle=subtitle)
        return alt.TitleParams(text=text)

    def with_title(chart, title: str, subtitle: str | None = None):
        return chart.properties(title=chart_title(title, subtitle))

    return (
        Path,
        RandomForestModel,
        alt,
        dedent,
        math,
        mo,
        np,
        os,
        pl,
        with_title,
    )


@app.cell(hide_code=True)
def _():
    # Notebook polish summary:
    # - Performance: fast list->numpy conversion for vector columns.
    # - Legends: theme-level legend/title colors for visibility on white.
    # - Plots: consistent titles/subtitles and RF/cluster UX improvements.
    # - Clusters: stable numeric ordering for distribution views.
    from dnadesign.opal.src import promoter_eda_utils as eda_utils

    apply_intensity_median_iqr = eda_utils.apply_intensity_median_iqr
    build_color_dropdown_options = eda_utils.build_color_dropdown_options
    build_label_events = eda_utils.build_label_events
    campaign_label_from_path = eda_utils.campaign_label_from_path
    compute_sfxi_metrics = eda_utils.compute_sfxi_metrics
    compute_sfxi_params = eda_utils.compute_sfxi_params
    coerce_selection_dataframe = eda_utils.coerce_selection_dataframe
    dedupe_columns = eda_utils.dedupe_columns
    dedupe_exprs = eda_utils.dedupe_exprs
    dedup_latest_labels = eda_utils.dedup_latest_labels
    find_latest_model_artifact = eda_utils.find_latest_model_artifact
    find_repo_root = eda_utils.find_repo_root
    fit_intensity_median_iqr = eda_utils.fit_intensity_median_iqr
    get_feature_importances = eda_utils.get_feature_importances
    invert_intensity_median_iqr = eda_utils.invert_intensity_median_iqr
    is_altair_undefined = eda_utils.is_altair_undefined
    list_campaign_paths = eda_utils.list_campaign_paths
    list_series_to_numpy = eda_utils.list_series_to_numpy
    list_usr_datasets = eda_utils.list_usr_datasets
    load_campaign_yaml = eda_utils.load_campaign_yaml
    load_intensity_params_from_round_ctx = eda_utils.load_intensity_params_from_round_ctx
    load_model_artifact = eda_utils.load_model_artifact
    missingness_summary = eda_utils.missingness_summary
    namespace_summary = eda_utils.namespace_summary
    parse_campaign_info = eda_utils.parse_campaign_info
    resolve_dataset_path = eda_utils.resolve_dataset_path
    resolve_usr_root = eda_utils.resolve_usr_root
    safe_is_numeric = eda_utils.safe_is_numeric
    unwrap_artifact_model = eda_utils.unwrap_artifact_model
    valid_vec8_mask_expr = eda_utils.valid_vec8_mask_expr
    return (
        apply_intensity_median_iqr,
        build_color_dropdown_options,
        build_label_events,
        campaign_label_from_path,
        coerce_selection_dataframe,
        compute_sfxi_metrics,
        compute_sfxi_params,
        dedup_latest_labels,
        dedupe_columns,
        dedupe_exprs,
        find_latest_model_artifact,
        find_repo_root,
        fit_intensity_median_iqr,
        get_feature_importances,
        invert_intensity_median_iqr,
        is_altair_undefined,
        list_campaign_paths,
        list_series_to_numpy,
        list_usr_datasets,
        load_campaign_yaml,
        load_intensity_params_from_round_ctx,
        load_model_artifact,
        missingness_summary,
        namespace_summary,
        parse_campaign_info,
        resolve_dataset_path,
        resolve_usr_root,
        safe_is_numeric,
        unwrap_artifact_model,
        valid_vec8_mask_expr,
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
    1. Pick a dataset (`records.parquet`) and explore the schema/missingness.
    2. Peruse the dataset or do some exploratory data analysis to learn its structure.
    3. Inspect a record: view its sequence and TFBS placement.
    4. Explore UMAP neighborhoods and brush select a working subset.
    5. Look at the SFXI metric; tweak the setpoint and watch sequence scores change.
    6. (Optional) Export a tabular subset for downstream analysis.
    """
        )
    )
    return


@app.cell
def _(Path, find_repo_root):
    notebook_path = Path(__file__).resolve()
    repo_root = find_repo_root(notebook_path)
    repo_root_error = None
    if repo_root is None:
        repo_root_error = "Could not find repo root (pyproject.toml). Provide an absolute path."
    return repo_root, repo_root_error


@app.cell
def _(list_usr_datasets, os, repo_root, resolve_usr_root):
    usr_root = None
    usr_root_error = None
    usr_root_mode = "default"
    usr_root_env = os.environ.get("DNADESIGN_USR_ROOT")

    try:
        usr_root = resolve_usr_root(repo_root, usr_root_env)
        if usr_root_env:
            usr_root_mode = "env"
    except ValueError as exc:
        usr_root_error = str(exc)

    usr_datasets = list_usr_datasets(usr_root) if usr_root else []
    return usr_datasets, usr_root, usr_root_error, usr_root_mode


@app.cell
def _(mo, usr_datasets):
    dataset_dropdown = mo.ui.dropdown(
        options=usr_datasets or ["(none found)"],
        value=usr_datasets[0] if usr_datasets else "(none found)",
        label="USR dataset",
        full_width=True,
    )
    custom_path_input = mo.ui.text(
        value="",
        label="Custom path override (absolute or repo-root-relative)",
        full_width=True,
    )
    dataset_form = mo.ui.form(
        mo.ui.array([dataset_dropdown, custom_path_input]),
        label="Load dataset",
        bordered=True,
    )
    return custom_path_input, dataset_dropdown, dataset_form


@app.cell
def _(
    custom_path_input,
    dataset_dropdown,
    dataset_form,
    mo,
    pl,
    repo_root,
    repo_root_error,
    resolve_dataset_path,
    usr_root,
):
    custom_path_text = custom_path_input.value.strip()
    if custom_path_text and dataset_form.value is None:
        mo.stop(True, mo.md("Submit the form to load the custom path."))
    _repo_root_error_md = mo.md(repo_root_error) if repo_root_error is not None else mo.md("")
    mo.stop(repo_root_error is not None, _repo_root_error_md)

    _dataset_choice = dataset_dropdown.value if dataset_dropdown is not None else None
    if _dataset_choice in {"(none found)", "(none)"}:
        _dataset_choice = None
    if not custom_path_text and not _dataset_choice:
        mo.stop(True, mo.md("No datasets found. Provide a custom path or set `DNADESIGN_USR_ROOT`."))

    try:
        dataset_path, dataset_mode = resolve_dataset_path(
            repo_root=repo_root,
            usr_root=usr_root,
            dataset_name=_dataset_choice,
            custom_path=custom_path_input.value,
        )
    except ValueError as exc:
        mo.stop(True, mo.md(f"Dataset path error: {exc}"))

    if not dataset_path.exists():
        hint = (
            "Did you mean `src/dnadesign/usr/datasets/<name>/records.parquet`?"
            if dataset_mode == "usr"
            else "Provide an absolute path or a repo-root-relative path."
        )
        mo.stop(
            True,
            mo.md(
                "\n".join(
                    [
                        "Dataset not found.",
                        f"Resolved path: `{dataset_path}`",
                        f"Mode: `{dataset_mode}`",
                        hint,
                    ]
                )
            ),
        )

    try:
        df_raw = pl.read_parquet(dataset_path)
    except Exception as exc:
        mo.stop(
            True,
            mo.md(
                "\n".join(
                    [
                        "Failed to read Parquet.",
                        f"Resolved path: `{dataset_path}`",
                        f"Error: {exc}",
                    ]
                )
            ),
        )

    dataset_name = dataset_path.parent.name if dataset_mode == "usr" else dataset_path.stem
    dataset_status_md = mo.md(
        "\n".join(
            [
                f"Dataset: `{dataset_name}`",
                f"Rows: `{df_raw.height}`",
                f"Columns: `{len(df_raw.columns)}`",
            ]
        )
    )
    return dataset_name, dataset_path, dataset_status_md, df_raw


@app.cell
def _(df_raw):
    if "__row_id" in df_raw.columns:
        df_prelim = df_raw
    else:
        df_prelim = df_raw.with_row_index("__row_id")
    return (df_prelim,)


@app.cell
def _(df_prelim, mo):
    required_cols = ["id", "cluster__ldn_v1__umap_x", "cluster__ldn_v1__umap_y"]
    missing = [col for col in required_cols if col not in df_prelim.columns]
    if missing:
        mo.stop(
            True,
            mo.md(
                "\n".join(
                    [
                        "Missing required columns for this notebook:",
                        ", ".join(f"`{col}`" for col in missing),
                        "Load a dataset that includes the expected identifiers and UMAP coordinates.",
                    ]
                )
            ),
        )
    return


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
def _(df_prelim, mo, pl):
    if df_prelim.is_empty():
        schema_df = pl.DataFrame({"column": [], "dtype": []})
    else:
        schema_df = pl.DataFrame(
            {
                "column": list(df_prelim.schema.keys()),
                "dtype": [str(dtype) for dtype in df_prelim.schema.values()],
            }
        )
    schema_table = mo.ui.table(schema_df, page_size=5)
    return (schema_table,)


@app.cell
def _(df_prelim, mo):
    dataset_table = mo.ui.table(df_prelim, page_size=5)
    return (dataset_table,)


@app.cell
def _(
    build_color_dropdown_options,
    df_prelim,
    df_umap_overlay,
    mo,
    safe_is_numeric,
):
    numeric_cols = [name for name, dtype in df_prelim.schema.items() if safe_is_numeric(dtype)]
    _preferred_x_exact = [
        "infer__evo2_7b",
        "infer_evo2_7b",
        "infer__evo2_7b__score",
    ]
    _preferred_y_exact = [
        "densegen__compression_ratio",
        "densegen_compression_ratio",
    ]

    def _pick_preferred(cols: list[str], exact: list[str], substrings: list[str]) -> str | None:
        if not cols:
            return None
        for name in exact:
            if name in cols:
                return name
        if substrings:
            for name in cols:
                lower = name.lower()
                if all(token in lower for token in substrings):
                    return name
        return None

    _x_default = _pick_preferred(numeric_cols, _preferred_x_exact, ["infer", "evo2", "7"])
    if _x_default is None:
        _x_default = numeric_cols[0] if numeric_cols else "(none)"

    _y_default = _pick_preferred(numeric_cols, _preferred_y_exact, ["compression", "ratio"])
    if _y_default is None:
        if len(numeric_cols) > 1:
            _y_default = numeric_cols[1]
        elif numeric_cols:
            _y_default = numeric_cols[0]
        else:
            _y_default = "(none)"

    plot_type_dropdown = mo.ui.dropdown(
        options=["scatter", "histogram"],
        value="scatter",
        label="Plot type",
    )
    x_dropdown = mo.ui.dropdown(
        options=numeric_cols or ["(none)"],
        value=_x_default,
        label="X column",
        full_width=True,
    )
    y_dropdown = mo.ui.dropdown(
        options=numeric_cols or ["(none)"],
        value=_y_default,
        label="Y column (scatter only)",
        full_width=True,
    )
    _extra_color_cols = [
        "opal__transient__score",
        "opal__transient__logic_fidelity",
        "opal__transient__effect_scaled",
        "opal__transient__rank",
        "opal__transient__observed_event",
        "opal__transient__top_k",
    ]
    _df_explorer_source = df_umap_overlay if df_umap_overlay is not None else df_prelim
    _color_options = build_color_dropdown_options(
        _df_explorer_source,
        extra=_extra_color_cols,
        include_none=True,
    )
    dataset_explorer_color_dropdown = mo.ui.dropdown(
        options=_color_options,
        value="(none)",
        label="Color by (scatter only)",
        full_width=True,
    )
    bins_slider = mo.ui.slider(5, 200, value=30, label="Histogram bins")
    return (
        bins_slider,
        dataset_explorer_color_dropdown,
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
    df_prelim,
    df_umap_overlay,
    mo,
    pl,
    plot_type_dropdown,
    safe_is_numeric,
    with_title,
    x_dropdown,
    y_dropdown,
):
    plot_type = plot_type_dropdown.value
    x_col = x_dropdown.value
    y_col = y_dropdown.value
    _note_lines = []
    if not hasattr(alt, "_DNAD_PLOT_SIZE"):
        alt._DNAD_PLOT_SIZE = 420
    _plot_size = alt._DNAD_PLOT_SIZE
    _df_explorer_source = df_umap_overlay if df_umap_overlay is not None else df_prelim

    _dataset_label = dataset_name or "dataset"
    _color_value = dataset_explorer_color_dropdown.value
    _color_encoding = alt.Undefined
    _color_title = None
    _color_tooltip = None
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
        if x_col not in _df_explorer_source.columns or y_col not in _df_explorer_source.columns:
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
            if y_col == x_col:
                y_col_plot = f"{y_col}__y"
                _select_exprs = [
                    pl.col("__row_id"),
                    pl.col("id"),
                    pl.col(x_col),
                    pl.col(y_col).alias(y_col_plot),
                ]
            else:
                y_col_plot = y_col
                _select_exprs = [
                    pl.col("__row_id"),
                    pl.col("id"),
                    pl.col(x_col),
                    pl.col(y_col),
                ]
            if (
                _color_value
                and _color_value != "(none)"
                and _color_value in _df_explorer_source.columns
                and _color_value not in {x_col, y_col}
            ):
                _select_exprs.append(pl.col(_color_value))
            df_plot = _df_explorer_source.select(dedupe_exprs(_select_exprs))
            if _color_value and _color_value != "(none)":
                if _color_value not in df_plot.columns:
                    _note_lines.append(f"Color `{_color_value}` unavailable; rendering without color.")
                else:
                    _dtype = _df_explorer_source.schema.get(_color_value)
                    _non_null_count = df_plot.select(pl.col(_color_value).count()).item() if df_plot.height else 0
                    _is_nested = False
                    if _dtype is not None:
                        try:
                            _is_nested = bool(getattr(_dtype, "is_nested")())
                        except Exception:
                            _is_nested = False
                    if _non_null_count == 0:
                        _note_lines.append(f"Color `{_color_value}` has no non-null values; rendering without color.")
                    elif _is_nested:
                        _note_lines.append(f"Color `{_color_value}` is nested; rendering without color.")
                    elif _color_value == "opal__transient__observed_event":
                        _label_col = f"{_color_value}__label"
                        df_plot = df_plot.with_columns(
                            pl.when(pl.col(_color_value))
                            .then(pl.lit("Observed"))
                            .otherwise(pl.lit("Not observed"))
                            .alias(_label_col)
                        )
                        _color_title = "Observed events (ingest_y)"
                        _color_tooltip = _label_col
                        _color_encoding = alt.Color(
                            f"{_label_col}:N",
                            title=_color_title,
                            scale=alt.Scale(
                                domain=["Observed", "Not observed"],
                                range=[_okabe_ito[2], "#B0B0B0"],
                            ),
                            legend=alt.Legend(title=_color_title),
                        )
                    elif _color_value == "opal__transient__top_k":
                        _label_col = f"{_color_value}__label"
                        df_plot = df_plot.with_columns(
                            pl.when(pl.col(_color_value))
                            .then(pl.lit("Top-K"))
                            .otherwise(pl.lit("Not Top-K"))
                            .alias(_label_col)
                        )
                        _color_title = "Transient Top-K"
                        _color_tooltip = _label_col
                        _color_encoding = alt.Color(
                            f"{_label_col}:N",
                            title=_color_title,
                            scale=alt.Scale(
                                domain=["Top-K", "Not Top-K"],
                                range=[_okabe_ito[5], "#B0B0B0"],
                            ),
                            legend=alt.Legend(title=_color_title),
                        )
                    elif _dtype is not None and safe_is_numeric(_dtype):
                        _color_title = _color_value
                        _color_tooltip = _color_value
                        _color_encoding = alt.Color(
                            f"{_color_value}:Q",
                            title=_color_title,
                            legend=alt.Legend(title=_color_title, format=".2f", tickCount=5),
                        )
                    else:
                        _n_unique = df_plot.select(pl.col(_color_value).n_unique()).item() if df_plot.height else 0
                        _color_scale = (
                            alt.Scale(range=_okabe_ito)
                            if _n_unique <= len(_okabe_ito)
                            else alt.Scale(scheme=_fallback_scheme)
                        )
                        _color_title = _color_value
                        _color_tooltip = _color_value
                        _color_encoding = alt.Color(
                            f"{_color_value}:N",
                            title=_color_title,
                            scale=_color_scale,
                            legend=alt.Legend(title=_color_title),
                        )
            _brush = alt.selection_interval(name="explorer_brush", encodings=["x", "y"])
            _tooltip_cols = [c for c in ["id", "__row_id", x_col, y_col_plot] if c in df_plot.columns]
            if _color_tooltip and _color_tooltip in df_plot.columns and _color_tooltip not in _tooltip_cols:
                _tooltip_cols.append(_color_tooltip)
            chart = (
                alt.Chart(df_plot)
                .mark_circle(opacity=0.7, stroke=None, strokeWidth=0)
                .encode(
                    x=alt.X(x_col, title=x_col),
                    y=alt.Y(y_col_plot, title=y_col),
                    color=_color_encoding,
                    tooltip=_tooltip_cols,
                )
                .add_params(_brush)
            )
    else:
        if x_col not in _df_explorer_source.columns:
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
            df_plot = _df_explorer_source.select(dedupe_exprs([pl.col("__row_id"), pl.col("id"), pl.col(x_col)]))
            chart = (
                alt.Chart(df_plot)
                .mark_bar()
                .encode(
                    x=alt.X(
                        x_col,
                        bin=alt.Bin(maxbins=int(bins_slider.value)),
                        title=x_col,
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
    dataset_explorer = mo.vstack(
        [
            plot_type_dropdown,
            mo.hstack([x_dropdown, y_dropdown]),
            dataset_explorer_color_dropdown,
            bins_slider,
            note_md,
            dataset_explorer_chart_ui,
        ]
    )
    return dataset_explorer, dataset_explorer_chart_ui


@app.cell
def _(
    coerce_selection_dataframe,
    dataset_explorer_chart_ui,
    df_prelim,
    df_umap_overlay,
    is_altair_undefined,
    mo,
    pl,
    x_dropdown,
    y_dropdown,
):
    _unused = (x_dropdown, y_dropdown)
    _df_explorer_source = df_umap_overlay if df_umap_overlay is not None else df_prelim
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
def _(df_prelim):
    df_active = df_prelim
    return (df_active,)


@app.cell
def _(df_active, mo):
    df_active_note_md = mo.md(f"Active rows: `{df_active.height}`. Summaries reflect the loaded dataset.")
    return (df_active_note_md,)


@app.cell
def _(df_active, mo, opal_campaign_info, pl, safe_is_numeric):
    metric_cols = [name for name, dtype in df_active.schema.items() if safe_is_numeric(dtype) and dtype != pl.Boolean]
    slug = opal_campaign_info.slug if opal_campaign_info is not None else None
    preferred_cols = []
    if slug:
        preferred_cols.append(f"opal__{slug}__latest_pred_scalar")
    preferred_cols.append("opal__transient__score")
    preferred_cols.append("opal__transient__logic_fidelity")
    preferred_cols.append("opal__transient__effect_scaled")
    preferred_cols.append("opal__transient__rank")
    for _col in preferred_cols:
        if _col not in metric_cols:
            metric_cols.append(_col)
    if not metric_cols:
        metric_cols = ["opal__transient__score"]
    default_metric = metric_cols[0]
    if slug and f"opal__{slug}__latest_pred_scalar" in metric_cols:
        default_metric = f"opal__{slug}__latest_pred_scalar"
    elif "opal__transient__score" in metric_cols:
        default_metric = "opal__transient__score"
    transient_cluster_metric_dropdown = mo.ui.dropdown(
        options=metric_cols,
        value=default_metric,
        label="Cluster plot metric",
        full_width=True,
    )
    hue_options = [
        "(none)",
        "Leiden cluster",
        "opal__transient__top_k",
        "opal__transient__observed_event",
        "opal__transient__sfxi_scored_label",
    ]
    transient_cluster_hue_dropdown = mo.ui.dropdown(
        options=hue_options,
        value="Leiden cluster",
        label="Color by (hue)",
        full_width=True,
    )
    return transient_cluster_hue_dropdown, transient_cluster_metric_dropdown


@app.cell
def _(mo):
    rf_model_source = mo.ui.radio(
        options=[
            "Ephemeral (refit in notebook)",
            "OPAL artifact (model.joblib)",
        ],
        value="Ephemeral (refit in notebook)",
        label="RF model source",
    )
    return (rf_model_source,)


@app.cell(column=1)
def _(
    dataset_form,
    dataset_preview_md,
    mo,
    usr_root,
    usr_root_error,
    usr_root_mode,
):
    header = mo.md("## Dataset selection")
    intro = mo.md(
        "Choose which `records.parquet` to explore. The dataset is the source of truth for DenseGen "
        "metadata, TFBS annotations, Evo2 embeddings, and OPAL/SFXI fields used downstream."
    )
    usr_status = mo.md("")
    if usr_root_error:
        usr_status = mo.md(f"USR root error: {usr_root_error}")
    elif usr_root:
        usr_status = mo.md(f"USR root ({usr_root_mode}): `{usr_root}`")
    mo.vstack([header, intro, usr_status, dataset_form, dataset_preview_md])
    return


@app.cell
def _(missingness_table, mo, namespace_table, schema_table):
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
            mo.md(
                "**Schema & dtypes**  \n"
                "Dtypes matter for plots and filters. (e.g., which fields are numeric, categorical, or nested)."
            ),
            schema_table,
        ]
    )
    return


@app.cell
def _(build_color_dropdown_options, df_active, mo):
    default_x = None
    default_y = None
    for name in df_active.columns:
        if name.endswith("__umap_x"):
            default_x = name
            break
    for name in df_active.columns:
        if name.endswith("__umap_y"):
            default_y = name
            break
    if "cluster__ldn_v1__umap_x" in df_active.columns:
        default_x = "cluster__ldn_v1__umap_x"
    if "cluster__ldn_v1__umap_y" in df_active.columns:
        default_y = "cluster__ldn_v1__umap_y"

    umap_x_input = mo.ui.text(
        value=default_x or "",
        label="UMAP X column",
        full_width=True,
    )
    umap_y_input = mo.ui.text(
        value=default_y or "",
        label="UMAP Y column",
        full_width=True,
    )
    extra_color_options = [
        "opal__transient__score",
        "opal__transient__logic_fidelity",
        "opal__transient__effect_scaled",
        "opal__transient__observed_event",
        "opal__transient__top_k",
    ]
    umap_color_cols = build_color_dropdown_options(
        df_active,
        extra=extra_color_options,
        include_none=False,
    )
    umap_color_default = "cluster__ldn_v1" if "cluster__ldn_v1" in umap_color_cols else "(none)"
    umap_color_dropdown = mo.ui.dropdown(
        options=["(none)"] + umap_color_cols,
        value=umap_color_default,
        label="Color by",
        full_width=True,
    )
    umap_size_slider = mo.ui.slider(5, 200, value=30, label="Point size")
    umap_opacity_slider = mo.ui.slider(0.1, 1.0, value=0.7, label="Opacity", step=0.05)
    return (
        umap_color_dropdown,
        umap_opacity_slider,
        umap_size_slider,
        umap_x_input,
        umap_y_input,
    )


@app.cell
def _(alt, pl):
    _unused = (pl,)
    if not hasattr(alt, "_DNAD_PLOT_SIZE"):
        alt._DNAD_PLOT_SIZE = 420
    return


@app.cell
def _(
    alt,
    dataset_name,
    dedupe_columns,
    df_active,
    df_umap_overlay,
    mo,
    pl,
    safe_is_numeric,
    umap_color_dropdown,
    umap_opacity_slider,
    umap_size_slider,
    umap_x_input,
    umap_y_input,
    with_title,
):
    _x_col = umap_x_input.value.strip()
    _y_col = umap_y_input.value.strip()
    umap_valid = False
    _x_name = _x_col or "umap_x"
    _y_name = _y_col or "umap_y"
    umap_note = None
    if not hasattr(alt, "_DNAD_PLOT_SIZE"):
        alt._DNAD_PLOT_SIZE = 420
    _plot_size = alt._DNAD_PLOT_SIZE
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

    df_umap_source = df_umap_overlay if df_umap_overlay is not None else df_active
    _color_title = None

    if "id" not in df_umap_source.columns:
        umap_note = mo.md("UMAP missing: required column `id` is absent.")
    elif not _x_col or not _y_col:
        umap_note = mo.md(
            "UMAP missing: provide x/y columns. "
            "To attach coords: `uv run cluster umap --dataset <dataset> "
            "--name ldn_v1 --attach-coords --write --allow-overwrite`"
        )
    elif _x_col not in df_umap_source.columns or _y_col not in df_umap_source.columns:
        umap_note = mo.md(
            "UMAP missing: x/y columns must exist. "
            "To attach coords: `uv run cluster umap --dataset <dataset> "
            "--name ldn_v1 --attach-coords --write --allow-overwrite`"
        )
    elif not (safe_is_numeric(df_umap_source.schema[_x_col]) and safe_is_numeric(df_umap_source.schema[_y_col])):
        umap_note = mo.md(
            "UMAP missing: x/y columns must be numeric. "
            "To attach coords: `uv run cluster umap --dataset <dataset> "
            "--name ldn_v1 --attach-coords --write --allow-overwrite`"
        )
    else:
        umap_valid = True

    df_umap_plot = df_umap_source
    if not umap_valid:
        df_umap_chart = pl.DataFrame(
            schema={
                "__row_id": pl.Int64,
                "id": pl.Utf8,
                _x_name: pl.Float64,
                _y_name: pl.Float64,
            }
        )
        _chart = (
            alt.Chart(df_umap_chart)
            .mark_circle(stroke=None, strokeWidth=0)
            .encode(
                x=alt.X(_x_name),
                y=alt.Y(_y_name),
                tooltip=[c for c in ["id", "__row_id", _x_name, _y_name] if c in df_umap_chart.columns],
            )
            .properties(width=_plot_size, height=_plot_size)
        )
        umap_chart_note_md = umap_note
    else:
        umap_chart_note_md = mo.md(f"Plotting full dataset: `{df_umap_plot.height}` points.")

        _color_value = umap_color_dropdown.value
        _color_encoding = alt.Undefined
        _color_field = None
        _color_title = None
        _color_tooltip = None
        plot_cols = dedupe_columns([col for col in ["__row_id", "id", _x_col, _y_col] if col in df_umap_source.columns])
        if _color_value and _color_value != "(none)" and _color_value in df_umap_source.columns:
            if _color_value not in plot_cols:
                plot_cols.append(_color_value)
        df_umap_chart = df_umap_source.select(plot_cols)

        if _color_value and _color_value != "(none)" and _color_value in df_umap_chart.columns:
            _dtype = df_umap_source.schema[_color_value]
            _color_field = _color_value
            _color_title = _color_value
            _color_tooltip = _color_value
            _non_null_count = df_umap_chart.select(pl.col(_color_value).count()).item() if df_umap_chart.height else 0
            if _non_null_count == 0:
                umap_chart_note_md = mo.md(
                    f"Plotting full dataset: `{df_umap_plot.height}` points. "
                    f"Color `{_color_value}` has no non-null values; rendering without color."
                )
                _color_field = None
                _color_title = None
                _color_tooltip = None
                _color_encoding = alt.Undefined
            elif _color_value == "opal__transient__observed_event":
                _label_col = f"{_color_value}__label"
                df_umap_chart = df_umap_chart.with_columns(
                    pl.when(pl.col(_color_value))
                    .then(pl.lit("Observed"))
                    .otherwise(pl.lit("Not observed"))
                    .alias(_label_col)
                )
                _color_field = _label_col
                _color_title = "Observed events (ingest_y)"
                _color_tooltip = _label_col
                _color_scale = alt.Scale(
                    domain=["Observed", "Not observed"],
                    range=[_okabe_ito[2], "#B0B0B0"],
                )
                _color_encoding = alt.Color(
                    f"{_color_field}:N",
                    title=_color_title,
                    scale=_color_scale,
                    legend=alt.Legend(title=_color_title),
                )
            elif _color_value == "opal__transient__top_k":
                _label_col = f"{_color_value}__label"
                df_umap_chart = df_umap_chart.with_columns(
                    pl.when(pl.col(_color_value)).then(pl.lit("Top-K")).otherwise(pl.lit("Not Top-K")).alias(_label_col)
                )
                _color_field = _label_col
                _color_title = "Transient Top-K"
                _color_tooltip = _label_col
                _color_scale = alt.Scale(
                    domain=["Top-K", "Not Top-K"],
                    range=[_okabe_ito[5], "#B0B0B0"],
                )
                _color_encoding = alt.Color(
                    f"{_color_field}:N",
                    title=_color_title,
                    scale=_color_scale,
                    legend=alt.Legend(title=_color_title),
                )
            elif safe_is_numeric(_dtype):
                _color_encoding = alt.Color(
                    f"{_color_field}:Q",
                    title=_color_title,
                    legend=alt.Legend(title=_color_title, format=".2f", tickCount=5),
                )
            else:
                _n_unique = df_umap_chart.select(pl.col(_color_value).n_unique()).item() if df_umap_chart.height else 0
                if _n_unique <= len(_okabe_ito):
                    _color_scale = alt.Scale(range=_okabe_ito)
                else:
                    _color_scale = alt.Scale(scheme=_fallback_scheme)
                _color_encoding = alt.Color(
                    f"{_color_field}:N",
                    title=_color_title,
                    scale=_color_scale,
                    legend=alt.Legend(title=_color_title),
                )

        _brush = alt.selection_interval(name="umap_brush", encodings=["x", "y"])
        _tooltip_cols = [c for c in ["id", "__row_id", _x_col, _y_col] if c in df_umap_chart.columns]
        if _color_tooltip and _color_tooltip in df_umap_chart.columns and _color_tooltip not in _tooltip_cols:
            _tooltip_cols.append(_color_tooltip)
        _chart = (
            alt.Chart(df_umap_chart)
            .mark_circle(
                size=umap_size_slider.value,
                opacity=umap_opacity_slider.value,
                stroke=None,
                strokeWidth=0,
            )
            .encode(
                x=alt.X(_x_col, title=_x_col),
                y=alt.Y(_y_col, title=_y_col),
                color=_color_encoding,
                tooltip=_tooltip_cols,
            )
            .add_params(_brush)
            .properties(width=_plot_size, height=_plot_size)
        )
        if "opal__transient__top_k" in df_umap_chart.columns:
            _df_top = df_umap_chart.filter(pl.col("opal__transient__top_k"))
            if _df_top.height:
                _top_layer = (
                    alt.Chart(_df_top)
                    .mark_circle(
                        size=umap_size_slider.value * 1.8,
                        stroke="#000000",
                        strokeWidth=1.5,
                        fillOpacity=0.0,
                        opacity=1.0,
                    )
                    .encode(
                        x=alt.X(_x_col, title=_x_col),
                        y=alt.Y(_y_col, title=_y_col),
                        tooltip=_tooltip_cols,
                    )
                )
                _chart = _chart + _top_layer

    _color_context = _color_title or "none"
    _subtitle = f"{dataset_name or 'dataset'} · color={_color_context}"
    _chart = with_title(_chart, "UMAP explorer (Evo2 embedding)", _subtitle)

    umap_chart_ui = mo.ui.altair_chart(_chart)
    return df_umap_plot, umap_chart_note_md, umap_chart_ui, umap_valid


@app.cell
def _(
    coerce_selection_dataframe,
    df_umap_plot,
    is_altair_undefined,
    mo,
    pl,
    umap_chart_ui,
    umap_valid,
    umap_x_input,
    umap_y_input,
):
    _unused = (umap_x_input, umap_y_input)
    _selected_raw = umap_chart_ui.value

    if not umap_valid:
        df_umap_selected = df_umap_plot.head(0)
        umap_selection_note_md = mo.md("UMAP missing; selection disabled.")
    else:
        if _selected_raw is None or is_altair_undefined(_selected_raw):
            df_umap_selected = df_umap_plot.head(0)
            umap_selection_note_md = mo.md("No points selected.")
        else:
            _selected_df = coerce_selection_dataframe(_selected_raw)

            if _selected_df is None or "__row_id" not in _selected_df.columns:
                df_umap_selected = df_umap_plot.head(0)
                umap_selection_note_md = mo.md("Selection missing row ids.")
            else:
                _selected_ids = _selected_df.select(pl.col("__row_id").drop_nulls().unique()).to_series().to_list()
                if not _selected_ids:
                    df_umap_selected = df_umap_plot.head(0)
                    umap_selection_note_md = mo.md("No points selected.")
                else:
                    df_umap_selected = df_umap_plot.filter(pl.col("__row_id").is_in(_selected_ids))
                    umap_selection_note_md = mo.md(f"Selected rows: `{df_umap_selected.height}`")

    umap_selected_explorer = mo.ui.table(df_umap_selected)
    return df_umap_selected, umap_selected_explorer, umap_selection_note_md


@app.cell
def _(
    export_button,
    export_format_dropdown,
    export_source_dropdown,
    export_status_md,
    mo,
):
    _export_label = export_source_dropdown.value
    export_note_md = mo.md(
        "## Export a dataframe\nDestination: `src/dnadesign/opal/notebooks/_outputs/promoter_eda_export.<format>`"
    )
    mo.vstack(
        [
            export_note_md,
            export_source_dropdown,
            export_format_dropdown,
            export_button,
            export_status_md,
        ]
    )
    return


@app.cell
def _(
    custom_path_input,
    dataset_dropdown,
    mo,
    repo_root,
    resolve_dataset_path,
    usr_root,
):
    preview_exists = False
    preview_error = None
    preview_mode = None
    preview_path = None

    _dataset_choice = dataset_dropdown.value if dataset_dropdown is not None else None
    if _dataset_choice in {"(none found)", "(none)"}:
        _dataset_choice = None

    try:
        preview_path, preview_mode = resolve_dataset_path(
            repo_root=repo_root,
            usr_root=usr_root,
            dataset_name=_dataset_choice,
            custom_path=custom_path_input.value,
        )
        preview_exists = preview_path.exists()
    except ValueError as exc:
        preview_error = str(exc)

    status_lines = []
    if preview_path is not None:
        status_lines.append(f"Mode: `{preview_mode}`")
        status_lines.append(f"Exists: `{preview_exists}`")
    if preview_error:
        status_lines.append(f"Error: {preview_error}")

    dataset_preview_md = mo.md("\n".join(status_lines) if status_lines else "")
    return (dataset_preview_md,)


@app.cell
def _(
    RandomForestModel,
    alt,
    apply_intensity_median_iqr,
    compute_sfxi_metrics,
    compute_sfxi_params,
    dataset_name,
    dedupe_columns,
    df_active,
    find_latest_model_artifact,
    fit_intensity_median_iqr,
    get_feature_importances,
    invert_intensity_median_iqr,
    list_series_to_numpy,
    load_intensity_params_from_round_ctx,
    load_model_artifact,
    math,
    mo,
    np,
    opal_campaign_info,
    opal_labels_asof_df,
    opal_labels_current_df,
    opal_labels_view_df,
    opal_selected_round,
    pl,
    rf_model_source,
    sfxi_beta_input,
    sfxi_fixed_params,
    sfxi_gamma_input,
    sfxi_p00_slider,
    sfxi_p01_slider,
    sfxi_p10_slider,
    sfxi_p11_slider,
    unwrap_artifact_model,
    valid_vec8_mask_expr,
    with_title,
):
    df_sfxi = df_active.head(0)
    df_umap_overlay = df_active
    sfxi_meta_md = mo.md("")
    sfxi_notice_md = mo.md("")
    transient_lines = []
    transient_feature_chart = None
    rf_model_source_note_md = mo.md("Ephemeral RF (session-scoped) · refit on UI changes.")

    labels_ready = True
    if opal_campaign_info is None:
        sfxi_notice_md = mo.md("Campaign config unavailable; SFXI disabled.")
        labels_ready = False
    elif opal_campaign_info.objective_name != "sfxi_v1":
        sfxi_notice_md = mo.md(f"Objective `{opal_campaign_info.objective_name}` is not supported here.")
        labels_ready = False
    elif opal_campaign_info.y_expected_length not in (None, 8):
        sfxi_notice_md = mo.md("SFXI expects 8-length label vectors; campaign uses a different length.")
        labels_ready = False

    _rf_source_value = rf_model_source.value if rf_model_source is not None else "Ephemeral (refit in notebook)"
    _use_artifact = _rf_source_value == "OPAL artifact (model.joblib)"
    _artifact_model = None
    _artifact_model_path = None
    _artifact_round_dir = None
    if _use_artifact and opal_campaign_info is None:
        _use_artifact = False
        rf_model_source_note_md = mo.md("No campaign selected; using ephemeral RF.")
    elif _use_artifact:
        try:
            _artifact_model_path, _artifact_round_dir = find_latest_model_artifact(opal_campaign_info)
        except Exception as exc:
            _use_artifact = False
            rf_model_source_note_md = mo.md(f"Artifact lookup failed: {exc}; using ephemeral RF.")
        else:
            if _artifact_model_path is None:
                _use_artifact = False
                rf_model_source_note_md = mo.md("No model.joblib found for this campaign; using ephemeral RF.")
            else:
                _artifact_obj, _artifact_err = load_model_artifact(_artifact_model_path)
                _artifact_model = unwrap_artifact_model(_artifact_obj)
                if _artifact_model is None:
                    _use_artifact = False
                    _err_msg = _artifact_err or "Unsupported artifact format"
                    rf_model_source_note_md = mo.md(f"Artifact load failed: {_err_msg}; using ephemeral RF.")
                else:
                    rf_model_source_note_md = mo.md(f"Loaded artifact: `{_artifact_model_path}`")
    _y_col = opal_campaign_info.y_column if opal_campaign_info is not None else None
    _x_col = opal_campaign_info.x_column if opal_campaign_info is not None else None

    delta = float(sfxi_fixed_params.get("delta", 0.0))
    p = float(sfxi_fixed_params.get("percentile", 95.0))
    fallback_p = float(sfxi_fixed_params.get("fallback_percentile", p))
    min_n = int(sfxi_fixed_params.get("min_n", 5))
    eps = float(sfxi_fixed_params.get("eps", 1.0e-8))

    params = compute_sfxi_params(
        setpoint=[
            sfxi_p00_slider.value,
            sfxi_p10_slider.value,
            sfxi_p01_slider.value,
            sfxi_p11_slider.value,
        ],
        beta=sfxi_beta_input.value,
        gamma=sfxi_gamma_input.value,
        delta=delta,
        p=p,
        fallback_p=fallback_p,
        min_n=min_n,
        eps=eps,
    )

    if labels_ready:
        if opal_selected_round is None:
            sfxi_notice_md = mo.md("No label rounds available for the selected campaign.")
        elif opal_labels_view_df.is_empty():
            sfxi_notice_md = mo.md("No label events available for the selected filters.")
        elif _y_col is None or _y_col not in opal_labels_view_df.columns:
            sfxi_notice_md = mo.md(f"Missing SFXI labels: `{_y_col}` not found.")
        else:
            sfxi_result = compute_sfxi_metrics(
                df=opal_labels_view_df,
                vec_col=_y_col,
                params=params,
                denom_pool_df=opal_labels_current_df,
            )
            df_sfxi = sfxi_result.df
            if sfxi_result.pool_size < params.min_n:
                sfxi_notice_md = mo.md(
                    f"Insufficient labels in current round for scaling; denom source: `{sfxi_result.denom_source}`."
                )
            elif df_sfxi.is_empty():
                sfxi_notice_md = mo.md("No valid SFXI vectors after filtering.")

            df_sfxi_table = df_sfxi
            _y_display_col = _y_col
            if _y_col and _y_col in df_sfxi.columns:
                _preview_col = f"{_y_col}_preview"

                def _preview_vec(val) -> str | None:
                    if val is None:
                        return None
                    text = str(val)
                    if len(text) > 120:
                        return text[:117] + "..."
                    return text

                df_sfxi_table = df_sfxi.with_columns(
                    pl.col(_y_col).map_elements(_preview_vec, return_dtype=pl.Utf8).alias(_preview_col)
                )
                _y_display_col = _preview_col

            sfxi_table_cols = [
                col
                for col in [
                    "__row_id",
                    "id",
                    "observed_round",
                    "label_src",
                    "logic_fidelity",
                    "effect_scaled",
                    "score",
                    _y_display_col,
                ]
                if col and col in df_sfxi_table.columns
            ]
            sfxi_labels_table = (
                mo.ui.table(df_sfxi_table.select(sfxi_table_cols), page_size=5)
                if sfxi_table_cols
                else mo.ui.table(df_sfxi_table, page_size=5)
            )
            sfxi_meta_md = mo.vstack(
                [
                    mo.md("Labels with computed SFXI metrics"),
                    sfxi_labels_table,
                ]
            )

    transient_cols = [
        "opal__transient__score",
        "opal__transient__rank",
        "opal__transient__logic_fidelity",
        "opal__transient__effect_scaled",
        "opal__transient__top_k",
    ]
    transient_hist_chart = None

    if not labels_ready:
        transient_lines.append("Transient predictions unavailable (campaign unsupported).")
    elif _x_col is None or _x_col not in df_active.columns:
        transient_lines.append(f"Missing X column: `{_x_col}`.")
    elif opal_labels_asof_df.is_empty():
        transient_lines.append("No labels available for transient model training.")
    elif _y_col is None or _y_col not in opal_labels_asof_df.columns:
        transient_lines.append(f"Missing label column `{_y_col}` for training.")
    else:
        df_train = opal_labels_asof_df.filter(pl.col(_x_col).is_not_null() & pl.col(_y_col).is_not_null())
        df_train = df_train.filter(pl.col(_x_col).list.len() > 0)
        df_train = df_train.filter(valid_vec8_mask_expr(_y_col))
        if df_train.is_empty():
            transient_lines.append("No valid training labels after filtering.")
        else:
            _len_stats = df_train.select(
                pl.col(_x_col).list.len().min().alias("min_len"),
                pl.col(_x_col).list.len().max().alias("max_len"),
            )
            _min_len, _max_len = _len_stats.row(0)
            if _min_len is None or _max_len is None or int(_min_len) != int(_max_len):
                transient_lines.append("X vectors must be fixed-length for transient RF training.")
            else:
                x_dim = int(_min_len)
                x_train = list_series_to_numpy(
                    df_train.select(pl.col(_x_col)).to_series(),
                    expected_len=x_dim,
                )
                y_train = list_series_to_numpy(
                    df_train.select(pl.col(_y_col)).to_series(),
                    expected_len=8,
                )
                if x_train is None or y_train is None or y_train.size == 0:
                    transient_lines.append("Unable to build training arrays from label data.")
                else:
                    intensity_params = None
                    for op in opal_campaign_info.y_ops:
                        if op.get("name") == "intensity_median_iqr":
                            intensity_params = dict(op.get("params") or {})
                            break
                    y_train_scaled = y_train
                    y_iqr_eps = float(eps)
                    y_med = np.zeros(4, dtype=float)
                    y_iqr = np.ones(4, dtype=float)
                    y_scale_enabled = False
                    if _use_artifact and _artifact_model is not None:
                        _loaded_params = None
                        if _artifact_round_dir is not None:
                            _loaded_params = load_intensity_params_from_round_ctx(_artifact_round_dir, eps_default=eps)
                        if _loaded_params is not None:
                            y_med, y_iqr, y_scale_enabled, y_iqr_eps = _loaded_params
                        elif intensity_params is not None:
                            y_iqr_eps = float(intensity_params.get("eps", eps))
                            _min_labels = int(intensity_params.get("min_labels", min_n))
                            try:
                                y_med, y_iqr, y_scale_enabled = fit_intensity_median_iqr(
                                    y_train,
                                    min_labels=_min_labels,
                                    eps=y_iqr_eps,
                                )
                            except Exception as exc:
                                transient_lines.append(f"Y-op intensity_median_iqr failed: {exc}")
                                y_scale_enabled = False
                        model = _artifact_model
                        model_ready = model is not None
                        if model_ready:
                            transient_lines.append("Using OPAL artifact model for predictions.")
                    else:
                        if intensity_params is not None:
                            y_iqr_eps = float(intensity_params.get("eps", eps))
                            _min_labels = int(intensity_params.get("min_labels", min_n))
                            try:
                                y_med, y_iqr, y_scale_enabled = fit_intensity_median_iqr(
                                    y_train,
                                    min_labels=_min_labels,
                                    eps=y_iqr_eps,
                                )
                                y_train_scaled = apply_intensity_median_iqr(
                                    y_train,
                                    y_med,
                                    y_iqr,
                                    eps=y_iqr_eps,
                                    enabled=y_scale_enabled,
                                )
                            except Exception as exc:
                                transient_lines.append(f"Y-op intensity_median_iqr failed: {exc}")
                                y_train_scaled = y_train
                                y_scale_enabled = False

                        transient_lines.append(
                            f"Transient RF: training started (n_labels={df_train.height}, x_dim={x_train.shape[1]})"
                        )
                        model = RandomForestModel(params=opal_campaign_info.model_params)
                        model_ready = False
                        try:
                            fit_metrics = model.fit(x_train, y_train_scaled)
                        except Exception as exc:
                            transient_lines.append(f"Model fit failed: {exc}")
                        else:
                            model_ready = True
                            if fit_metrics is not None:
                                transient_lines.append(
                                    "Transient RF fit metrics: "
                                    f"oob_r2={fit_metrics.oob_r2}, oob_mse={fit_metrics.oob_mse}"
                                )

                    if model_ready:
                        _feature_importances = get_feature_importances(model)
                        if _feature_importances is not None and np.size(_feature_importances):
                            _feature_count = int(np.size(_feature_importances))
                            df_importance = pl.DataFrame(
                                {
                                    "feature_idx": list(range(_feature_count)),
                                    "importance": np.asarray(_feature_importances, dtype=float),
                                }
                            )
                            if df_importance.height:
                                _rf_params = dict(opal_campaign_info.model_params or {})
                                _n_estimators = _rf_params.get("n_estimators", "default")
                                _max_depth = _rf_params.get("max_depth", "default")
                                _df_sorted = df_importance.sort("feature_idx")
                                _max_ticks = 40
                                _stride = max(1, math.ceil(_feature_count / _max_ticks))
                                _axis_values = list(range(0, _feature_count, _stride))
                                if _axis_values and _axis_values[-1] != _feature_count - 1:
                                    _axis_values.append(_feature_count - 1)
                                _subtitle = (
                                    f"{dataset_name or 'dataset'} · round={opal_selected_round} · "
                                    f"n_labels={df_train.height} · x_dim={x_train.shape[1]} · "
                                    f"n_features={_feature_count} · n_estimators={_n_estimators} · "
                                    f"max_depth={_max_depth}"
                                )
                                _imp_chart = (
                                    alt.Chart(_df_sorted)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X(
                                            "feature_idx:O",
                                            sort=alt.SortField(field="feature_idx", order="ascending"),
                                            axis=alt.Axis(
                                                values=_axis_values,
                                                title="Feature index",
                                                labelFontSize=8,
                                                labelAngle=0,
                                            ),
                                        ),
                                        y=alt.Y("importance:Q", title="Importance"),
                                        tooltip=["feature_idx", "importance"],
                                    )
                                )
                                _imp_chart = (
                                    with_title(_imp_chart, "Random Forest feature importance", _subtitle)
                                    .properties(
                                        width="container",
                                        height=240,
                                        autosize={"type": "fit", "contains": "padding"},
                                    )
                                    .configure_view(stroke=None)
                                )
                                transient_feature_chart = mo.ui.altair_chart(_imp_chart)

                        transient_lines.append("Transient RF: predicting over full pool ...")
                        df_x_all = df_active.select(dedupe_columns(["__row_id", "id", _x_col]))
                        df_x_valid = df_x_all.filter(
                            pl.col(_x_col).is_not_null() & (pl.col(_x_col).list.len() == x_dim)
                        )
                        if df_x_valid.is_empty():
                            transient_lines.append("No candidate X vectors available for prediction.")
                        else:
                            chunk_size = 50_000
                            pred_chunks = []
                            failed_chunk = False
                            for start in range(0, df_x_valid.height, chunk_size):
                                df_chunk = df_x_valid.slice(start, chunk_size)
                                x_chunk = list_series_to_numpy(
                                    df_chunk.select(pl.col(_x_col)).to_series(),
                                    expected_len=x_dim,
                                )
                                if x_chunk is None:
                                    failed_chunk = True
                                    break
                                try:
                                    pred_chunks.append(model.predict(x_chunk))
                                except Exception as exc:
                                    transient_lines.append(f"Model predict failed: {exc}")
                                    failed_chunk = True
                                    break
                            if failed_chunk or not pred_chunks:
                                transient_lines.append("Unable to build feature matrix for full pool.")
                            else:
                                y_pred = np.vstack(pred_chunks)
                                if y_scale_enabled:
                                    y_pred = invert_intensity_median_iqr(y_pred, y_med, y_iqr, enabled=y_scale_enabled)
                                df_pred = df_x_valid.with_columns(pl.Series("opal__transient__y_vec", y_pred.tolist()))
                                denom_pool = opal_labels_current_df.select(
                                    pl.col(_y_col).alias("opal__transient__y_vec")
                                )
                                pred_result = compute_sfxi_metrics(
                                    df=df_pred,
                                    vec_col="opal__transient__y_vec",
                                    params=params,
                                    denom_pool_df=denom_pool,
                                )
                                df_pred_scored = pred_result.df
                                if df_pred_scored.is_empty():
                                    transient_lines.append("No valid predictions after SFXI scoring.")
                                else:
                                    df_pred_scored = df_pred_scored.with_columns(
                                        pl.col("logic_fidelity")
                                        .alias("opal__transient__logic_fidelity")
                                        .cast(pl.Float64),
                                        pl.col("effect_scaled")
                                        .alias("opal__transient__effect_scaled")
                                        .cast(pl.Float64),
                                        pl.col("score").alias("opal__transient__score").cast(pl.Float64),
                                    )
                                    try:
                                        top_k = int(opal_campaign_info.selection_params.get("top_k", 10))
                                    except Exception:
                                        top_k = 10
                                        transient_lines.append("Invalid top_k in campaign config; using 10.")
                                    df_pred_scored = df_pred_scored.with_columns(
                                        pl.col("opal__transient__score")
                                        .rank(method="dense", descending=True)
                                        .alias("opal__transient__rank")
                                    ).with_columns(
                                        (pl.col("opal__transient__rank") <= top_k).alias("opal__transient__top_k")
                                    )
                                    pred_cols = [
                                        "__row_id",
                                        "opal__transient__score",
                                        "opal__transient__rank",
                                        "opal__transient__logic_fidelity",
                                        "opal__transient__effect_scaled",
                                        "opal__transient__top_k",
                                    ]
                                    if "id" in df_pred_scored.columns:
                                        pred_cols.append("id")
                                    df_umap_overlay = df_active.join(
                                        df_pred_scored.select(pred_cols),
                                        on="__row_id",
                                        how="left",
                                    )
                                    transient_lines.append(
                                        f"Transient RF done: trained on `{df_train.height}` labels; "
                                        f"predicted `{df_pred_scored.height}` candidates."
                                    )

                                    _hist_source = df_pred_scored
                                    if "__row_id" in _hist_source.columns and "__row_id" in df_train.columns:
                                        _hist_source = _hist_source.join(
                                            df_train.select("__row_id"),
                                            on="__row_id",
                                            how="anti",
                                        )
                                    elif "id" in _hist_source.columns and "id" in df_train.columns:
                                        _hist_source = _hist_source.join(
                                            df_train.select("id"),
                                            on="id",
                                            how="anti",
                                        )
                                    _hist_pred = _hist_source.filter(
                                        pl.col("opal__transient__score").is_not_null()
                                    ).select(pl.col("opal__transient__score"))
                                    _obs_scores = pl.DataFrame({"opal__transient__score": []})
                                    if "score" in df_sfxi.columns:
                                        _obs_scores = df_sfxi.filter(pl.col("score").is_not_null()).select(
                                            pl.col("score").alias("opal__transient__score")
                                        )
                                    _obs_n = int(_obs_scores.height)
                                    _obs_cap = 200
                                    if _obs_n > _obs_cap:
                                        _obs_plot = _obs_scores.sample(n=_obs_cap, seed=1)
                                        _obs_note = f"observed labels: n={_obs_n} (showing {_obs_cap})"
                                    elif _obs_n > 0:
                                        _obs_plot = _obs_scores
                                        _obs_note = f"observed labels: n={_obs_n}"
                                    else:
                                        _obs_plot = _obs_scores
                                        _obs_note = "observed labels: n=0"
                                    if _obs_plot.height:
                                        _obs_plot = _obs_plot.with_columns(pl.lit(0).alias("__baseline"))
                                    if _hist_pred.is_empty():
                                        hist_df = pl.DataFrame(
                                            schema={
                                                "bin_start": pl.Float64,
                                                "bin_end": pl.Float64,
                                                "count": pl.Int64,
                                            }
                                        )
                                        _hist_chart = (
                                            alt.Chart(hist_df)
                                            .mark_bar()
                                            .encode(
                                                x=alt.X(
                                                    "bin_start:Q",
                                                    bin=alt.Bin(binned=True),
                                                    title="Predicted scalar score",
                                                    scale=alt.Scale(nice=True),
                                                ),
                                                x2="bin_end:Q",
                                                y=alt.Y(
                                                    "count:Q",
                                                    title="Count",
                                                    scale=alt.Scale(domainMin=0),
                                                ),
                                            )
                                        )
                                        _hist_chart = (
                                            with_title(
                                                _hist_chart,
                                                "Predicted scalar score distribution",
                                                f"{dataset_name or 'dataset'} · n=0 · {_obs_note}",
                                            )
                                            .properties(width="container", height=240)
                                            .configure_view(stroke=None)
                                        )
                                        transient_hist_chart = mo.vstack(
                                            [_hist_chart, mo.md("No transient predictions available.")]
                                        )
                                    else:
                                        _score_min = _hist_pred.select(pl.col("opal__transient__score").min()).item()
                                        _score_max = _hist_pred.select(pl.col("opal__transient__score").max()).item()
                                        if _obs_plot.height:
                                            _obs_min = _obs_plot.select(pl.col("opal__transient__score").min()).item()
                                            _obs_max = _obs_plot.select(pl.col("opal__transient__score").max()).item()
                                            if _obs_min is not None:
                                                _score_min = min(_score_min, _obs_min)
                                            if _obs_max is not None:
                                                _score_max = max(_score_max, _obs_max)
                                        _score_span = (_score_max - _score_min) if _score_max is not None else 0.0
                                        _pad = max(_score_span * 0.02, 1e-6)
                                        _x_scale = alt.Scale(
                                            domain=[_score_min - _pad, _score_max + _pad],
                                            nice=True,
                                        )
                                        _hist_values = _hist_pred.select(pl.col("opal__transient__score")).to_numpy()
                                        if _score_min is None or _score_max is None:
                                            _score_min = 0.0
                                            _score_max = 1.0
                                        if _score_min == _score_max:
                                            _score_min -= 1e-3
                                            _score_max += 1e-3
                                        _hist_counts, _hist_edges = np.histogram(
                                            _hist_values, bins=30, range=(_score_min, _score_max)
                                        )
                                        hist_df = pl.DataFrame(
                                            {
                                                "bin_start": _hist_edges[:-1],
                                                "bin_end": _hist_edges[1:],
                                                "count": _hist_counts,
                                            }
                                        )
                                        _hist_chart = (
                                            alt.Chart(hist_df)
                                            .mark_bar(opacity=0.7)
                                            .encode(
                                                x=alt.X(
                                                    "bin_start:Q",
                                                    bin=alt.Bin(binned=True),
                                                    title="Predicted scalar score",
                                                    scale=_x_scale,
                                                    axis=alt.Axis(tickCount=7),
                                                ),
                                                x2="bin_end:Q",
                                                y=alt.Y(
                                                    "count:Q",
                                                    title="Count",
                                                    scale=alt.Scale(domainMin=0),
                                                ),
                                                tooltip=[
                                                    alt.Tooltip("bin_start:Q", title="bin start"),
                                                    alt.Tooltip("bin_end:Q", title="bin end"),
                                                    alt.Tooltip("count:Q", title="count"),
                                                ],
                                            )
                                        )
                                        if _obs_plot.height:
                                            _obs_rules = (
                                                alt.Chart(_obs_plot)
                                                .mark_rule(opacity=0.25, strokeDash=[4, 4])
                                                .encode(
                                                    x=alt.X(
                                                        "opal__transient__score:Q",
                                                        scale=_x_scale,
                                                    )
                                                )
                                            )
                                            _obs_lollipops = (
                                                alt.Chart(_obs_plot)
                                                .mark_point(filled=True, opacity=0.35, size=20)
                                                .encode(
                                                    x=alt.X(
                                                        "opal__transient__score:Q",
                                                        scale=_x_scale,
                                                    ),
                                                    y=alt.Y("__baseline:Q", axis=None),
                                                )
                                            )
                                            _obs_note_chart = (
                                                alt.Chart(pl.DataFrame({"note": [_obs_note]}))
                                                .mark_text(
                                                    align="left",
                                                    baseline="top",
                                                    dx=0,
                                                    dy=6,
                                                    fontSize=11,
                                                    color="#666666",
                                                )
                                                .encode(
                                                    text="note:N",
                                                    x=alt.value(110),
                                                    y=alt.value(0),
                                                )
                                            )
                                            _hist_chart = _hist_chart + _obs_rules + _obs_lollipops + _obs_note_chart
                                        _hist_chart = (
                                            with_title(
                                                _hist_chart,
                                                "Predicted scalar score distribution",
                                                f"{dataset_name or 'dataset'} · n={_hist_pred.height} · {_obs_note}",
                                            )
                                            .properties(width="container", height=240)
                                            .configure_view(stroke=None)
                                        )
                                        transient_hist_chart = _hist_chart

    missing_transient_cols = [c for c in transient_cols if c not in df_umap_overlay.columns]
    if missing_transient_cols:
        fill_exprs = []
        for col in missing_transient_cols:
            if col == "opal__transient__top_k":
                fill_exprs.append(pl.lit(False).alias(col))
            elif col == "opal__transient__rank":
                fill_exprs.append(pl.lit(None, dtype=pl.Int64).alias(col))
            else:
                fill_exprs.append(pl.lit(None, dtype=pl.Float64).alias(col))
        df_umap_overlay = df_umap_overlay.with_columns(fill_exprs)

    _observed_col = "opal__transient__observed_event"
    _label_hist_col = None
    if opal_campaign_info is not None:
        _label_hist_col = f"opal__{opal_campaign_info.slug}__label_hist"
    if _label_hist_col and _label_hist_col in df_umap_overlay.columns:
        try:
            df_umap_overlay = df_umap_overlay.with_columns(
                pl.col(_label_hist_col)
                .list.eval(pl.element().struct.field("src") == "ingest_y")
                .list.any()
                .fill_null(False)
                .alias(_observed_col)
            )
        except Exception:
            df_umap_overlay = df_umap_overlay.with_columns(pl.lit(False).alias(_observed_col))
    elif _observed_col not in df_umap_overlay.columns:
        df_umap_overlay = df_umap_overlay.with_columns(pl.lit(False).alias(_observed_col))

    _sfxi_scored_col = "opal__transient__sfxi_scored_label"
    if "__row_id" in df_umap_overlay.columns and "__row_id" in df_sfxi.columns and not df_sfxi.is_empty():
        _sfxi_ids = df_sfxi.select(pl.col("__row_id").drop_nulls().unique()).to_series().to_list()
        df_umap_overlay = df_umap_overlay.with_columns(pl.col("__row_id").is_in(_sfxi_ids).alias(_sfxi_scored_col))
    elif _sfxi_scored_col not in df_umap_overlay.columns:
        df_umap_overlay = df_umap_overlay.with_columns(pl.lit(False).alias(_sfxi_scored_col))

    transient_md = mo.md("\n".join(transient_lines)) if transient_lines else None

    rf_umap_cluster_chart = None
    rf_umap_score_chart = None
    _umap_x_col = "cluster__ldn_v1__umap_x"
    _umap_y_col = "cluster__ldn_v1__umap_y"
    if _umap_x_col in df_umap_overlay.columns and _umap_y_col in df_umap_overlay.columns:
        if not hasattr(alt, "_DNAD_PLOT_SIZE"):
            alt._DNAD_PLOT_SIZE = 420
        _plot_size = alt._DNAD_PLOT_SIZE
        _base_cols = [_umap_x_col, _umap_y_col]
        if "id" in df_umap_overlay.columns:
            _base_cols.append("id")
        if "cluster__ldn_v1" in df_umap_overlay.columns:
            df_umap_cluster = df_umap_overlay.select(_base_cols + ["cluster__ldn_v1"]).filter(
                pl.col("cluster__ldn_v1").is_not_null()
            )
            if not df_umap_cluster.is_empty():
                _cluster_chart = (
                    alt.Chart(df_umap_cluster)
                    .mark_circle(opacity=0.7, stroke=None, strokeWidth=0, size=40)
                    .encode(
                        x=alt.X(_umap_x_col, title="UMAP X"),
                        y=alt.Y(_umap_y_col, title="UMAP Y"),
                        color=alt.Color("cluster__ldn_v1:N", title="Leiden cluster"),
                        tooltip=[c for c in ["id", "cluster__ldn_v1"] if c in df_umap_cluster.columns],
                    )
                )
                _cluster_chart = with_title(
                    _cluster_chart,
                    "UMAP colored by Leiden cluster",
                    f"{dataset_name or 'dataset'} · n={df_umap_cluster.height}",
                ).properties(width=_plot_size, height=_plot_size)
                rf_umap_cluster_chart = mo.ui.altair_chart(_cluster_chart)

        _score_col = None
        _score_title = None
        _score_chart_title = None
        if opal_campaign_info is not None:
            _opal_latest_pred_col = f"opal__{opal_campaign_info.slug}__latest_pred_scalar"
            if _opal_latest_pred_col in df_umap_overlay.columns:
                _score_col = _opal_latest_pred_col
                _score_title = "OPAL latest predicted scalar"
                _score_chart_title = "UMAP colored by OPAL latest scalar"
        if _score_col is None and "opal__transient__score" in df_umap_overlay.columns:
            _score_col = "opal__transient__score"
            _score_title = "Transient RF score"
            _score_chart_title = "UMAP colored by transient RF score"

        if _score_col is not None:
            df_umap_score = df_umap_overlay.select(dedupe_columns(_base_cols + [_score_col])).filter(
                pl.col(_score_col).is_not_null()
            )
            if not df_umap_score.is_empty():
                _score_chart = (
                    alt.Chart(df_umap_score)
                    .mark_circle(opacity=0.7, stroke=None, strokeWidth=0, size=40)
                    .encode(
                        x=alt.X(_umap_x_col, title="UMAP X"),
                        y=alt.Y(_umap_y_col, title="UMAP Y"),
                        color=alt.Color(
                            f"{_score_col}:Q",
                            title=_score_title,
                            legend=alt.Legend(title=_score_title, format=".2f", tickCount=5),
                        ),
                        tooltip=[c for c in ["id", _score_col] if c in df_umap_score.columns],
                    )
                )
                _score_chart = with_title(
                    _score_chart,
                    _score_chart_title or "UMAP colored by score",
                    f"{dataset_name or 'dataset'} · n={df_umap_score.height}",
                ).properties(width=_plot_size, height=_plot_size)
                rf_umap_score_chart = mo.ui.altair_chart(_score_chart)
    _unused_rf = (rf_umap_cluster_chart, rf_umap_score_chart)
    return (
        df_sfxi,
        df_umap_overlay,
        rf_model_source_note_md,
        sfxi_meta_md,
        sfxi_notice_md,
        transient_feature_chart,
        transient_hist_chart,
        transient_md,
    )


@app.cell
def _(
    alt,
    dataset_name,
    dedupe_columns,
    df_umap_overlay,
    mo,
    pl,
    safe_is_numeric,
    transient_cluster_hue_dropdown,
    transient_cluster_metric_dropdown,
    with_title,
):
    _transient_cluster_chart = None
    _metric_col = transient_cluster_metric_dropdown.value
    _hue_value = transient_cluster_hue_dropdown.value
    if _metric_col and "cluster__ldn_v1" in df_umap_overlay.columns and _metric_col in df_umap_overlay.columns:
        _cluster_cols = dedupe_columns(["cluster__ldn_v1", _metric_col])
        if "id" in df_umap_overlay.columns:
            _cluster_cols.append("id")
        df_cluster_points = (
            df_umap_overlay.filter(pl.col(_metric_col).is_not_null())
            .select(_cluster_cols)
            .with_columns(pl.col("cluster__ldn_v1").cast(pl.Int64, strict=False).alias("cluster__ldn_v1__ord"))
            .with_columns(
                pl.when(pl.col("cluster__ldn_v1__ord").is_null())
                .then(pl.lit(1_000_000_000))
                .otherwise(pl.col("cluster__ldn_v1__ord"))
                .alias("cluster__ldn_v1__ord")
            )
        )
        if not df_cluster_points.is_empty():
            plot_height = 240
            sort_field = alt.SortField(field="cluster__ldn_v1__ord", order="ascending")
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
            _metric_plot_col = _metric_col
            _metric_title = _metric_col
            _metric_type = "Q"
            _metric_dtype = df_umap_overlay.schema.get(_metric_col)
            if _metric_dtype is None or not safe_is_numeric(_metric_dtype):
                _metric_type = "N"
            _hue_label = _hue_value if _hue_value else "(none)"
            _color_encoding = alt.Undefined
            _color_tooltip = None
            _label_map = {
                "opal__transient__top_k": ("Top-K", "Not Top-K", "Transient Top-K"),
                "opal__transient__observed_event": ("Observed", "Not observed", "Observed events (ingest_y)"),
                "opal__transient__sfxi_scored_label": ("SFXI label", "Not label", "SFXI scored label"),
            }
            if _hue_value == "Leiden cluster":
                _hue_label = "Leiden cluster"
                _color_tooltip = "cluster__ldn_v1"
                _color_encoding = alt.Color(
                    "cluster__ldn_v1:N",
                    title="Leiden cluster",
                    scale=alt.Scale(range=_okabe_ito),
                    legend=alt.Legend(title="Leiden cluster"),
                )
            elif _hue_value in _label_map and _hue_value in df_cluster_points.columns:
                _label_col = f"{_hue_value}__label"
                _yes_label, _no_label, _hue_label = _label_map[_hue_value]
                df_cluster_points = df_cluster_points.with_columns(
                    pl.when(pl.col(_hue_value)).then(pl.lit(_yes_label)).otherwise(pl.lit(_no_label)).alias(_label_col)
                )
                _color_tooltip = _label_col
                _color_encoding = alt.Color(
                    f"{_label_col}:N",
                    title=_hue_label,
                    scale=alt.Scale(domain=[_yes_label, _no_label], range=[_okabe_ito[2], "#B0B0B0"]),
                    legend=alt.Legend(title=_hue_label),
                )
            points = (
                alt.Chart(df_cluster_points)
                .transform_calculate(jitter="(random() - 0.5) * 0.6")
                .mark_circle(size=22, opacity=0.6)
                .encode(
                    x=alt.X(
                        "cluster__ldn_v1:N",
                        sort=sort_field,
                        title="Leiden cluster",
                        axis=alt.Axis(labelAngle=90, labelFontSize=8),
                    ),
                    xOffset="jitter:Q",
                    y=alt.Y(f"{_metric_plot_col}:{_metric_type}", title=_metric_title),
                    color=_color_encoding,
                    tooltip=[
                        c
                        for c in ["cluster__ldn_v1", _metric_plot_col, _metric_col, "id", _color_tooltip]
                        if c in df_cluster_points.columns
                    ],
                )
            )
            if _color_encoding is alt.Undefined:
                points = points.encode(color=alt.value(_okabe_ito[4]))
            _cluster_chart = (
                with_title(
                    points,
                    "Transient score by Leiden cluster",
                    f"{dataset_name or 'dataset'} · y={_metric_col} · hue={_hue_label} · n={df_cluster_points.height}",
                )
                .properties(width="container", height=plot_height)
                .configure_view(stroke=None)
            )
            _transient_cluster_chart = mo.ui.altair_chart(_cluster_chart)

    transient_cluster_chart = _transient_cluster_chart
    return (transient_cluster_chart,)


@app.cell
def _():
    _noop = None
    return


@app.cell(column=2)
def _(
    dataset_explorer,
    dataset_status_md,
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
            dataset_status_md,
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
    df_active,
    df_sfxi_selected,
    df_umap_selected,
    export_button,
    export_format_dropdown,
    export_source_dropdown,
    export_source_label_map,
    mo,
    repo_root,
):
    export_status_md = None

    if export_button.value:
        source_label = export_source_dropdown.value
        source = export_source_label_map.get(source_label, source_label)
        if source == "df_umap_selected":
            df_export = df_umap_selected
        elif source == "df_sfxi_selected":
            df_export = df_sfxi_selected
        elif source == "active_record":
            df_export = active_record if active_record is not None else df_active.head(0)
        else:
            df_export = df_active

        if df_export is None or df_export.is_empty():
            export_status_md = mo.md("Nothing to export.")
        else:
            _export_out_dir = repo_root / "src" / "dnadesign" / "opal" / "notebooks" / "_outputs"
            _export_out_dir.mkdir(parents=True, exist_ok=True)
            suffix = export_format_dropdown.value
            out_path = _export_out_dir / f"promoter_eda_export.{suffix}"
            if suffix == "csv":
                df_export.write_csv(out_path)
            else:
                df_export.write_parquet(out_path)
            export_status_md = mo.md(
                f"Saved `{out_path}` from `{source_label}` ({df_export.height} rows × {len(df_export.columns)} cols)."
            )
    return (export_status_md,)


@app.cell(column=4)
def _(
    mo,
    umap_chart_note_md,
    umap_chart_ui,
    umap_color_dropdown,
    umap_opacity_slider,
    umap_selected_explorer,
    umap_selection_note_md,
    umap_size_slider,
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
            mo.hstack([umap_x_input, umap_y_input]),
            umap_color_dropdown,
            mo.hstack([umap_size_slider, umap_opacity_slider]),
            umap_chart_note_md,
            umap_chart_ui,
            umap_selection_note_md,
            umap_selected_explorer,
        ]
    )
    return


@app.cell
def _(campaign_label_from_path, df_active, list_campaign_paths, mo, repo_root):
    _unused = (df_active,)
    campaign_paths = list_campaign_paths(repo_root)
    campaign_labels = [campaign_label_from_path(path, repo_root) for path in campaign_paths]
    campaign_path_map = dict(zip(campaign_labels, campaign_paths))
    default_label = None
    for label in campaign_labels:
        if "prom60-etoh-cipro-andgate/campaign.yaml" in label.replace("\\", "/"):
            default_label = label
            break
    if default_label is None and campaign_labels:
        default_label = campaign_labels[0]
    if not campaign_labels:
        campaign_labels = ["(no campaigns found)"]
        default_label = campaign_labels[0]
    opal_panel_prefix_dropdown = mo.ui.dropdown(
        options=campaign_labels,
        value=default_label,
        label="Campaign config",
        full_width=True,
    )
    opal_panel_prefix_default = default_label
    return (
        campaign_path_map,
        opal_panel_prefix_default,
        opal_panel_prefix_dropdown,
    )


@app.cell
def _(
    build_label_events,
    campaign_path_map,
    df_active,
    load_campaign_yaml,
    mo,
    opal_panel_prefix_default,
    opal_panel_prefix_dropdown,
    parse_campaign_info,
    pl,
):
    opal_campaign_info = None
    opal_label_events_df = df_active.head(0)
    opal_round_dropdown = None
    opal_round_value_map = {}
    opal_label_hist_col = None

    if opal_panel_prefix_dropdown is not None:
        campaign_label = opal_panel_prefix_dropdown.value or opal_panel_prefix_default
        campaign_path = campaign_path_map.get(campaign_label) if campaign_label else None
        if campaign_path is not None and not campaign_label.startswith("(no campaigns"):
            try:
                raw_cfg = load_campaign_yaml(campaign_path)
                opal_campaign_info = parse_campaign_info(
                    raw=raw_cfg,
                    path=campaign_path,
                    label=campaign_label,
                )
            except Exception:
                opal_campaign_info = None

    if opal_campaign_info is not None:
        opal_label_hist_col = f"opal__{opal_campaign_info.slug}__label_hist"
        opal_label_events_df = build_label_events(
            df=df_active,
            label_hist_col=opal_label_hist_col,
            y_col_name=opal_campaign_info.y_column,
        )

        rounds = []
        if not opal_label_events_df.is_empty() and "observed_round" in opal_label_events_df.columns:
            rounds = opal_label_events_df.select(pl.col("observed_round").drop_nulls().unique()).to_series().to_list()
            rounds = sorted({int(r) for r in rounds})
        if rounds:
            latest_round = rounds[-1]
            latest_label = f"latest (R={latest_round})"
            opal_round_value_map = {latest_label: latest_round}
            for r in rounds:
                opal_round_value_map[f"R={r}"] = r
            round_options = [latest_label] + [f"R={r}" for r in rounds]
            round_default = latest_label
        else:
            round_options = ["latest (R=none)"]
            round_default = round_options[0]
            opal_round_value_map = {round_default: None}
        opal_round_dropdown = mo.ui.dropdown(
            options=round_options,
            value=round_default,
            label="Labels as of round",
            full_width=True,
        )
    return (
        opal_campaign_info,
        opal_label_events_df,
        opal_round_dropdown,
        opal_round_value_map,
    )


@app.cell
def _(mo):
    view_options = [
        "as_of_round (dedup latest per id)",
        "current_round_only",
        "all_events",
    ]
    opal_label_view_dropdown = mo.ui.dropdown(
        options=view_options,
        value=view_options[0],
        label="Label view",
        full_width=True,
    )
    return (opal_label_view_dropdown,)


@app.cell
def _(
    dedup_latest_labels,
    mo,
    opal_campaign_info,
    opal_label_events_df,
    opal_label_view_dropdown,
    opal_round_dropdown,
    opal_round_value_map,
    pl,
):
    opal_selected_round = None
    if opal_round_dropdown is not None:
        selected_label = opal_round_dropdown.value
        opal_selected_round = opal_round_value_map.get(selected_label) if selected_label else None

    opal_labels_view_df = opal_label_events_df.head(0)
    opal_labels_current_df = opal_label_events_df.head(0)
    opal_labels_asof_df = opal_label_events_df.head(0)
    opal_labels_table_ui = mo.ui.table(opal_labels_view_df, page_size=5)

    if opal_campaign_info is None:
        pass
    elif opal_label_events_df.is_empty():
        pass
    else:
        df_labels = opal_label_events_df
        if "label_src" in df_labels.columns:
            if df_labels.filter(pl.col("label_src") == "ingest_y").height > 0:
                df_labels = df_labels.filter(pl.col("label_src") == "ingest_y")
            else:
                pass
        else:
            pass

        df_round_scope = df_labels
        if opal_selected_round is not None and "observed_round" in df_labels.columns:
            df_round_scope = df_labels.filter(pl.col("observed_round") <= int(opal_selected_round))
            opal_labels_current_df = df_labels.filter(pl.col("observed_round") == int(opal_selected_round))
            cumulative = bool(opal_campaign_info.training_policy.get("cumulative_training", True))
            opal_labels_asof_df = df_round_scope if cumulative else opal_labels_current_df
            policy = str(
                opal_campaign_info.training_policy.get("label_cross_round_deduplication_policy", "latest_only")
            )
            if policy == "latest_only":
                opal_labels_asof_df = dedup_latest_labels(
                    opal_labels_asof_df,
                    id_col="id",
                    round_col="observed_round",
                )
        else:
            opal_labels_asof_df = df_round_scope

        view_choice_default = "as_of_round (dedup latest per id)"
        view_choice = opal_label_view_dropdown.value if opal_label_view_dropdown is not None else view_choice_default
        if view_choice.startswith("as_of_round"):
            opal_labels_view_df = opal_labels_asof_df
        elif view_choice == "current_round_only":
            opal_labels_view_df = opal_labels_current_df
        else:
            opal_labels_view_df = df_round_scope

        sort_cols = [col for col in ["id", "observed_round", "label_ts"] if col in opal_labels_view_df.columns]
        if sort_cols:
            opal_labels_view_df = opal_labels_view_df.sort(sort_cols)

        _y_col_name = opal_campaign_info.y_column
        _y_display_col = _y_col_name
        if _y_col_name and _y_col_name in opal_labels_view_df.columns:
            _preview_col = f"{_y_col_name}_preview"

            def _preview_vec(val) -> str | None:
                if val is None:
                    return None
                text = str(val)
                if len(text) > 120:
                    return text[:117] + "..."
                return text

            opal_labels_view_df = opal_labels_view_df.with_columns(
                pl.col(_y_col_name).map_elements(_preview_vec, return_dtype=pl.Utf8).alias(_preview_col)
            )
            _y_display_col = _preview_col
        _display_cols = [
            col
            for col in ["id", "observed_round", "label_src", _y_display_col]
            if col and col in opal_labels_view_df.columns
        ]
        if _display_cols:
            opal_labels_table_ui = mo.ui.table(
                opal_labels_view_df.select(_display_cols),
                page_size=5,
            )
        else:
            opal_labels_table_ui = mo.ui.table(opal_labels_view_df, page_size=5)
    return (
        opal_labels_asof_df,
        opal_labels_current_df,
        opal_labels_table_ui,
        opal_labels_view_df,
        opal_selected_round,
    )


@app.cell
def _(mo, opal_campaign_info):
    objective_params = opal_campaign_info.objective_params if opal_campaign_info is not None else {}
    setpoint = objective_params.get("setpoint_vector") or [0.0, 0.0, 0.0, 1.0]
    if len(setpoint) != 4:
        setpoint = [0.0, 0.0, 0.0, 1.0]
    p00, p10, p01, p11 = (float(x) for x in setpoint)
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
        sfxi_fixed_params,
        sfxi_gamma_input,
        sfxi_p00_slider,
        sfxi_p01_slider,
        sfxi_p10_slider,
        sfxi_p11_slider,
    )


@app.cell
def _(
    mo,
    pl,
    sfxi_p00_slider,
    sfxi_p01_slider,
    sfxi_p10_slider,
    sfxi_p11_slider,
):
    _unused = (pl,)
    _p00 = float(sfxi_p00_slider.value)
    _p10 = float(sfxi_p10_slider.value)
    _p01 = float(sfxi_p01_slider.value)
    _p11 = float(sfxi_p11_slider.value)
    _setpoint_text = ", ".join([f"{v:.3f}" for v in (_p00, _p10, _p01, _p11)])
    sfxi_setpoint_md = mo.md(f"Current setpoint vector (state order: 00, 10, 01, 11): `p = [{_setpoint_text}]`")
    return (sfxi_setpoint_md,)


@app.cell
def _(build_color_dropdown_options, df_sfxi, mo):
    fallback_cols = [
        "sequence",
        "score",
        "logic_fidelity",
        "effect_scaled",
        "label_src",
        "observed_round",
        "cluster__ldn_v1",
    ]
    base_options = build_color_dropdown_options(df_sfxi, include_none=False)
    options = []
    for _name in fallback_cols:
        if _name in base_options and _name not in options:
            options.append(_name)
    for _name in base_options:
        if _name not in options:
            options.append(_name)
    if not options:
        options = ["score"]
    _bid_default = None
    for _name in options:
        if _name == "BID":
            _bid_default = _name
            break
    if _bid_default is None:
        for _name in options:
            if _name == "bid":
                _bid_default = _name
                break
    if _bid_default is None:
        for _name in options:
            if _name.lower().endswith("__bid"):
                _bid_default = _name
                break
    if _bid_default is not None:
        default = _bid_default
    elif "score" in options:
        default = "score"
    else:
        default = options[0]
    sfxi_color_dropdown = mo.ui.dropdown(
        options=options,
        value=default,
        label="Color by",
        full_width=True,
    )
    return (sfxi_color_dropdown,)


@app.cell
def _(
    alt,
    dataset_name,
    df_sfxi,
    mo,
    opal_campaign_info,
    pl,
    safe_is_numeric,
    sfxi_color_dropdown,
    with_title,
):
    if not hasattr(alt, "_DNAD_PLOT_SIZE"):
        alt._DNAD_PLOT_SIZE = 420
    _plot_size = alt._DNAD_PLOT_SIZE
    _sfxi_explain_text = "No SFXI data to plot."
    _slug = opal_campaign_info.slug if opal_campaign_info is not None else "campaign"
    _base_subtitle = f"{dataset_name or 'dataset'} · {_slug}"
    _color_value = sfxi_color_dropdown.value
    _color_title = _color_value if _color_value else None
    _color_tooltip = None
    _color_encoding = alt.Undefined
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
    if df_sfxi.is_empty():
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
        )
        _chart = with_title(
            _chart,
            "SFXI components: logic fidelity vs scaled effect",
            _base_subtitle,
        ).properties(width=_plot_size, height=_plot_size)
        sfxi_chart_note_md = mo.md(_sfxi_explain_text)
    else:
        df_sfxi_plot = df_sfxi.select(
            [col for col in ["__row_id", "id", "logic_fidelity", "effect_scaled", "score"] if col in df_sfxi.columns]
        )
        if (
            _color_value
            and _color_value in df_sfxi.columns
            and _color_value not in df_sfxi_plot.columns
            and "__row_id" in df_sfxi_plot.columns
        ):
            df_sfxi_plot = df_sfxi_plot.join(
                df_sfxi.select(["__row_id", _color_value]),
                on="__row_id",
                how="left",
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
            if c in df_sfxi_plot.columns
        ]
        _color_note = None
        if _color_value and _color_value in df_sfxi_plot.columns:
            _dtype = df_sfxi.schema.get(_color_value)
            _non_null_count = df_sfxi_plot.select(pl.col(_color_value).count()).item() if df_sfxi_plot.height else 0
            _is_nested = False
            if _dtype is not None:
                try:
                    _is_nested = bool(getattr(_dtype, "is_nested")())
                except Exception:
                    _is_nested = False
            if _non_null_count == 0:
                _color_note = f"Color `{_color_value}` has no non-null values; rendering without color."
            elif _is_nested:
                _color_note = f"Color `{_color_value}` is nested; rendering without color."
            elif _color_value == "opal__transient__top_k":
                _label_col = f"{_color_value}__label"
                df_sfxi_plot = df_sfxi_plot.with_columns(
                    pl.when(pl.col(_color_value)).then(pl.lit("Top-K")).otherwise(pl.lit("Not Top-K")).alias(_label_col)
                )
                _color_title = "Transient Top-K"
                _color_tooltip = _label_col
                _color_encoding = alt.Color(
                    f"{_label_col}:N",
                    title=_color_title,
                    scale=alt.Scale(
                        domain=["Top-K", "Not Top-K"],
                        range=[_okabe_ito[5], "#B0B0B0"],
                    ),
                    legend=alt.Legend(title=_color_title),
                )
            elif _dtype is not None and safe_is_numeric(_dtype):
                _color_title = _color_value
                _color_tooltip = _color_value
                _color_encoding = alt.Color(
                    f"{_color_value}:Q",
                    title=_color_title,
                    legend=alt.Legend(title=_color_title, format=".2f", tickCount=5),
                )
            else:
                _n_unique = df_sfxi_plot.select(pl.col(_color_value).n_unique()).item() if df_sfxi_plot.height else 0
                _color_scale = (
                    alt.Scale(range=_okabe_ito) if _n_unique <= len(_okabe_ito) else alt.Scale(scheme=_fallback_scheme)
                )
                _color_title = _color_value
                _color_tooltip = _color_value
                _color_encoding = alt.Color(
                    f"{_color_value}:N",
                    title=_color_title,
                    scale=_color_scale,
                    legend=alt.Legend(title=_color_title),
                )
        else:
            _color_note = f"Color `{_color_value}` unavailable; rendering without color."

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
def _(df_sfxi, opal_campaign_info, pl, sfxi_chart_ui):
    _unused = (opal_campaign_info, pl, sfxi_chart_ui)
    df_sfxi_selected = df_sfxi
    return (df_sfxi_selected,)


@app.cell(column=5)
def _(
    mo,
    opal_label_view_dropdown,
    opal_labels_table_ui,
    opal_panel_prefix_dropdown,
    opal_round_dropdown,
    sfxi_beta_input,
    sfxi_chart_note_md,
    sfxi_chart_ui,
    sfxi_color_dropdown,
    sfxi_gamma_input,
    sfxi_meta_md,
    sfxi_notice_md,
    sfxi_p00_slider,
    sfxi_p01_slider,
    sfxi_p10_slider,
    sfxi_p11_slider,
    sfxi_setpoint_md,
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
    setpoint_row_top = mo.hstack([sfxi_p00_slider, sfxi_p10_slider, sfxi_p01_slider, sfxi_p11_slider])
    exponent_row = mo.hstack([sfxi_beta_input, sfxi_gamma_input])
    sfxi_controls = mo.vstack([setpoint_row_top, exponent_row])
    sfxi_defs_md = mo.md(
        "- **logic_fidelity**: proximity to the setpoint logic vector.\n"
        "- **effect_scaled**: intensity normalized by the current scaling denom.\n"
        "- **score** = (logic_fidelity ** beta) * (effect_scaled ** gamma)"
    )

    opal_controls = []
    if opal_panel_prefix_dropdown is not None:
        opal_controls = [opal_panel_prefix_dropdown]

    opal_label_controls = [c for c in [opal_round_dropdown, opal_label_view_dropdown] if c is not None]
    opal_label_controls_row = mo.hstack(opal_label_controls) if opal_label_controls else mo.md("")

    sfxi_panel = mo.vstack(
        [
            mo.md("## Characterizing promoters by their SFXI score"),
            mo.md(
                "SFXI (setpoint fidelity x intensity) compresses a 2-factor logic response into a single "
                "scalar that balances logic correctness and brightness. OPAL is the active-learning harness "
                "that tries to predict an 8-vector response and reduces it via an objective to rank candidates."
            ),
            _opal_references_md,
            *opal_controls,
            mo.md("### Labels (observed events)"),
            mo.md(
                "These rows reflect experimental label events ingested for the selected campaign "
                "(e.g., via `opal ingest-y`). The **as-of round** view defines the training cutoff "
                "for active learning."
            ),
            opal_label_controls_row,
            opal_labels_table_ui,
            mo.md("### SFXI: setpoint + exponents"),
            mo.md(
                "The **setpoint** defines the desired logic profile `p ∈ [0,1]^4` in state order "
                "`[00, 10, 01, 11]`. Adjusting `p` changes what “good” logic looks like and how intensity is "
                "weighted across states. Exponents β/γ tune the trade-off between logic fidelity and intensity."
            ),
            sfxi_controls,
            sfxi_notice_md,
            sfxi_setpoint_md,
            sfxi_defs_md,
            sfxi_color_dropdown,
            sfxi_chart_note_md,
            sfxi_chart_ui,
            sfxi_meta_md,
        ]
    )
    sfxi_panel
    return


@app.cell(column=6)
def _(
    mo,
    rf_model_source,
    rf_model_source_note_md,
    transient_cluster_chart,
    transient_cluster_hue_dropdown,
    transient_cluster_metric_dropdown,
    transient_feature_chart,
    transient_hist_chart,
    transient_md,
):
    _unused = (transient_md,)
    feature_panel = (
        transient_feature_chart if transient_feature_chart is not None else mo.md("Feature importance unavailable.")
    )
    hist_panel = (
        transient_hist_chart if transient_hist_chart is not None else mo.md("Transient score histogram unavailable.")
    )
    cluster_panel = (
        transient_cluster_chart
        if transient_cluster_chart is not None
        else mo.md("Leiden cluster score plot unavailable.")
    )
    rf_blocks = [
        mo.md("## Reactive, session‑scoped (ephemeral) Random Forest surrogate"),
        rf_model_source,
        rf_model_source_note_md,
        mo.md(
            "**OPAL runs** produce round-scoped, versioned model artifacts (e.g., `model.joblib`) "
            "and write an append-only ledger for auditability and reproducibility "
            "(e.g., `outputs/round_<k>/...` and `outputs/ledger.*`).\n"
            "In contrast, this notebook view uses a reactive, session-scoped (ephemeral) surrogate model "
            "that is refit in memory on each UI-driven change and is not persisted as an artifact. "
            "Notebook predictions are exploratory overlays; ledger-backed outputs require running OPAL."
        ),
        feature_panel,
        mo.md(
            "Feature importance highlights which input dimensions the transient model relies on most for the "
            "current training set."
        ),
        hist_panel,
        mo.md("The histogram shows the distribution of predicted SFXI scores across the full candidate pool."),
        transient_cluster_metric_dropdown,
        transient_cluster_hue_dropdown,
        cluster_panel,
        mo.md("The cluster plot compares predicted scores across Leiden clusters, ordered numerically for stability."),
    ]
    mo.vstack(rf_blocks)
    return


@app.cell
def _(df_umap_overlay, pl):
    df_transient_top_k_pool = df_umap_overlay.head(0)
    if "opal__transient__top_k" in df_umap_overlay.columns:
        df_transient_top_k_pool = df_umap_overlay.filter(pl.col("opal__transient__top_k").fill_null(False))
    return (df_transient_top_k_pool,)


@app.cell
def _(mo):
    inspect_pool_label_map = {
        "Full dataset (all rows)": "df_active",
        "UMAP brush selection": "df_umap_selected",
        "SFXI scored labels (current view)": "df_sfxi_selected",
        "Transient surrogate Top-K": "df_transient_top_k_pool",
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
    df_sfxi_selected,
    df_transient_top_k_pool,
    df_umap_selected,
    inspect_pool_dropdown,
    inspect_pool_label_map,
):
    pool_choice = inspect_pool_label_map.get(inspect_pool_dropdown.value, inspect_pool_dropdown.value)
    if pool_choice == "df_umap_selected":
        df_pool = df_umap_selected
    elif pool_choice == "df_sfxi_selected":
        df_pool = df_sfxi_selected
    elif pool_choice == "df_transient_top_k_pool":
        df_pool = df_transient_top_k_pool
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
                import matplotlib.pyplot as plt

                plt.close(_fig)
                save_pdf_status_md = mo.md(f"Saved PDF: `{save_pdf_path}`")
            except Exception as exc:
                save_pdf_status_md = mo.md(f"Failed to save PDF at `{save_pdf_path}`: {exc}")
    return save_pdf_status_md, save_pdf_target_md


if __name__ == "__main__":
    app.run()
