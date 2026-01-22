import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _():
    import os
    from pathlib import Path
    from textwrap import dedent

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

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
        alt,
        dedent,
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
    from dnadesign.opal.src.analysis.dashboard import context as dash_context
    from dnadesign.opal.src.analysis.dashboard import datasets as dash_datasets
    from dnadesign.opal.src.analysis.dashboard import diagnostics as dash_diagnostics
    from dnadesign.opal.src.analysis.dashboard import labels as dash_labels
    from dnadesign.opal.src.analysis.dashboard import ledger as dash_ledger
    from dnadesign.opal.src.analysis.dashboard import mismatch as dash_mismatch
    from dnadesign.opal.src.analysis.dashboard import models as dash_models
    from dnadesign.opal.src.analysis.dashboard import plots as dash_plots
    from dnadesign.opal.src.analysis.dashboard import scores as dash_scores
    from dnadesign.opal.src.analysis.dashboard import selection as dash_selection
    from dnadesign.opal.src.analysis.dashboard import sfxi as dash_sfxi
    from dnadesign.opal.src.analysis.dashboard import transient as dash_transient
    from dnadesign.opal.src.analysis.dashboard import ui as dash_ui
    from dnadesign.opal.src.analysis.dashboard import util as dash_util

    apply_score_overlay = dash_scores.apply_score_overlay
    ScoreOverlayDiagnostics = dash_scores.ScoreOverlayDiagnostics
    build_umap_chart = dash_plots.build_umap_chart
    build_cluster_chart = dash_plots.build_cluster_chart
    build_umap_explorer_chart = dash_plots.build_umap_explorer_chart
    build_umap_overlay_charts = dash_plots.build_umap_overlay_charts
    build_color_dropdown_options = dash_util.build_color_dropdown_options
    build_friendly_column_labels = dash_util.build_friendly_column_labels
    build_label_events = dash_labels.build_label_events
    build_label_events_from_ledger = dash_labels.build_label_events_from_ledger
    build_label_sfxi_view = dash_sfxi.build_label_sfxi_view
    apply_overlay_label_flags = dash_sfxi.apply_overlay_label_flags
    build_umap_controls = dash_ui.build_umap_controls
    campaign_label_from_path = dash_datasets.campaign_label_from_path
    compute_sfxi_params = dash_sfxi.compute_sfxi_params
    ensure_selection_columns = dash_selection.ensure_selection_columns
    compute_transient_overlay = dash_transient.compute_transient_overlay
    coerce_selection_dataframe = dash_selection.coerce_selection_dataframe
    compare_selection_to_ledger = dash_mismatch.compare_selection_to_ledger
    Diagnostics = dash_diagnostics.Diagnostics
    diagnostics_to_lines = dash_diagnostics.diagnostics_to_lines
    DashboardContext = dash_context.DashboardContext
    dedupe_columns = dash_util.dedupe_columns
    dedupe_exprs = dash_util.dedupe_exprs
    dedup_latest_labels = dash_labels.dedup_latest_labels
    find_repo_root = dash_datasets.find_repo_root
    is_altair_undefined = dash_util.is_altair_undefined
    list_campaign_paths = dash_datasets.list_campaign_paths
    list_usr_datasets = dash_datasets.list_usr_datasets
    load_campaign_yaml = dash_datasets.load_campaign_yaml
    list_campaign_dataset_refs = dash_datasets.list_campaign_dataset_refs
    load_ledger_labels = dash_ledger.load_ledger_labels
    load_ledger_predictions = dash_ledger.load_ledger_predictions
    load_ledger_runs = dash_ledger.load_ledger_runs
    ledger_diagnostics = dash_ledger.LedgerDiagnostics
    resolve_artifact_state = dash_models.resolve_artifact_state
    missingness_summary = dash_util.missingness_summary
    namespace_summary = dash_util.namespace_summary
    parse_campaign_info = dash_datasets.parse_campaign_info
    resolve_objective_mode = dash_selection.resolve_objective_mode
    resolve_brush_selection = dash_selection.resolve_brush_selection
    resolve_sfxi_readiness = dash_sfxi.resolve_sfxi_readiness
    resolve_campaign_workdir = dash_datasets.resolve_campaign_workdir
    resolve_dataset_path = dash_datasets.resolve_dataset_path
    resolve_usr_root = dash_datasets.resolve_usr_root
    safe_is_numeric = dash_util.safe_is_numeric
    return (
        apply_score_overlay,
        ScoreOverlayDiagnostics,
        build_umap_chart,
        build_cluster_chart,
        build_umap_explorer_chart,
        build_umap_overlay_charts,
        build_color_dropdown_options,
        build_friendly_column_labels,
        build_label_events,
        build_label_events_from_ledger,
        build_label_sfxi_view,
        apply_overlay_label_flags,
        build_umap_controls,
        campaign_label_from_path,
        coerce_selection_dataframe,
        compute_sfxi_params,
        compute_transient_overlay,
        compare_selection_to_ledger,
        Diagnostics,
        diagnostics_to_lines,
        DashboardContext,
        dedup_latest_labels,
        dedupe_columns,
        dedupe_exprs,
        find_repo_root,
        is_altair_undefined,
        list_campaign_paths,
        list_campaign_dataset_refs,
        list_usr_datasets,
        load_campaign_yaml,
        load_ledger_labels,
        load_ledger_predictions,
        load_ledger_runs,
        ledger_diagnostics,
        resolve_artifact_state,
        missingness_summary,
        namespace_summary,
        parse_campaign_info,
        resolve_objective_mode,
        resolve_brush_selection,
        resolve_sfxi_readiness,
        ensure_selection_columns,
        resolve_campaign_workdir,
        resolve_dataset_path,
        resolve_usr_root,
        safe_is_numeric,
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
def _(campaign_label_from_path, list_campaign_paths, repo_root):
    campaign_paths = list_campaign_paths(repo_root)
    campaign_labels = [campaign_label_from_path(path, repo_root) for path in campaign_paths]
    campaign_path_map = dict(zip(campaign_labels, campaign_paths))
    default_campaign_label = None
    for label in campaign_labels:
        if "prom60-etoh-cipro-andgate/campaign.yaml" in label.replace("\\", "/"):
            default_campaign_label = label
            break
    if default_campaign_label is None and campaign_labels:
        default_campaign_label = campaign_labels[0]
    if not campaign_labels:
        campaign_labels = ["(no campaigns found)"]
        default_campaign_label = campaign_labels[0]
    return campaign_labels, campaign_path_map, default_campaign_label


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
def _(default_campaign_label, list_campaign_dataset_refs, repo_root):
    campaign_dataset_refs = list_campaign_dataset_refs(repo_root)
    campaign_dataset_names = sorted({ref.dataset_name for ref in campaign_dataset_refs if ref.dataset_name})
    default_campaign_dataset_name = None
    for _ref in campaign_dataset_refs:
        if _ref.campaign_label == default_campaign_label:
            if _ref.dataset_name:
                default_campaign_dataset_name = _ref.dataset_name
            elif _ref.records_path and _ref.records_path.name == "records.parquet":
                default_campaign_dataset_name = _ref.records_path.parent.name
            break
    return campaign_dataset_names, campaign_dataset_refs, default_campaign_dataset_name


@app.cell
def _(campaign_dataset_names, default_campaign_dataset_name, mo, usr_datasets):
    dataset_options = list(usr_datasets or [])
    for _name in campaign_dataset_names:
        if _name not in dataset_options:
            dataset_options.append(_name)
    if not dataset_options:
        dataset_options = ["(none found)"]
    preferred_default = "60bp_dual_promoter_cpxR_LexA"
    if usr_datasets and preferred_default in usr_datasets:
        dataset_default_value = preferred_default
    elif not usr_datasets and preferred_default in dataset_options:
        dataset_default_value = preferred_default
    elif usr_datasets:
        dataset_default_value = usr_datasets[0]
    elif default_campaign_dataset_name and default_campaign_dataset_name in dataset_options:
        dataset_default_value = default_campaign_dataset_name
    else:
        dataset_default_value = dataset_options[0]
    dataset_dropdown = mo.ui.dropdown(
        options=dataset_options,
        value=dataset_default_value,
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
    return custom_path_input, dataset_dropdown, dataset_form, dataset_default_value


@app.cell
def _():
    dataset_submit_override = False
    return (dataset_submit_override,)


@app.cell
def _(mo):
    dataset_autoload_state, set_dataset_autoload_state = mo.state(False)
    return dataset_autoload_state, set_dataset_autoload_state


@app.cell
def _(
    custom_path_input,
    dataset_autoload_state,
    dataset_default_value,
    dataset_dropdown,
    dataset_form,
):
    _custom_path_text = custom_path_input.value.strip()
    dataset_auto_submit = (
        dataset_form.value is None
        and not dataset_autoload_state()
        and dataset_dropdown is not None
        and dataset_dropdown.value == dataset_default_value
        and not _custom_path_text
    )
    dataset_autoload_default_ok = (
        dataset_form.value is None
        and dataset_autoload_state()
        and dataset_dropdown is not None
        and dataset_dropdown.value == dataset_default_value
        and not _custom_path_text
    )
    return dataset_auto_submit, dataset_autoload_default_ok


@app.cell
def _(
    dataset_auto_submit,
    dataset_autoload_default_ok,
    dataset_form,
    dataset_submit_override,
    mo,
    os,
):
    dataset_submit_override_env = os.environ.get("DNADESIGN_HEADLESS") == "1"
    notice_lines = []
    if dataset_submit_override and not dataset_submit_override_env:
        notice_lines.append(
            "Headless dataset override requested but disabled. "
            "Set `DNADESIGN_HEADLESS=1` to enable non-interactive loads."
        )
    if dataset_auto_submit:
        notice_lines.append("Auto-loading the default dataset. Submit the form to change datasets.")
    elif dataset_autoload_default_ok:
        notice_lines.append("Default dataset loaded. Submit the form to change datasets.")
    elif dataset_form.value is None and not (dataset_submit_override and dataset_submit_override_env):
        notice_lines.append("Selection pending: submit the form to load the dataset.")
    elif dataset_submit_override and dataset_submit_override_env:
        notice_lines.append("Headless dataset override enabled (no submit required).")
    dataset_submit_notice_md = mo.md("\n".join(notice_lines)) if notice_lines else mo.md("")
    return dataset_submit_notice_md, dataset_submit_override_env


@app.cell
def _(
    campaign_dataset_refs,
    custom_path_input,
    dataset_auto_submit,
    dataset_autoload_default_ok,
    dataset_autoload_state,
    dataset_dropdown,
    dataset_form,
    dataset_submit_override,
    dataset_submit_override_env,
    mo,
    pl,
    repo_root,
    repo_root_error,
    resolve_dataset_path,
    set_dataset_autoload_state,
    usr_datasets,
    usr_root,
):
    custom_path_text = custom_path_input.value.strip()
    if (
        dataset_form.value is None
        and not (dataset_submit_override and dataset_submit_override_env)
        and not dataset_auto_submit
        and not dataset_autoload_default_ok
    ):
        mo.stop(True, mo.md("Submit the form to load the dataset selection."))
    if dataset_auto_submit and not dataset_autoload_state():
        set_dataset_autoload_state(True)
    _repo_root_error_md = mo.md(repo_root_error) if repo_root_error is not None else mo.md("")
    mo.stop(repo_root_error is not None, _repo_root_error_md)

    _dataset_choice = dataset_dropdown.value if dataset_dropdown is not None else None
    if _dataset_choice in {"(none found)", "(none)"}:
        _dataset_choice = None
    if not custom_path_text and not _dataset_choice:
        mo.stop(True, mo.md("No datasets found. Provide a custom path or set `DNADESIGN_USR_ROOT`."))

    dataset_path = None
    dataset_mode = None
    if not custom_path_text and _dataset_choice:
        if usr_root is not None and usr_datasets and _dataset_choice in usr_datasets:
            dataset_path = (usr_root / _dataset_choice / "records.parquet").resolve()
            dataset_mode = "usr"
    if not custom_path_text and _dataset_choice and dataset_path is None:
        for _ref in campaign_dataset_refs:
            if _ref.dataset_name == _dataset_choice and _ref.records_path is not None:
                dataset_path = _ref.records_path
                dataset_mode = "campaign"
                break

    if dataset_path is None:
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
        if dataset_mode == "usr":
            hint = "Did you mean `src/dnadesign/usr/datasets/<name>/records.parquet`?"
        elif dataset_mode == "campaign":
            hint = "Resolved from campaign.yaml; ensure the dataset exists or set `DNADESIGN_USR_ROOT`."
        else:
            hint = "Provide an absolute path or a repo-root-relative path."
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

    if dataset_path.name == "records.parquet":
        dataset_name = dataset_path.parent.name
    else:
        dataset_name = dataset_path.stem
    status_lines = [
        f"Dataset: `{dataset_name}`",
        f"Mode: `{dataset_mode}`",
        f"Rows: `{df_raw.height}`",
        f"Columns: `{len(df_raw.columns)}`",
    ]
    dataset_status_md = mo.md("\n".join(status_lines))
    return dataset_name, dataset_path, dataset_status_md, df_raw


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
    build_friendly_column_labels,
    df_prelim,
    df_umap_overlay,
    mo,
    opal_campaign_info,
    score_source_dropdown,
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
        "opal__score__scalar",
        "opal__score__rank",
        "opal__score__top_k",
        "opal__ledger__score",
        "opal__ledger__top_k",
        "opal__cache__score",
        "opal__cache__top_k",
        "opal__overlay__score",
        "opal__overlay__logic_fidelity",
        "opal__overlay__effect_scaled",
        "opal__overlay__rank",
        "opal__overlay__observed_event",
        "opal__overlay__top_k",
    ]
    _rf_prefix = "Overlay (artifact)"
    _slug = opal_campaign_info.slug if opal_campaign_info is not None else None
    _df_explorer_source = df_umap_overlay if df_umap_overlay is not None else df_prelim
    _extra_color_cols = [col for col in _extra_color_cols if col in _df_explorer_source.columns]
    _color_options = build_color_dropdown_options(
        _df_explorer_source,
        extra=_extra_color_cols,
        include_none=True,
    )
    _color_options = [_name for _name in _color_options if _name != "id_right"]
    _score_source_label = score_source_dropdown.value if score_source_dropdown is not None else "Selected"
    _friendly_color_labels = build_friendly_column_labels(
        score_source_label=_score_source_label,
        rf_prefix=_rf_prefix,
        campaign_slug=_slug,
    )
    dataset_explorer_color_label_map = {}
    dataset_explorer_color_options = []
    for _raw in _color_options:
        _label = _friendly_color_labels.get(_raw, _raw)
        if _label in dataset_explorer_color_label_map:
            _label = _raw
        dataset_explorer_color_label_map[_label] = _raw
        dataset_explorer_color_options.append(_label)
    _color_default_label = "(none)"
    if _color_default_label not in dataset_explorer_color_options and dataset_explorer_color_options:
        _color_default_label = dataset_explorer_color_options[0]
    dataset_explorer_color_dropdown = mo.ui.dropdown(
        options=dataset_explorer_color_options,
        value=_color_default_label,
        label="Color by (scatter only)",
        full_width=True,
    )
    bins_slider = mo.ui.slider(5, 200, value=30, label="Histogram bins")
    return (
        bins_slider,
        dataset_explorer_color_label_map,
        dataset_explorer_color_dropdown,
        plot_type_dropdown,
        x_dropdown,
        y_dropdown,
    )


@app.cell
def _(
    alt,
    bins_slider,
    dataset_explorer_color_label_map,
    dataset_explorer_color_dropdown,
    dataset_name,
    dedupe_exprs,
    df_prelim,
    df_umap_overlay,
    mo,
    np,
    opal_campaign_info,
    pl,
    plot_type_dropdown,
    resolve_objective_mode,
    safe_is_numeric,
    with_title,
    x_dropdown,
    y_dropdown,
):
    plot_type = plot_type_dropdown.value
    _x_col = x_dropdown.value
    _y_col = y_dropdown.value
    _note_lines = []
    if not hasattr(alt, "_DNAD_PLOT_SIZE"):
        alt._DNAD_PLOT_SIZE = 420
    _plot_size = alt._DNAD_PLOT_SIZE
    _df_explorer_source = df_umap_overlay if df_umap_overlay is not None else df_prelim

    _dataset_label = dataset_name or "dataset"
    _color_label = dataset_explorer_color_dropdown.value
    _color_value = dataset_explorer_color_label_map.get(_color_label, _color_label)
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
    _selected_ids = None
    _selected_id_col = None
    if _color_value == "opal__score__top_k" and "opal__score__top_k" in _df_explorer_source.columns:
        _selected_id_col = "id" if "id" in _df_explorer_source.columns else "__row_id"
        if _selected_id_col in _df_explorer_source.columns:
            _selected_ids = (
                _df_explorer_source.filter(pl.col("opal__score__top_k").fill_null(False))
                .select(pl.col(_selected_id_col).cast(pl.Utf8))
                .to_series()
                .to_list()
            )
    if plot_type == "scatter":
        if _x_col not in _df_explorer_source.columns or _y_col not in _df_explorer_source.columns:
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
                _select_exprs = [
                    pl.col("__row_id"),
                    pl.col("id"),
                    pl.col(_x_col),
                    pl.col(_y_col).alias(y_col_plot),
                ]
            else:
                y_col_plot = _y_col
                _select_exprs = [
                    pl.col("__row_id"),
                    pl.col("id"),
                    pl.col(_x_col),
                    pl.col(_y_col),
                ]
            if (
                _color_value
                and _color_value != "(none)"
                and _color_value in _df_explorer_source.columns
                and _color_value not in {_x_col, _y_col}
            ):
                _select_exprs.append(pl.col(_color_value))
            df_plot = _df_explorer_source.select(dedupe_exprs(_select_exprs))
            if (
                _color_value == "opal__score__top_k"
                and _selected_ids is not None
                and _selected_id_col in df_plot.columns
            ):
                df_plot = df_plot.with_columns(
                    pl.col(_selected_id_col).cast(pl.Utf8).is_in(_selected_ids).alias("opal__score__top_k")
                )
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
                    elif _color_value == "opal__overlay__observed_event":
                        _label_col = f"{_color_value}__label"
                        df_plot = df_plot.with_columns(
                            pl.when(pl.col(_color_value))
                            .then(pl.lit("Observed"))
                            .otherwise(pl.lit("Not observed"))
                            .alias(_label_col)
                        )
                        _color_title = _color_label
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
                    elif _color_value == "opal__score__top_k":
                        _label_col = f"{_color_value}__label"
                        df_plot = df_plot.with_columns(
                            pl.when(pl.col(_color_value))
                            .then(pl.lit("Top-K"))
                            .otherwise(pl.lit("Not Top-K"))
                            .alias(_label_col)
                        )
                        _color_title = _color_label
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
                        _color_title = _color_label
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
                        _color_title = _color_label
                        _color_tooltip = _color_value
                        _color_encoding = alt.Color(
                            f"{_color_value}:N",
                            title=_color_title,
                            scale=_color_scale,
                            legend=alt.Legend(title=_color_title),
                        )
            _brush = alt.selection_interval(name="explorer_brush", encodings=["x", "y"])
            _tooltip_cols = [c for c in ["id", "__row_id", _x_col, y_col_plot] if c in df_plot.columns]
            if _color_tooltip and _color_tooltip in df_plot.columns and _color_tooltip not in _tooltip_cols:
                _tooltip_cols.append(_color_tooltip)
            chart = (
                alt.Chart(df_plot)
                .mark_circle(opacity=0.7, stroke=None, strokeWidth=0)
                .encode(
                    x=alt.X(_x_col, title=_x_col),
                    y=alt.Y(y_col_plot, title=_y_col),
                    color=_color_encoding,
                    tooltip=_tooltip_cols,
                )
                .add_params(_brush)
            )
    else:
        if _x_col not in _df_explorer_source.columns:
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
            df_plot = _df_explorer_source.select(dedupe_exprs([pl.col("__row_id"), pl.col("id"), pl.col(_x_col)]))
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
def _(mo):
    transient_cluster_hue_state, set_transient_cluster_hue_state = mo.state("Leiden cluster")
    transient_cluster_metric_state, set_transient_cluster_metric_state = mo.state(None)
    return (
        transient_cluster_hue_state,
        transient_cluster_metric_state,
        set_transient_cluster_hue_state,
        set_transient_cluster_metric_state,
    )


@app.cell
def _(
    build_friendly_column_labels,
    df_active,
    df_umap_overlay,
    mo,
    opal_campaign_info,
    pl,
    score_source_dropdown,
    safe_is_numeric,
    transient_cluster_hue_state,
    transient_cluster_metric_state,
):
    df_metric_source = df_umap_overlay if df_umap_overlay is not None else df_active
    metric_cols = [
        name for name, dtype in df_metric_source.schema.items() if safe_is_numeric(dtype) and dtype != pl.Boolean
    ]
    metric_cols = [_name for _name in metric_cols if not _name.startswith("opal__overlay__y_hat_")]
    _slug = opal_campaign_info.slug if opal_campaign_info is not None else None
    preferred_cols = []
    if _slug:
        preferred_cols.append(f"opal__{_slug}__latest_pred_scalar")
    preferred_cols.extend(
        [
            "opal__score__scalar",
            "opal__score__rank",
            "opal__overlay__score",
            "opal__overlay__logic_fidelity",
            "opal__overlay__effect_scaled",
            "opal__overlay__rank",
        ]
    )
    preferred_present = [col for col in preferred_cols if col in metric_cols]
    metric_cols = preferred_present + [col for col in metric_cols if col not in preferred_present]
    if not metric_cols:
        metric_cols = []
    default_metric = metric_cols[0] if metric_cols else None
    if "opal__score__scalar" in metric_cols:
        default_metric = "opal__score__scalar"
    elif _slug and f"opal__{_slug}__latest_pred_scalar" in metric_cols:
        default_metric = f"opal__{_slug}__latest_pred_scalar"
    elif "opal__overlay__score" in metric_cols:
        default_metric = "opal__overlay__score"
    _rf_prefix = "Overlay (artifact)"
    _score_source_label = score_source_dropdown.value if score_source_dropdown is not None else "Selected"
    _friendly_metric_labels = build_friendly_column_labels(
        score_source_label=_score_source_label,
        rf_prefix=_rf_prefix,
        campaign_slug=_slug,
    )
    transient_cluster_metric_label_map = {}
    metric_options = []
    for _col in metric_cols:
        _label = _friendly_metric_labels.get(_col, _col)
        if _label in transient_cluster_metric_label_map:
            _label = _col
        transient_cluster_metric_label_map[_label] = _col
        metric_options.append(_label)
    default_metric_label = None
    for _label, _col in transient_cluster_metric_label_map.items():
        if _col == default_metric:
            default_metric_label = _label
            break
    if default_metric_label is None:
        default_metric_label = default_metric or "(none)"
    _metric_state_value = transient_cluster_metric_state()
    if _metric_state_value in metric_cols:
        _metric_default_raw = _metric_state_value
    elif _metric_state_value in transient_cluster_metric_label_map:
        _metric_default_raw = transient_cluster_metric_label_map[_metric_state_value]
    else:
        _metric_default_raw = default_metric
    _metric_default_value = None
    for _label, _raw in transient_cluster_metric_label_map.items():
        if _raw == _metric_default_raw:
            _metric_default_value = _label
            break
    if _metric_default_value is None:
        _metric_default_value = default_metric_label
    if not metric_options:
        metric_options = ["(none)"]
        _metric_default_value = "(none)"
    transient_cluster_metric_dropdown = mo.ui.dropdown(
        options=metric_options,
        value=_metric_default_value,
        label="Cluster plot metric",
        full_width=True,
    )
    _raw_hue_values = ["(none)", "Leiden cluster"]
    _candidate_hues = [
        "opal__score__scalar",
        "opal__score__rank",
        "opal__score__top_k",
        "opal__overlay__score",
        "opal__overlay__logic_fidelity",
        "opal__overlay__effect_scaled",
        "opal__overlay__top_k",
        "opal__overlay__observed_event",
        "opal__overlay__sfxi_scored_label",
    ]
    _raw_hue_values.extend([col for col in _candidate_hues if col in df_metric_source.columns])
    _hue_raw_to_label = {
        "(none)": "(none)",
        "Leiden cluster": "Leiden cluster",
    }
    for _key, _label in _friendly_metric_labels.items():
        if _key not in _hue_raw_to_label:
            _hue_raw_to_label[_key] = _label
    transient_cluster_hue_label_map = {}
    hue_options = []
    for _raw in _raw_hue_values:
        _label = _hue_raw_to_label.get(_raw, _raw)
        if _label in transient_cluster_hue_label_map:
            _label = _raw
        transient_cluster_hue_label_map[_label] = _raw
        hue_options.append(_label)
    default_hue_label = None
    for _label, _raw in transient_cluster_hue_label_map.items():
        if _raw == "Leiden cluster":
            default_hue_label = _label
            break
    if default_hue_label is None:
        default_hue_label = "Leiden cluster"
    _hue_state_value = transient_cluster_hue_state()
    if _hue_state_value in _raw_hue_values:
        _hue_default_raw = _hue_state_value
    elif _hue_state_value in transient_cluster_hue_label_map:
        _hue_default_raw = transient_cluster_hue_label_map[_hue_state_value]
    else:
        _hue_default_raw = "Leiden cluster"
    _hue_default_value = None
    for _label, _raw in transient_cluster_hue_label_map.items():
        if _raw == _hue_default_raw:
            _hue_default_value = _label
            break
    if _hue_default_value is None:
        _hue_default_value = default_hue_label
    transient_cluster_hue_dropdown = mo.ui.dropdown(
        options=hue_options,
        value=_hue_default_value,
        label="Color by (hue)",
        full_width=True,
    )
    return (
        transient_cluster_hue_dropdown,
        transient_cluster_metric_dropdown,
        transient_cluster_hue_label_map,
        transient_cluster_metric_label_map,
    )


@app.cell
def _(mo):
    rf_model_source_value = "OPAL artifact (model.joblib)"
    rf_model_source_md = mo.md(f"**RF model source**: `{rf_model_source_value}`")
    return rf_model_source_md, rf_model_source_value


@app.cell
def _(mo):
    analysis_mode_radio = None
    return (analysis_mode_radio,)


@app.cell
def _(analysis_mode_radio):
    _unused = (analysis_mode_radio,)
    analysis_mode_value = None
    return (analysis_mode_value,)


@app.cell
def _():
    def build_overlay_cache_key(**kwargs):
        import json

        payload = {key: value for key, value in kwargs.items()}
        key = json.dumps(payload, sort_keys=True, default=str)
        return key, payload

    return (build_overlay_cache_key,)


@app.cell
def _():
    from dataclasses import dataclass
    from typing import Any

    @dataclass(frozen=True)
    class LabelViewInputs:
        df_labels_filtered: Any
        label_source_value: str | None
        label_view_value: str | None
        round_label: str | None
        round_value_map: dict[str, int | None]
        campaign_info: Any
        pred_run_id: str | None

    @dataclass(frozen=True)
    class OverlayComputeInputs:
        df_base: Any
        labels_asof_df: Any
        labels_current_df: Any
        df_sfxi: Any
        x_col: str | None
        y_col: str | None
        sfxi_params: Any
        artifact_result: Any
        dataset_path: Any
        opal_selected_round: int | None
        opal_pred_selected_round: int | None
        opal_pred_selected_run_id: str | None
        campaign_info: Any

    return LabelViewInputs, OverlayComputeInputs


@app.cell
def _():
    overlay_pred_cache = {}
    return (overlay_pred_cache,)


@app.cell
def _(mo):
    overlay_compute_button = mo.ui.run_button(label="Compute overlay predictions")
    overlay_clear_cache_button = mo.ui.run_button(label="Clear overlay cache")
    return overlay_clear_cache_button, overlay_compute_button


@app.cell
def _(
    set_transient_cluster_hue_state,
    set_transient_cluster_metric_state,
    transient_cluster_hue_dropdown,
    transient_cluster_hue_state,
    transient_cluster_hue_label_map,
    transient_cluster_metric_dropdown,
    transient_cluster_metric_state,
    transient_cluster_metric_label_map,
):
    _hue_label = transient_cluster_hue_dropdown.value
    _metric_label = transient_cluster_metric_dropdown.value
    _hue_value = transient_cluster_hue_label_map.get(_hue_label, _hue_label)
    _metric_value = transient_cluster_metric_label_map.get(_metric_label, _metric_label)
    if _hue_value != transient_cluster_hue_state():
        set_transient_cluster_hue_state(_hue_value)
    if _metric_value != transient_cluster_metric_state():
        set_transient_cluster_metric_state(_metric_value)
    return


@app.cell(column=1)
def _(
    campaign_dataset_names,
    dataset_form,
    dataset_preview_md,
    dataset_submit_notice_md,
    mo,
    usr_datasets,
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
        if not usr_datasets and campaign_dataset_names:
            usr_status = mo.md(
                f"USR root ({usr_root_mode}): `{usr_root}` · no datasets found; "
                "showing dataset names referenced by campaigns."
            )
        else:
            usr_status = mo.md(f"USR root ({usr_root_mode}): `{usr_root}`")
    mo.vstack([header, intro, usr_status, dataset_submit_notice_md, dataset_form, dataset_preview_md])
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
def _(
    build_umap_controls,
    df_active,
    mo,
    opal_campaign_info,
    score_source_dropdown,
):
    _score_source_value = score_source_dropdown.value if score_source_dropdown is not None else None
    if _score_source_value and (
        str(_score_source_value).startswith("(select") or str(_score_source_value).startswith("(no score")
    ):
        _score_source_value = None
    _slug = opal_campaign_info.slug if opal_campaign_info is not None else None
    controls = build_umap_controls(
        mo=mo,
        df_active=df_active,
        score_source_value=_score_source_value,
        campaign_slug=_slug,
    )
    umap_color_dropdown = controls.umap_color_dropdown
    umap_color_label_map = controls.umap_color_label_map
    umap_opacity_slider = controls.umap_opacity_slider
    umap_size_slider = controls.umap_size_slider
    umap_x_input = controls.umap_x_input
    umap_y_input = controls.umap_y_input
    return (
        umap_color_dropdown,
        umap_color_label_map,
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
    build_umap_explorer_chart,
    dataset_name,
    df_active,
    df_umap_overlay,
    mo,
    umap_color_dropdown,
    umap_color_label_map,
    umap_opacity_slider,
    umap_size_slider,
    umap_x_input,
    umap_y_input,
):
    _x_col = umap_x_input.value.strip()
    _y_col = umap_y_input.value.strip()
    if not hasattr(alt, "_DNAD_PLOT_SIZE"):
        alt._DNAD_PLOT_SIZE = 420
    _plot_size = alt._DNAD_PLOT_SIZE

    df_umap_source = df_umap_overlay if df_umap_overlay is not None else df_active
    _color_label = umap_color_dropdown.value
    _color_value = umap_color_label_map.get(_color_label, _color_label)

    _result = build_umap_explorer_chart(
        df=df_umap_source,
        x_col=_x_col,
        y_col=_y_col,
        color_value=_color_value,
        color_label=_color_label,
        point_size=umap_size_slider.value,
        opacity=umap_opacity_slider.value,
        plot_size=_plot_size,
        dataset_name=dataset_name,
    )
    df_umap_plot = _result.df_plot
    umap_valid = _result.valid
    umap_chart_note_md = mo.md(_result.note) if _result.note else mo.md("")
    umap_chart_ui = mo.ui.altair_chart(_result.chart)
    return df_umap_plot, umap_chart_note_md, umap_chart_ui, umap_valid


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


@app.cell
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
        "Note: exports are for EDA and may include overlay columns "
        "(`opal__overlay__*`), which are non-canonical."
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

    _status_lines = []
    if preview_path is not None:
        _status_lines.append(f"Mode: `{preview_mode}`")
        _status_lines.append(f"Exists: `{preview_exists}`")
    if preview_error:
        _status_lines.append(f"Error: {preview_error}")

    dataset_preview_md = mo.md("\n".join(_status_lines) if _status_lines else "")
    return (dataset_preview_md,)


@app.cell
def _(
    ledger_runs_df,
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
        run_id=opal_pred_selected_run_id,
        ledger_runs_df=ledger_runs_df,
        allow_fallback=False,
    )
    artifact_ready = artifact_result.use_artifact

    model_lines = ["**Model status**"]
    if artifact_ready:
        model_lines.append(f"- Model: OPAL artifact (run_id=`{opal_pred_selected_run_id}`)")
        if artifact_result.model_path is not None:
            model_lines.append(f"- Path: `{artifact_result.model_path}`")
    else:
        model_lines.append("- Model: OPAL artifact required but unavailable.")
        model_lines.append(f"- Details: {artifact_result.note}")

    if opal_pred_selected_round is None:
        model_lines.append("- Artifact round: **missing** (select a prediction round to enable overlay)")
    else:
        model_lines.append(f"- Artifact round: `R={opal_pred_selected_round}`")
    if opal_selected_round is None:
        model_lines.append("- Labels view round: `all` (no round filter)")
    else:
        model_lines.append(f"- Labels view round: `R={opal_selected_round}`")

    rf_model_source_note_md = mo.md("\n".join(model_lines))
    return artifact_result, rf_model_source_note_md


@app.cell
def _(
    build_label_sfxi_view,
    compute_sfxi_params,
    df_opal_overlay_base,
    mo,
    opal_campaign_info,
    opal_labels_current_df,
    opal_labels_view_df,
    opal_selected_round,
    resolve_sfxi_readiness,
    sfxi_beta_input,
    sfxi_fixed_params,
    sfxi_gamma_input,
    sfxi_p00_slider,
    sfxi_p01_slider,
    sfxi_p10_slider,
    sfxi_p11_slider,
):
    df_sfxi = df_opal_overlay_base.head(0)
    sfxi_meta_md = mo.md("")
    sfxi_notice_md = mo.md("")

    readiness = resolve_sfxi_readiness(opal_campaign_info)
    labels_ready = readiness.ready
    if readiness.notice:
        sfxi_notice_md = mo.md(readiness.notice)
    y_col = readiness.y_col
    x_col = readiness.x_col

    delta = float(sfxi_fixed_params.get("delta", 0.0))
    p = float(sfxi_fixed_params.get("percentile", 95.0))
    fallback_p = float(sfxi_fixed_params.get("fallback_percentile", p))
    min_n = int(sfxi_fixed_params.get("min_n", 5))
    eps = float(sfxi_fixed_params.get("eps", 1.0e-8))

    sfxi_params = compute_sfxi_params(
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
        table_df = label_view.table_df
        table_cols = label_view.table_cols
        sfxi_labels_table = (
            mo.ui.table(table_df.select(table_cols), page_size=5) if table_cols else mo.ui.table(table_df, page_size=5)
        )
        sfxi_meta_md = mo.vstack(
            [
                mo.md("Labels with computed SFXI metrics"),
                sfxi_labels_table,
            ]
        )

    return (
        df_sfxi,
        labels_ready,
        sfxi_meta_md,
        sfxi_notice_md,
        sfxi_params,
        readiness,
        x_col,
        y_col,
    )


@app.cell
def _(
    OverlayComputeInputs,
    artifact_result,
    dataset_path,
    df_opal_overlay_base,
    df_sfxi,
    opal_campaign_info,
    opal_labels_asof_df,
    opal_labels_current_df,
    opal_pred_selected_round,
    opal_pred_selected_run_id,
    opal_selected_round,
    sfxi_params,
    x_col,
    y_col,
):
    overlay_compute_inputs = OverlayComputeInputs(
        df_base=df_opal_overlay_base,
        labels_asof_df=opal_labels_asof_df,
        labels_current_df=opal_labels_current_df,
        df_sfxi=df_sfxi,
        x_col=x_col,
        y_col=y_col,
        sfxi_params=sfxi_params,
        artifact_result=artifact_result,
        dataset_path=dataset_path,
        opal_selected_round=opal_selected_round,
        opal_pred_selected_round=opal_pred_selected_round,
        opal_pred_selected_run_id=opal_pred_selected_run_id,
        campaign_info=opal_campaign_info,
    )
    return (overlay_compute_inputs,)


@app.cell
def _(
    overlay_compute_inputs,
    build_overlay_cache_key,
    compute_transient_overlay,
    dashboard_context,
    Diagnostics,
    labels_ready,
    mo,
    overlay_clear_cache_button,
    overlay_compute_button,
    overlay_pred_cache,
):
    df_overlay_raw = overlay_compute_inputs.df_base
    transient_diag = Diagnostics()
    transient_feature_chart = None
    transient_hist_chart = None
    overlay_cache_status_md = mo.md("")

    if overlay_clear_cache_button.value:
        overlay_pred_cache.clear()
        transient_diag = transient_diag.add_warning("Overlay cache cleared.")

    if not labels_ready:
        transient_diag = transient_diag.add_error("Overlay predictions unavailable (campaign unsupported).")
    else:

        def _stat_mtime(path):
            try:
                return path.stat().st_mtime
            except Exception:
                return None

        _dataset_mtime = (
            _stat_mtime(overlay_compute_inputs.dataset_path)
            if overlay_compute_inputs.dataset_path is not None
            else None
        )
        _artifact_path = overlay_compute_inputs.artifact_result.model_path
        _artifact_mtime = _stat_mtime(_artifact_path) if _artifact_path is not None else None
        _model_params_sig = None
        _y_ops_sig = None
        if overlay_compute_inputs.campaign_info is not None:
            import json as _json

            _model_params_sig = _json.dumps(
                overlay_compute_inputs.campaign_info.model_params or {}, sort_keys=True, default=str
            )
            _y_ops_sig = _json.dumps(overlay_compute_inputs.campaign_info.y_ops or [], sort_keys=True, default=str)

        overlay_cache_key, _overlay_cache_payload = build_overlay_cache_key(
            dataset_path=str(overlay_compute_inputs.dataset_path)
            if overlay_compute_inputs.dataset_path is not None
            else None,
            dataset_mtime=_dataset_mtime,
            campaign_slug=overlay_compute_inputs.campaign_info.slug
            if overlay_compute_inputs.campaign_info is not None
            else None,
            x_col=overlay_compute_inputs.x_col,
            artifact_path=str(_artifact_path) if _artifact_path is not None else None,
            artifact_mtime=_artifact_mtime,
            prediction_round=overlay_compute_inputs.opal_pred_selected_round,
            run_id=overlay_compute_inputs.opal_pred_selected_run_id,
            model_params_sig=_model_params_sig,
            y_ops_sig=_y_ops_sig,
        )

        cache_hit = overlay_cache_key in overlay_pred_cache
        compute_overlay = bool(overlay_compute_button.value) if overlay_compute_button is not None else False
        if not cache_hit and not compute_overlay:
            transient_diag = transient_diag.add_warning(
                "Overlay cache miss. Click 'Compute overlay predictions' to run the overlay model."
            )

        if overlay_compute_inputs.opal_pred_selected_run_id is None:
            transient_diag = transient_diag.add_error("Artifact overlay requires a run_id selection.")
        if overlay_compute_inputs.opal_pred_selected_round is None:
            transient_diag = transient_diag.add_error("Artifact overlay requires a round selection.")
        if not overlay_compute_inputs.artifact_result.use_artifact:
            transient_diag = transient_diag.add_error(
                "Artifact model unavailable; resolve the model.joblib for the selected run."
            )

        if not transient_diag.errors:
            overlay_round = overlay_compute_inputs.opal_pred_selected_round
            transient_result = compute_transient_overlay(
                df_base=overlay_compute_inputs.df_base,
                labels_asof_df=overlay_compute_inputs.labels_asof_df,
                labels_current_df=overlay_compute_inputs.labels_current_df,
                df_sfxi=overlay_compute_inputs.df_sfxi,
                x_col=overlay_compute_inputs.x_col,
                y_col=overlay_compute_inputs.y_col,
                sfxi_params=overlay_compute_inputs.sfxi_params,
                selected_round=overlay_round,
                artifact_model=overlay_compute_inputs.artifact_result.model,
                artifact_round_dir=overlay_compute_inputs.artifact_result.round_dir,
                run_id=overlay_compute_inputs.opal_pred_selected_run_id,
                context=dashboard_context,
                pred_cache=overlay_pred_cache,
                cache_key=overlay_cache_key,
                compute_if_missing=compute_overlay,
            )
            df_overlay_raw = transient_result.df_overlay
            transient_diag = transient_diag.merge(transient_result.diagnostics)
            if transient_result.feature_chart is not None:
                transient_feature_chart = mo.ui.altair_chart(transient_result.feature_chart)
            if transient_result.hist_chart is not None:
                hist_element = mo.ui.altair_chart(transient_result.hist_chart)
                if transient_result.hist_note:
                    transient_hist_chart = mo.vstack([hist_element, mo.md(transient_result.hist_note)])
                else:
                    transient_hist_chart = hist_element

        cache_lines = [
            f"- Cache entries: `{len(overlay_pred_cache)}`",
            f"- Cache hit: `{cache_hit}`",
            f"- Compute requested: `{compute_overlay}`",
            f"- Cache key: `{overlay_cache_key[:96]}`",
        ]
        if overlay_compute_inputs.opal_pred_selected_run_id is None:
            cache_lines.append("- Artifact overlay requires a run_id selection.")
        if overlay_compute_inputs.opal_pred_selected_round is None:
            cache_lines.append("- Artifact overlay requires a round selection.")
        overlay_cache_status_md = mo.md("\n".join(cache_lines))

    return (
        df_overlay_raw,
        overlay_cache_status_md,
        transient_diag,
        transient_feature_chart,
        transient_hist_chart,
    )


@app.cell
def _(
    alt,
    apply_overlay_label_flags,
    apply_score_overlay,
    artifact_result,
    build_umap_overlay_charts,
    dashboard_context,
    dataset_name,
    diagnostics_to_lines,
    df_overlay_raw,
    df_sfxi,
    Diagnostics,
    ScoreOverlayDiagnostics,
    mo,
    opal_campaign_info,
    opal_labels_view_df,
    opal_pred_selected_round,
    opal_selected_round,
    pl,
    score_source_dropdown,
    transient_diag,
):
    df_umap_overlay = df_overlay_raw
    _transient_diag = transient_diag

    overlay_cols = [
        "opal__overlay__score",
        "opal__overlay__rank",
        "opal__overlay__logic_fidelity",
        "opal__overlay__effect_scaled",
        "opal__overlay__top_k",
    ]

    _required_overlay_cols = ["opal__overlay__run_id", "opal__overlay__round_mode"]
    _missing_overlay = [col for col in _required_overlay_cols if col not in df_umap_overlay.columns]
    if _missing_overlay:
        _transient_diag = _transient_diag.add_error(
            "Overlay provenance contract violated; missing required columns: "
            f"{_missing_overlay}. This is a bug — please report."
        )
        df_umap_overlay = df_umap_overlay.with_columns([pl.lit(None).alias(col) for col in _missing_overlay])
    if (
        df_umap_overlay.height
        and "opal__overlay__round" in df_umap_overlay.columns
        and "opal__overlay__round_mode" in df_umap_overlay.columns
    ):
        if not (artifact_result.requested_artifact and opal_pred_selected_round is None):
            _round_missing = df_umap_overlay.select(
                ((pl.col("opal__overlay__round_mode") == "explicit") & pl.col("opal__overlay__round").is_null()).sum()
            ).item()
            if _round_missing:
                _transient_diag = _transient_diag.add_error(
                    "Overlay round provenance missing while round_mode=explicit. "
                    "Select a round or switch to headless overlay."
                )

    missing_overlay_cols = [c for c in overlay_cols if c not in df_umap_overlay.columns]
    if missing_overlay_cols:
        fill_exprs = []
        for _col in missing_overlay_cols:
            if _col == "opal__overlay__top_k":
                fill_exprs.append(pl.lit(False).alias(_col))
            elif _col == "opal__overlay__rank":
                fill_exprs.append(pl.lit(None, dtype=pl.Int64).alias(_col))
            else:
                fill_exprs.append(pl.lit(None, dtype=pl.Float64).alias(_col))
        df_umap_overlay = df_umap_overlay.with_columns(fill_exprs)

    _score_source_value = score_source_dropdown.value if score_source_dropdown is not None else None
    score_source_selected = bool(_score_source_value)
    if score_source_selected:
        df_umap_overlay, score_diag = apply_score_overlay(
            df_umap_overlay,
            score_source_value=_score_source_value,
            selected_round=opal_selected_round,
            context=dashboard_context,
        )
        _transient_diag = _transient_diag.merge(score_diag.diagnostics)
    else:
        score_diag = ScoreOverlayDiagnostics(
            source_key="unselected",
            warnings=[],
            scalar_col=None,
            rank_col=None,
            top_k_col=None,
            diagnostics=Diagnostics(errors=["Select a score source to populate opal__score__* columns."]),
        )
        _transient_diag = _transient_diag.merge(score_diag.diagnostics)
    transient_lines = diagnostics_to_lines(_transient_diag)
    transient_md = mo.md("\n".join(transient_lines)) if transient_lines else None

    df_umap_overlay = apply_overlay_label_flags(
        df_overlay=df_umap_overlay,
        labels_view_df=opal_labels_view_df,
        df_sfxi=df_sfxi,
        label_src="ingest_y",
        id_col="id",
    )

    rf_umap_cluster_chart = None
    rf_umap_score_chart = None
    if not hasattr(alt, "_DNAD_PLOT_SIZE"):
        alt._DNAD_PLOT_SIZE = 420
    _plot_size = alt._DNAD_PLOT_SIZE
    _slug = opal_campaign_info.slug if opal_campaign_info is not None else None
    _cluster_chart, _score_chart = build_umap_overlay_charts(
        df=df_umap_overlay,
        dataset_name=dataset_name,
        campaign_slug=_slug,
        use_artifact=True,
        plot_size=_plot_size,
    )
    if _cluster_chart is not None:
        rf_umap_cluster_chart = mo.ui.altair_chart(_cluster_chart)
    if _score_chart is not None:
        rf_umap_score_chart = mo.ui.altair_chart(_score_chart)
    _unused_rf = (rf_umap_cluster_chart, rf_umap_score_chart)
    return (
        df_umap_overlay,
        transient_md,
        score_diag,
    )


@app.cell
def _(
    build_cluster_chart,
    dataset_name,
    df_umap_overlay,
    mo,
    score_source_dropdown,
    transient_cluster_hue_dropdown,
    transient_cluster_hue_label_map,
    transient_cluster_metric_dropdown,
    transient_cluster_metric_label_map,
):
    _transient_cluster_chart = None
    _metric_label = transient_cluster_metric_dropdown.value
    _metric_col = transient_cluster_metric_label_map.get(_metric_label, _metric_label)
    _hue_label_display = transient_cluster_hue_dropdown.value
    _hue_value = transient_cluster_hue_label_map.get(_hue_label_display, _hue_label_display)
    if _metric_col and "cluster__ldn_v1" in df_umap_overlay.columns and _metric_col in df_umap_overlay.columns:
        _id_col = "id" if "id" in df_umap_overlay.columns else "__row_id"
        _rf_prefix = "Overlay (artifact)"
        _score_source_label = score_source_dropdown.value if score_source_dropdown is not None else "Selected"
        _cluster_chart = build_cluster_chart(
            df=df_umap_overlay,
            cluster_col="cluster__ldn_v1",
            metric_col=_metric_col,
            metric_label=_metric_label or _metric_col,
            hue_value=_hue_value,
            hue_label_display=_hue_label_display,
            rf_prefix=_rf_prefix,
            score_source_label=_score_source_label,
            dataset_name=dataset_name,
            id_col=_id_col,
            title="Overlay metric by Leiden cluster",
        )
        if _cluster_chart is not None:
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
    df_active,
    df_score_top_k_pool,
    df_sfxi_selected,
    df_overlay_top_k_pool,
    df_umap_selected,
    dataset_name,
    export_button,
    export_format_dropdown,
    export_source_dropdown,
    export_source_label_map,
    mo,
    opal_campaign_info,
    dataset_path,
    pl,
    repo_root,
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

        if df_export is not None and not df_export.is_empty():
            _overlay_cols = [c for c in df_export.columns if c.startswith("opal__overlay__")]
            _score_cols = [c for c in df_export.columns if c.startswith("opal__score__")]
            if _overlay_cols and _score_cols:
                export_note_md = mo.md(
                    "**Export note**: this export mixes overlay columns with canonical/cache score columns. "
                    "Overlay columns are non-canonical and not persisted."
                )
            elif _overlay_cols:
                export_note_md = mo.md(
                    "**Export note**: this export includes overlay columns only. "
                    "Overlay results are non-canonical and not persisted."
                )

        if df_export is None or df_export.is_empty():
            export_status_md = mo.md("Nothing to export.")
        else:
            import json as _json
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
            out_path = _export_out_dir / f"promoter_eda_{dataset_slug}_{ts}.{suffix}"
            if suffix == "csv":
                df_export.write_csv(out_path)
            else:
                df_export.write_parquet(out_path)
            manifest_path = None
            if _overlay_cols:

                def _first_non_null(col: str):
                    if col not in df_export.columns:
                        return None
                    series = df_export.select(pl.col(col).drop_nulls().head(1)).to_series()
                    if series.len() == 0:
                        return None
                    return series[0]

                manifest = {
                    "exported_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "dataset_path": str(dataset_path) if dataset_path is not None else None,
                    "campaign_slug": opal_campaign_info.slug if opal_campaign_info is not None else None,
                    "export_source": source_label,
                    "export_format": suffix,
                    "rows": int(df_export.height),
                    "columns": int(len(df_export.columns)),
                    "overlay": {
                        "present": True,
                        "source": _first_non_null("opal__overlay__source"),
                        "run_id": _first_non_null("opal__overlay__run_id"),
                        "round": _first_non_null("opal__overlay__round"),
                        "round_mode": _first_non_null("opal__overlay__round_mode"),
                        "campaign_slug": _first_non_null("opal__overlay__campaign_slug"),
                        "score_source": _first_non_null("opal__score__source"),
                    },
                }
                manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
                try:
                    manifest_path.write_text(_json.dumps(manifest, indent=2))
                except Exception as exc:
                    export_note_md = mo.md(f"**Export note**: failed to write overlay manifest ({exc}).")

            status_line = (
                f"Saved `{out_path}` from `{source_label}` ({df_export.height} rows × {len(df_export.columns)} cols)."
            )
            if manifest_path is not None:
                status_line += f" Manifest: `{manifest_path}`"
            export_status_md = mo.md(status_line)
    return export_note_md, export_status_md


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
def _(campaign_labels, campaign_path_map, default_campaign_label, df_active, mo):
    _unused = (df_active,)
    opal_panel_prefix_dropdown = mo.ui.dropdown(
        options=campaign_labels,
        value=default_campaign_label,
        label="Campaign config",
        full_width=True,
    )
    opal_panel_prefix_default = default_campaign_label
    return (
        campaign_path_map,
        opal_panel_prefix_default,
        opal_panel_prefix_dropdown,
    )


@app.cell
def _(
    build_label_events,
    build_label_events_from_ledger,
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
    opal_label_diag = None
    opal_label_hist_col = None
    opal_campaign_path = None
    opal_campaign_notice_md = mo.md("")

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
                opal_campaign_path = campaign_path
            except Exception as exc:
                opal_campaign_info = None
                opal_campaign_notice_md = mo.md(
                    "\n".join(
                        [
                            "**Campaign unsupported**",
                            f"Config: `{campaign_path}`",
                            f"Error: `{exc}`",
                        ]
                    )
                )

    if opal_campaign_info is not None:
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
        opal_label_events_df,
        opal_label_diag,
        opal_label_hist_col,
        opal_campaign_path,
    )


@app.cell
def _(
    load_ledger_labels,
    load_ledger_runs,
    opal_campaign_info,
    resolve_campaign_workdir,
):
    opal_workdir = None
    ledger_runs_df = None
    ledger_runs_diag = None
    ledger_labels_df = None
    ledger_labels_diag = None
    if opal_campaign_info is not None:
        opal_workdir = resolve_campaign_workdir(opal_campaign_info)
        ledger_runs_df, ledger_runs_diag = load_ledger_runs(opal_workdir)
        ledger_labels_df, ledger_labels_diag = load_ledger_labels(opal_workdir)
    return (
        ledger_labels_df,
        ledger_labels_diag,
        ledger_runs_df,
        ledger_runs_diag,
        opal_workdir,
    )


@app.cell
def _(
    df_active,
    ledger_runs_df,
    mo,
    opal_campaign_info,
    pl,
):
    opal_pred_round_dropdown = None
    opal_run_id_dropdown = None
    opal_pred_round_value_map = {}
    opal_pred_selected_round = None
    opal_pred_selected_run_id = None

    if ledger_runs_df is not None and not ledger_runs_df.is_empty() and "as_of_round" in ledger_runs_df.columns:
        _rounds = ledger_runs_df.select(pl.col("as_of_round").drop_nulls().unique()).to_series().to_list()
        _rounds = sorted({int(_r) for _r in _rounds})
        if _rounds:
            _round_prompt = "(select round)"
            opal_pred_round_value_map = {_round_prompt: None}
            for _r in _rounds:
                opal_pred_round_value_map[f"R={_r}"] = _r
            _round_options = [_round_prompt] + [f"R={_r}" for _r in _rounds]
            _round_default = _round_prompt
        else:
            _round_options = ["(no rounds available)"]
            _round_default = _round_options[0]
            opal_pred_round_value_map = {_round_default: None}
        opal_pred_round_dropdown = mo.ui.dropdown(
            options=_round_options,
            value=_round_default,
            label="Prediction round (ledger)",
            full_width=True,
        )
        _selected_label = opal_pred_round_dropdown.value
        opal_pred_selected_round = opal_pred_round_value_map.get(_selected_label)

        if opal_pred_selected_round is not None:
            run_ids = []
            if "run_id" in ledger_runs_df.columns:
                run_ids = (
                    ledger_runs_df.filter(pl.col("as_of_round") == int(opal_pred_selected_round))
                    .select(pl.col("run_id").drop_nulls().unique())
                    .to_series()
                    .to_list()
                )
            run_ids = sorted({str(_r) for _r in run_ids})
            if run_ids:
                _run_id_prompt = "(select run_id)"
                _run_id_options = [_run_id_prompt] + run_ids
                _run_id_default = _run_id_prompt
                opal_run_id_dropdown = mo.ui.dropdown(
                    options=_run_id_options,
                    value=_run_id_default,
                    label="Run ID (ledger)",
                    full_width=True,
                )
                _selected_run = opal_run_id_dropdown.value
                opal_pred_selected_run_id = None if _selected_run == _run_id_prompt else _selected_run

    _slug = opal_campaign_info.slug if opal_campaign_info is not None else None
    _cache_col = f"opal__{_slug}__latest_pred_scalar" if _slug else None
    score_source_options = ["Overlay (RF)"]
    if ledger_runs_df is not None and not ledger_runs_df.is_empty() and opal_pred_selected_run_id is not None:
        score_source_options.append("Ledger predictions (run-aware)")
    if _cache_col and _cache_col in df_active.columns:
        score_source_options.append("Records cache (latest_pred_scalar)")
    score_source_default = "Overlay (RF)"
    score_source_dropdown = mo.ui.dropdown(
        options=score_source_options,
        value=score_source_default,
        label="Score source",
        full_width=True,
    )
    return (
        opal_pred_round_dropdown,
        opal_run_id_dropdown,
        opal_pred_selected_round,
        opal_pred_selected_run_id,
        score_source_dropdown,
    )


@app.cell
def _(
    build_label_events_from_ledger,
    dashboard_context,
    df_active,
    ledger_labels_df,
):
    opal_label_events_ledger = ledger_labels_df.head(0) if ledger_labels_df is not None else df_active.head(0)
    if ledger_labels_df is not None and not ledger_labels_df.is_empty():
        _label_events = build_label_events_from_ledger(
            ledger_labels_df=ledger_labels_df,
            df_active=df_active,
            y_col_name=None,
            x_col_name=None,
            source_kind="ledger",
            context=dashboard_context,
        )
        opal_label_events_ledger = _label_events.df
    return (opal_label_events_ledger,)


@app.cell
def _(
    mo,
    opal_label_events_df,
    opal_label_events_ledger,
    opal_label_diag,
):
    label_source_dropdown = None
    label_source_notice_md = mo.md("")

    label_sources = []
    if opal_label_diag is not None:
        label_sources.append("Records label_hist (cache)")
    if opal_label_events_ledger is not None and not opal_label_events_ledger.is_empty():
        label_sources.append("Ledger labels (canonical)")
    if not label_sources:
        label_sources = ["(no label sources available)"]
    if "Records label_hist (cache)" in label_sources:
        label_source_default = "Records label_hist (cache)"
    else:
        label_source_default = label_sources[0]

    label_source_dropdown = mo.ui.dropdown(
        options=label_sources,
        value=label_source_default,
        label="Label data source",
        full_width=True,
    )

    if opal_label_diag and opal_label_diag.status in {
        "parse_warning",
        "missing_column",
        "error",
        "empty_df",
    }:
        err_preview = ""
        if opal_label_diag.errors:
            err_preview = f"Errors: `{opal_label_diag.errors}`"
        diag_warns = ""
        diag_errors = ""
        if getattr(opal_label_diag, "diagnostics", None) is not None:
            if opal_label_diag.diagnostics.warnings:
                diag_warns = f"Diagnostics warnings: `{opal_label_diag.diagnostics.warnings}`"
            if opal_label_diag.diagnostics.errors:
                diag_errors = f"Diagnostics errors: `{opal_label_diag.diagnostics.errors}`"
        schema_text = ""
        schema_val = opal_label_diag.schema
        if schema_val:
            schema_text = str(schema_val)
            if len(schema_text) > 400:
                schema_text = schema_text[:397] + "..."
        exception_text = opal_label_diag.exception
        label_source_notice_md = mo.md(
            "\n".join(
                [
                    "**Label history warning**",
                    f"Status: `{opal_label_diag.status}`",
                    f"Column: `{opal_label_diag.label_hist_col}`",
                    f"Dtype: `{opal_label_diag.dtype}`",
                    f"Schema: `{schema_text}`" if schema_text else "Schema: `(unavailable)`",
                    f"Sample: `{opal_label_diag.sample}`",
                    f"Message: {opal_label_diag.message or ''}",
                    f"Exception: `{exception_text}`" if exception_text else "",
                    f"Remediation: {opal_label_diag.suggested_remediation or ''}",
                    diag_warns,
                    diag_errors,
                    err_preview,
                ]
            )
        )

    return (
        label_source_dropdown,
        label_source_notice_md,
        label_sources,
    )


@app.cell
def _(
    label_source_dropdown,
    label_sources,
    opal_label_events_df,
    opal_label_events_ledger,
    mo,
    pl,
):
    df_labels = pl.DataFrame()
    label_src_multiselect = None
    label_source_value = None
    label_source_selection_notice_md = mo.md("")

    if label_source_dropdown is not None and label_sources != ["(no label sources available)"]:
        selected_source = label_source_dropdown.value
        label_source_value = selected_source
        if selected_source == "Ledger labels (canonical)":
            df_labels = opal_label_events_ledger
        elif selected_source == "Records label_hist (cache)":
            df_labels = opal_label_events_df
    else:
        label_source_selection_notice_md = mo.md("No label sources available for the selected campaign.")

    label_src_values = []
    if df_labels is not None and not df_labels.is_empty() and "label_src" in df_labels.columns:
        label_src_values = df_labels.select(pl.col("label_src").drop_nulls().unique()).to_series().to_list()
    label_src_values = sorted({str(x) for x in label_src_values if x is not None})
    if label_src_values:
        label_src_multiselect = mo.ui.multiselect(
            options=label_src_values,
            value=label_src_values,
            label="Label sources (filter; default ALL)",
            full_width=True,
        )

    return df_labels, label_src_multiselect, label_source_selection_notice_md, label_source_value


@app.cell
def _(label_source_value, ledger_runs_df, mo):
    label_rounds_from_ledger_checkbox = None
    label_rounds_from_ledger_note_md = mo.md("")
    if label_source_value is not None:
        if ledger_runs_df is None or ledger_runs_df.is_empty():
            label_rounds_from_ledger_note_md = mo.md("Ledger runs unavailable; round derivation is disabled.")
        else:
            label_rounds_from_ledger_checkbox = mo.ui.checkbox(
                label="Include ledger-derived rounds (explicit)",
                value=False,
            )
    return label_rounds_from_ledger_checkbox, label_rounds_from_ledger_note_md


@app.cell
def _(
    df_labels,
    label_source_value,
    label_src_multiselect,
    label_rounds_from_ledger_checkbox,
    ledger_runs_df,
    mo,
    pl,
):
    opal_round_dropdown = None
    opal_round_value_map = {}

    df_labels_filtered = df_labels
    if label_source_value is not None:
        if (
            df_labels_filtered is not None
            and not df_labels_filtered.is_empty()
            and label_src_multiselect is not None
            and "label_src" in df_labels_filtered.columns
        ):
            selected_sources = label_src_multiselect.value
            if selected_sources:
                df_labels_filtered = df_labels_filtered.filter(pl.col("label_src").is_in(selected_sources))

    if label_source_value is not None:
        _rounds = []
        _round_source = "labels"
        if (
            df_labels_filtered is not None
            and not df_labels_filtered.is_empty()
            and "observed_round" in df_labels_filtered.columns
        ):
            _rounds = df_labels_filtered.select(pl.col("observed_round").drop_nulls().unique()).to_series().to_list()
            _rounds = sorted({int(_r) for _r in _rounds})
        _use_ledger_rounds = (
            bool(label_rounds_from_ledger_checkbox.value) if label_rounds_from_ledger_checkbox is not None else False
        )
        if not _rounds and _use_ledger_rounds and ledger_runs_df is not None and not ledger_runs_df.is_empty():
            if "as_of_round" in ledger_runs_df.columns:
                _rounds = ledger_runs_df.select(pl.col("as_of_round").drop_nulls().unique()).to_series().to_list()
                _rounds = sorted({int(_r) for _r in _rounds})
                _round_source = "ledger"
        if _rounds:
            _all_rounds_label = "All rounds (no filter)"
            opal_round_value_map = {_all_rounds_label: None}
            for _r in _rounds:
                if _round_source == "ledger":
                    opal_round_value_map[f"R={_r} (ledger)"] = _r
                else:
                    opal_round_value_map[f"R={_r}"] = _r
            _round_options = [_all_rounds_label] + [
                label for label in opal_round_value_map.keys() if label != _all_rounds_label
            ]
            _round_default = _all_rounds_label
        else:
            _all_rounds_label = "All rounds (no filter)"
            _round_options = [_all_rounds_label]
            _round_default = _all_rounds_label
            opal_round_value_map = {_all_rounds_label: None}
        opal_round_dropdown = mo.ui.dropdown(
            options=_round_options,
            value=_round_default,
            label="Labels as of round",
            full_width=True,
        )
    return df_labels_filtered, opal_round_dropdown, opal_round_value_map


@app.cell
def _(
    Diagnostics,
    ledger_diagnostics,
    load_ledger_predictions,
    opal_pred_selected_round,
    opal_pred_selected_run_id,
    opal_workdir,
    pl,
):
    ledger_preds_df = pl.DataFrame()
    ledger_preds_diag = None
    if opal_workdir is not None and opal_pred_selected_run_id:
        ledger_preds_df, ledger_preds_diag = load_ledger_predictions(
            opal_workdir,
            run_id=opal_pred_selected_run_id,
            as_of_round=opal_pred_selected_round,
        )
        if ledger_preds_df is not None and not ledger_preds_df.is_empty() and "id" in ledger_preds_df.columns:
            ledger_preds_df = ledger_preds_df.with_columns(pl.col("id").cast(pl.Utf8))
    elif opal_workdir is not None and not opal_pred_selected_run_id:
        missing_msg = "Select a run_id to load ledger predictions."
        ledger_preds_diag = ledger_diagnostics(
            status="missing_run_id",
            path=str(opal_workdir / "outputs" / "ledger.predictions"),
            rows=0,
            error=missing_msg,
            diagnostics=Diagnostics().add_error(missing_msg),
            run_id=None,
            as_of_round=opal_pred_selected_round,
        )
    return ledger_preds_df, ledger_preds_diag


@app.cell
def _(
    DashboardContext,
    dataset_name,
    dataset_path,
    ledger_labels_df,
    ledger_preds_df,
    ledger_runs_df,
    opal_campaign_info,
    opal_workdir,
):
    dashboard_context = DashboardContext(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        campaign_info=opal_campaign_info,
        workdir=opal_workdir,
        ledger_runs_df=ledger_runs_df,
        ledger_labels_df=ledger_labels_df,
        ledger_preds_df=ledger_preds_df,
    )
    return (dashboard_context,)


@app.cell
def _(
    cache_pred_meta,
    cache_warning,
    ledger_warning,
    dataset_name,
    dataset_path,
    ledger_labels_diag,
    ledger_preds_diag,
    ledger_runs_df,
    ledger_runs_diag,
    mo,
    opal_campaign_info,
    opal_campaign_path,
    opal_pred_selected_round,
    opal_pred_selected_run_id,
    opal_workdir,
    rf_model_source_value,
    score_diag,
    score_source_dropdown,
):
    def _format_diag(label, diag) -> str | None:
        if diag is None:
            return None
        status = getattr(diag, "status", "unknown")
        rows = getattr(diag, "rows", None)
        parts = [f"{label}: status=`{status}`"]
        if rows is not None:
            parts.append(f"rows={rows}")
        detail = None
        diag_obj = getattr(diag, "diagnostics", None)
        details = []
        if diag_obj is not None:
            if diag_obj.errors:
                details.extend(diag_obj.errors)
            if diag_obj.warnings:
                details.extend(diag_obj.warnings)
        elif getattr(diag, "warnings", None):
            details.extend(diag.warnings)
        if details:
            detail = str(details[0])
            if len(details) > 1:
                detail = f"{detail} (+{len(details) - 1} more)"
        if detail:
            parts.append(detail)
        return "- " + " · ".join(parts)

    _score_source_value = score_source_dropdown.value if score_source_dropdown is not None else None

    summary_lines = ["### Provenance (summary)"]
    if dataset_name:
        summary_lines.append(f"- Dataset: `{dataset_name}`")
    if opal_campaign_info is not None:
        summary_lines.append(f"- Campaign: `{opal_campaign_info.slug}`")
    if _score_source_value:
        summary_lines.append(f"- Score source: `{_score_source_value}`")
    if rf_model_source_value:
        summary_lines.append(f"- Overlay model: `{rf_model_source_value}`")
    _summary_md = mo.md("\n".join(summary_lines))

    details_lines = ["### Provenance details"]
    dataset_lines = []
    if dataset_path is not None:
        dataset_lines.append(f"- Path: `{dataset_path}`")
    if dataset_lines:
        details_lines.extend(["**Dataset**", *dataset_lines])

    campaign_lines = []
    if opal_campaign_path is not None:
        campaign_lines.append(f"- Config: `{opal_campaign_path}`")
    if opal_campaign_info is not None:
        campaign_lines.append(f"- Slug: `{opal_campaign_info.slug}`")
    if opal_workdir is not None:
        campaign_lines.append(f"- Workdir: `{opal_workdir}`")
        campaign_lines.append(f"- Outputs: `{opal_workdir / 'outputs'}`")
    if campaign_lines:
        details_lines.extend(["**Campaign**", *campaign_lines])

    ledger_lines = []
    _runs_line = _format_diag("Ledger runs", ledger_runs_diag)
    if _runs_line:
        ledger_lines.append(_runs_line)
    _labels_line = _format_diag("Ledger labels", ledger_labels_diag)
    if _labels_line:
        ledger_lines.append(_labels_line)
    _preds_line = _format_diag("Ledger preds", ledger_preds_diag)
    if _preds_line:
        ledger_lines.append(_preds_line)
    if ledger_lines:
        details_lines.extend(["**Ledger**", *ledger_lines])

    selection_lines = []
    if rf_model_source_value or (ledger_runs_df is not None and not ledger_runs_df.is_empty()):
        selection_lines.append(
            f"- Selected round: `{opal_pred_selected_round}`"
            if opal_pred_selected_round is not None
            else "- Selected round: `(none)`"
        )
        selection_lines.append(f"- Selected run_id: `{opal_pred_selected_run_id}`")
        if (
            ledger_runs_df is not None
            and not ledger_runs_df.is_empty()
            and opal_pred_selected_round is not None
            and "as_of_round" in ledger_runs_df.columns
            and "run_id" in ledger_runs_df.columns
        ):
            _df_round = ledger_runs_df.filter(ledger_runs_df["as_of_round"] == int(opal_pred_selected_round))
            _n_runs = int(_df_round["run_id"].n_unique()) if not _df_round.is_empty() else 0
            if _n_runs > 1:
                selection_lines.append(
                    f"- Run warning: {int(_n_runs)} run_id values exist for round {opal_pred_selected_round}. "
                    "Select a run_id explicitly to avoid mixing reruns."
                )
                if opal_pred_selected_run_id is None:
                    selection_lines.append("- Run warning: run_id not selected; ledger predictions are disabled.")
    if score_source_dropdown is not None and _score_source_value:
        selection_lines.append(f"- Score source: `{_score_source_value}`")
        if _score_source_value.startswith("Records cache"):
            selection_lines.append("- Cache note: latest_pred_scalar is not run-aware; compare with caution.")
            if cache_pred_meta:
                cache_run = cache_pred_meta.get("run_id")
                cache_round = cache_pred_meta.get("as_of_round")
                cache_ts = cache_pred_meta.get("written_at")
                selection_lines.append(
                    f"- Cache provenance: run_id=`{cache_run}` round=`{cache_round}` written_at=`{cache_ts}`"
                )
                if not cache_run or cache_round is None:
                    selection_lines.append(
                        "- Cache warning: cache provenance missing run_id/round; comparisons are not run-aware."
                    )
                elif opal_pred_selected_run_id and str(cache_run) != str(opal_pred_selected_run_id):
                    selection_lines.append(
                        f"- Cache warning: cache run_id {cache_run} != selected run_id {opal_pred_selected_run_id}"
                    )
            else:
                selection_lines.append(
                    "- Cache warning: cache provenance columns missing; comparisons are not run-aware."
                )
    if selection_lines:
        details_lines.extend(["**Selections**", *selection_lines])

    warning_lines = []
    if score_diag is not None:
        diag_obj = getattr(score_diag, "diagnostics", None)
        if diag_obj is not None:
            if diag_obj.errors:
                warning_lines.append(f"- Score errors: `{diag_obj.errors}`")
            if diag_obj.warnings:
                warning_lines.append(f"- Score warnings: `{diag_obj.warnings}`")
        elif getattr(score_diag, "warnings", None):
            warning_lines.append(f"- Score warnings: `{score_diag.warnings}`")
    if cache_warning and _score_source_value and _score_source_value.startswith("Records cache"):
        warning_lines.append(f"- Cache warning: {cache_warning}")
    if ledger_warning and _score_source_value and _score_source_value.startswith("Ledger"):
        warning_lines.append(f"- Ledger warning: {ledger_warning}")
    if warning_lines:
        details_lines.extend(["**Warnings**", *warning_lines])

    notes = []
    if _score_source_value and _score_source_value.startswith("Overlay"):
        notes.append(
            "- Overlay run_id (`opal__overlay__run_id`) records the artifact run used for predictions; "
            "overlay scores remain non-canonical."
        )
    notes.append("- Running rounds does not create labels. Labels come from ingest-y / attach / external measurements.")
    if notes:
        details_lines.extend(["**Notes**", *notes])

    details_md = mo.md("\n".join(details_lines))
    opal_provenance_md = mo.ui.tabs({"Summary": _summary_md, "Details": details_md})
    return (opal_provenance_md,)


@app.cell
def _(
    ledger_runs_df,
    mo,
    opal_pred_selected_run_id,
    pl,
):
    default_selection_csv = ""
    if (
        ledger_runs_df is not None
        and not ledger_runs_df.is_empty()
        and opal_pred_selected_run_id
        and "run_id" in ledger_runs_df.columns
        and "artifacts" in ledger_runs_df.columns
    ):
        rows = (
            ledger_runs_df.filter(pl.col("run_id") == str(opal_pred_selected_run_id))
            .select(pl.col("artifacts"))
            .to_series()
            .to_list()
        )
        if rows:
            _artifacts = rows[0]
            if isinstance(_artifacts, dict):
                preferred_keys = [
                    f"selection_top_k__run_{opal_pred_selected_run_id}.csv",
                    "selection_top_k.csv",
                ]
                for key in preferred_keys:
                    if key not in _artifacts:
                        continue
                    sel_entry = _artifacts.get(key)
                    if isinstance(sel_entry, (list, tuple)) and len(sel_entry) >= 2:
                        default_selection_csv = str(sel_entry[1])
                        break
                    if isinstance(sel_entry, str):
                        default_selection_csv = sel_entry
                        break
    selection_csv_input = mo.ui.text(
        value=default_selection_csv,
        label="selection_top_k.csv path (for mismatch debugger)",
        full_width=True,
    )
    return default_selection_csv, selection_csv_input


@app.cell
def _(
    compare_selection_to_ledger,
    dashboard_context,
    ledger_preds_df,
    mo,
    opal_pred_selected_run_id,
    selection_csv_input,
):
    csv_path = selection_csv_input.value.strip() if selection_csv_input is not None else ""
    result = compare_selection_to_ledger(
        selection_path=csv_path,
        context=dashboard_context,
        ledger_preds_df=ledger_preds_df,
        selected_run_id=opal_pred_selected_run_id,
    )
    mismatch_md = mo.md(result.message)
    mismatch_table = mo.ui.table(result.table, page_size=10)
    return mismatch_md, mismatch_table


@app.cell
def _(
    df_active,
    ensure_selection_columns,
    ledger_preds_df,
    opal_campaign_info,
    pl,
):
    df_opal_overlay_base = df_active
    _cache_col = None
    cache_warning = None
    ledger_warning = None
    cache_pred_meta = {}

    if ledger_preds_df is not None and not ledger_preds_df.is_empty() and "id" in ledger_preds_df.columns:
        ledger_cols = ["id"]
        for _col in ["pred__y_obj_scalar", "sel__rank_competition", "sel__is_selected", "run_id", "as_of_round"]:
            if _col in ledger_preds_df.columns:
                ledger_cols.append(_col)
        df_ledger = ledger_preds_df.select(ledger_cols).with_columns(pl.col("id").cast(pl.Utf8))
        df_ledger = df_ledger.rename(
            {
                "pred__y_obj_scalar": "opal__ledger__score",
                "sel__rank_competition": "opal__ledger__rank",
                "sel__is_selected": "opal__ledger__top_k",
                "run_id": "opal__ledger__run_id",
                "as_of_round": "opal__ledger__round",
            }
        )
        if (
            "opal__ledger__rank" not in df_ledger.columns or "opal__ledger__top_k" not in df_ledger.columns
        ) and "opal__ledger__score" in df_ledger.columns:
            if opal_campaign_info is None:
                ledger_warning = (
                    "Ledger selection fields missing and campaign config unavailable; cannot reconstruct rank/top_k."
                )
            else:
                _sel_params = dict(opal_campaign_info.selection_params or {})
                df_ledger, _warnings, _err = ensure_selection_columns(
                    df_ledger,
                    id_col="id",
                    score_col="opal__ledger__score",
                    selection_params=_sel_params,
                    rank_col="opal__ledger__rank",
                    top_k_col="opal__ledger__top_k",
                )
                if _warnings:
                    ledger_warning = "; ".join(_warnings)
                if _err:
                    ledger_warning = _err if ledger_warning is None else f"{ledger_warning}; {_err}"
        df_opal_overlay_base = df_opal_overlay_base.join(df_ledger, on="id", how="left")

    if opal_campaign_info is not None:
        _cache_col = f"opal__{opal_campaign_info.slug}__latest_pred_scalar"
        cache_run_col = f"opal__{opal_campaign_info.slug}__latest_pred_run_id"
        cache_round_col = f"opal__{opal_campaign_info.slug}__latest_pred_as_of_round"
        cache_ts_col = f"opal__{opal_campaign_info.slug}__latest_pred_written_at"
    else:
        cache_run_col = None
        cache_round_col = None
        cache_ts_col = None
    if _cache_col and _cache_col in df_opal_overlay_base.columns:
        df_opal_overlay_base = df_opal_overlay_base.with_columns(
            pl.col(_cache_col).cast(pl.Float64).alias("opal__cache__score")
        )
        if cache_run_col and cache_run_col in df_opal_overlay_base.columns:
            df_opal_overlay_base = df_opal_overlay_base.with_columns(
                pl.col(cache_run_col).cast(pl.Utf8).alias("opal__cache__run_id")
            )
        if cache_round_col and cache_round_col in df_opal_overlay_base.columns:
            df_opal_overlay_base = df_opal_overlay_base.with_columns(
                pl.col(cache_round_col).cast(pl.Int64).alias("opal__cache__round")
            )
        if cache_ts_col and cache_ts_col in df_opal_overlay_base.columns:
            df_opal_overlay_base = df_opal_overlay_base.with_columns(
                pl.col(cache_ts_col).cast(pl.Utf8).alias("opal__cache__written_at")
            )

        def _first_non_null(col: str) -> str | None:
            try:
                series = df_opal_overlay_base.select(pl.col(col).drop_nulls().head(1)).to_series()
            except Exception:
                return None
            if series.len() == 0:
                return None
            return str(series[0])

        if "opal__cache__run_id" in df_opal_overlay_base.columns:
            cache_pred_meta["run_id"] = _first_non_null("opal__cache__run_id")
        if "opal__cache__round" in df_opal_overlay_base.columns:
            cache_pred_meta["as_of_round"] = _first_non_null("opal__cache__round")
        if "opal__cache__written_at" in df_opal_overlay_base.columns:
            cache_pred_meta["written_at"] = _first_non_null("opal__cache__written_at")
        _sel_params = dict(opal_campaign_info.selection_params or {}) if opal_campaign_info is not None else {}
        _id_col = "id" if "id" in df_opal_overlay_base.columns else "__row_id"
        df_opal_overlay_base, _warnings, _err = ensure_selection_columns(
            df_opal_overlay_base,
            id_col=_id_col,
            score_col="opal__cache__score",
            selection_params=_sel_params,
            rank_col="opal__cache__rank",
            top_k_col="opal__cache__top_k",
        )
        if _warnings:
            cache_warning = "; ".join(_warnings)
        if _err:
            cache_warning = _err if cache_warning is None else f"{cache_warning}; {_err}"

    cache_col = _cache_col
    return df_opal_overlay_base, cache_col, cache_warning, cache_pred_meta, ledger_warning


@app.cell
def _(label_source_value, mo, opal_round_dropdown, opal_round_value_map):
    opal_label_view_dropdown = None
    if label_source_value is not None and opal_round_dropdown is not None:
        view_options = [
            "all_events",
            "as_of_round (dedup latest per id)",
            "current_round_only",
        ]
        view_label = "Label view"
        view_default = "all_events"
        round_value = None
        selected_label = opal_round_dropdown.value
        round_value = opal_round_value_map.get(selected_label) if selected_label else None
        if round_value is None:
            view_options = ["all_events"]
            view_default = "all_events"
            view_label = "Label view (all rounds)"
        opal_label_view_dropdown = mo.ui.dropdown(
            options=view_options,
            value=view_default,
            label=view_label,
            full_width=True,
        )
    return (opal_label_view_dropdown,)


@app.cell
def _(
    LabelViewInputs,
    df_labels_filtered,
    label_source_value,
    opal_campaign_info,
    opal_label_view_dropdown,
    opal_pred_selected_run_id,
    opal_round_dropdown,
    opal_round_value_map,
):
    label_view_value = opal_label_view_dropdown.value if opal_label_view_dropdown is not None else None
    round_label = opal_round_dropdown.value if opal_round_dropdown is not None else None
    label_view_inputs = LabelViewInputs(
        df_labels_filtered=df_labels_filtered,
        label_source_value=label_source_value,
        label_view_value=label_view_value,
        round_label=round_label,
        round_value_map=opal_round_value_map,
        campaign_info=opal_campaign_info,
        pred_run_id=opal_pred_selected_run_id,
    )
    return (label_view_inputs,)


@app.cell
def _(
    dedup_latest_labels,
    label_view_inputs,
    mo,
    pl,
):
    _df_labels_filtered = label_view_inputs.df_labels_filtered
    _campaign_info = label_view_inputs.campaign_info
    _label_source_value = label_view_inputs.label_source_value
    _pred_run_id = label_view_inputs.pred_run_id
    opal_label_view_value = label_view_inputs.label_view_value
    _round_label = label_view_inputs.round_label
    _round_value_map = label_view_inputs.round_value_map
    opal_selected_round = None
    opal_round_notice_md = mo.md("")
    _selected_label = _round_label
    opal_selected_round = _round_value_map.get(_selected_label) if _selected_label else None

    opal_labels_view_df = _df_labels_filtered if _df_labels_filtered is not None else pl.DataFrame()
    opal_labels_current_df = opal_labels_view_df
    opal_labels_asof_df = opal_labels_view_df
    opal_labels_table_ui = mo.ui.table(opal_labels_view_df, page_size=5)

    if _campaign_info is None:
        pass
    elif _df_labels_filtered is None or _df_labels_filtered.is_empty():
        pass
    else:
        df_round_scope = _df_labels_filtered
        if opal_selected_round is not None and "observed_round" in _df_labels_filtered.columns:
            df_round_scope = _df_labels_filtered.filter(pl.col("observed_round") <= int(opal_selected_round))
            opal_labels_current_df = _df_labels_filtered.filter(pl.col("observed_round") == int(opal_selected_round))
        else:
            opal_labels_current_df = df_round_scope

        cumulative = bool(_campaign_info.training_policy.get("cumulative_training", True))
        opal_labels_asof_df = df_round_scope if cumulative else opal_labels_current_df
        if opal_selected_round is not None:
            policy = str(_campaign_info.training_policy.get("label_cross_round_deduplication_policy", "latest_only"))
            if policy == "latest_only":
                opal_labels_asof_df = dedup_latest_labels(
                    opal_labels_asof_df,
                    id_col="id",
                    round_col="observed_round",
                )

        view_choice_default = "all_events"
        view_choice = opal_label_view_value or view_choice_default
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

        _y_col_name = _campaign_info.y_column if _campaign_info is not None else "y_obs"
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
    _campaign_slug = _campaign_info.slug if _campaign_info is not None else None
    opal_labels_view_df = opal_labels_view_df.with_columns(
        pl.lit(_campaign_slug).alias("opal__label__campaign_slug"),
        pl.lit(opal_selected_round).alias("opal__label__as_of_round_view"),
        pl.lit(_pred_run_id).alias("opal__label__run_id"),
        pl.lit(_label_source_value).alias("opal__label__score_source"),
    )
    return (
        opal_labels_asof_df,
        opal_labels_current_df,
        opal_round_notice_md,
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
    base_options = [_name for _name in base_options if _name != "id_right"]
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
            elif _color_value in {"opal__overlay__top_k", "opal__score__top_k"}:
                _label_col = f"{_color_value}__label"
                df_sfxi_plot = df_sfxi_plot.with_columns(
                    pl.when(pl.col(_color_value)).then(pl.lit("Top-K")).otherwise(pl.lit("Not Top-K")).alias(_label_col)
                )
                _color_title = "Top-K"
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
    label_source_dropdown,
    label_source_notice_md,
    label_source_selection_notice_md,
    label_src_multiselect,
    label_rounds_from_ledger_checkbox,
    label_rounds_from_ledger_note_md,
    mismatch_md,
    mismatch_table,
    opal_label_view_dropdown,
    opal_labels_table_ui,
    opal_campaign_notice_md,
    opal_round_notice_md,
    opal_panel_prefix_dropdown,
    opal_pred_round_dropdown,
    opal_provenance_md,
    opal_round_dropdown,
    opal_run_id_dropdown,
    rf_model_source_md,
    score_source_dropdown,
    selection_csv_input,
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

    opal_controls = []
    if opal_panel_prefix_dropdown is not None:
        opal_controls.append(opal_panel_prefix_dropdown)

    opal_label_controls = [
        c
        for c in [
            label_source_dropdown,
            label_src_multiselect,
            label_rounds_from_ledger_checkbox,
            opal_round_dropdown,
            opal_label_view_dropdown,
        ]
        if c is not None
    ]
    opal_label_controls_row = mo.hstack(opal_label_controls) if opal_label_controls else mo.md("")
    opal_pred_controls = [
        c
        for c in [
            rf_model_source_md,
            opal_pred_round_dropdown,
            opal_run_id_dropdown,
            score_source_dropdown,
        ]
        if c is not None
    ]
    opal_pred_controls_row = mo.hstack(opal_pred_controls) if opal_pred_controls else mo.md("")

    pred_section = [
        mo.md("### Predictions (canonical inputs, overlay outputs)"),
        mo.md(
            "- **Ledger/cache**: canonical, immutable scores already persisted by OPAL runs.\n"
            "- **Overlay**: notebook-scoped predictions from the selected artifact model.\n"
            "- **Setpoint tuning** updates overlay scores only; canonical files are never modified."
        ),
        opal_pred_controls_row,
        selection_csv_input,
        mismatch_md,
        mismatch_table,
    ]

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
            opal_campaign_notice_md,
            mo.md("### Labels (observed events)"),
            mo.md(
                "These rows reflect experimental label events ingested for the selected campaign "
                "(e.g., via `opal ingest-y`). The **as-of round** view defines the training cutoff "
                "for active learning."
            ),
            opal_provenance_md,
            opal_label_controls_row,
            label_source_notice_md,
            label_source_selection_notice_md,
            label_rounds_from_ledger_note_md,
            opal_round_notice_md,
            opal_labels_table_ui,
            *pred_section,
        ]
    )
    sfxi_panel
    return


@app.cell(column=6)
def _(
    mo,
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
    setpoint_row_top = mo.hstack([sfxi_p00_slider, sfxi_p10_slider, sfxi_p01_slider, sfxi_p11_slider])
    exponent_row = mo.hstack([sfxi_beta_input, sfxi_gamma_input])
    sfxi_controls = mo.vstack([setpoint_row_top, exponent_row])
    sfxi_defs_md = mo.md(
        "- **logic_fidelity**: proximity to the setpoint logic vector.\n"
        "- **effect_scaled**: intensity normalized by the current scaling denom.\n"
        "- **score** = (logic_fidelity ** beta) * (effect_scaled ** gamma)"
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
    return


@app.cell(column=7)
def _(
    mo,
    overlay_cache_status_md,
    overlay_clear_cache_button,
    overlay_compute_button,
    rf_model_source_md,
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
        "**OPAL runs** produce round-scoped, versioned model artifacts (e.g., `model.joblib`) "
        "and write an append-only ledger for auditability and reproducibility "
        "(e.g., `outputs/round_<k>/...` and `outputs/ledger.*`).\n"
        "Use **Compute overlay** to score the full pool from the selected artifact; setpoint changes "
        "rescore cached predictions. Overlay scores are non-canonical."
    )

    feature_panel = (
        transient_feature_chart if transient_feature_chart is not None else mo.md("Feature importance unavailable.")
    )
    hist_panel = (
        transient_hist_chart if transient_hist_chart is not None else mo.md("Overlay score histogram unavailable.")
    )
    cluster_panel = (
        transient_cluster_chart
        if transient_cluster_chart is not None
        else mo.md("Leiden cluster score plot unavailable.")
    )
    rf_blocks = [
        mo.md(rf_header),
        rf_model_source_md,
        mo.hstack([overlay_compute_button, overlay_clear_cache_button]),
        overlay_cache_status_md,
        rf_model_source_note_md,
        transient_md if transient_md is not None else mo.md(""),
        mo.md(rf_note),
        feature_panel,
        mo.md(
            "Feature importance highlights which input dimensions the artifact model relies on most for its "
            "predictions."
        ),
        hist_panel,
        mo.md("The histogram shows the distribution of predicted SFXI scores across the full candidate pool."),
        transient_cluster_metric_dropdown,
        transient_cluster_hue_dropdown,
        cluster_panel,
        mo.md(
            "The cluster plot compares the selected metric across Leiden clusters, ordered numerically for stability."
        ),
    ]
    overlay_panel = mo.vstack(rf_blocks)
    overlay_panel
    return


@app.cell
def _(df_umap_overlay, pl):
    df_overlay_top_k_pool = df_umap_overlay.head(0)
    df_score_top_k_pool = df_umap_overlay.head(0)
    if "opal__overlay__top_k" in df_umap_overlay.columns:
        df_overlay_top_k_pool = df_umap_overlay.filter(pl.col("opal__overlay__top_k").fill_null(False))
    if "opal__score__top_k" in df_umap_overlay.columns:
        df_score_top_k_pool = df_umap_overlay.filter(pl.col("opal__score__top_k").fill_null(False))
    return df_overlay_top_k_pool, df_score_top_k_pool


@app.cell
def _(mo):
    inspect_pool_label_map = {
        "Full dataset (all rows)": "df_active",
        "UMAP brush selection": "df_umap_selected",
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
    df_sfxi_selected,
    df_score_top_k_pool,
    df_overlay_top_k_pool,
    df_umap_selected,
    inspect_pool_dropdown,
    inspect_pool_label_map,
):
    pool_choice = inspect_pool_label_map.get(inspect_pool_dropdown.value, inspect_pool_dropdown.value)
    if pool_choice == "df_umap_selected":
        df_pool = df_umap_selected
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


if __name__ == "__main__":
    app.run()
