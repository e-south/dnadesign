# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib==3.10.8",
#     "numpy==2.4.0",
#     "pandas==2.3.3",
#     "pyarrow==22.0.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    cache_dir = Path("/tmp/mpl-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import colors as mcolors

    DATASET_ROOT = Path("src/dnadesign/usr/datasets")
    DEFAULT_DATASET = "60bp_dual_promoter_cpxR_LexA"
    OPAL_PREFIX = "opal__prom60-etoh-cipro-andgate"
    SFXI_STATE_NAMES = ["p00", "p10", "p01", "p11"]
    return (
        DATASET_ROOT,
        DEFAULT_DATASET,
        OPAL_PREFIX,
        Path,
        SFXI_STATE_NAMES,
        cm,
        mcolors,
        mo,
        np,
        pd,
        plt,
        pq,
    )


@app.cell
def _(SFXI_STATE_NAMES, np, pd):
    def namespace_summary(columns: list[str]) -> pd.DataFrame:
        buckets: dict[str, list[str]] = {}
        for col in columns:
            ns = col.split("__", 1)[0] if "__" in col else "core"
            buckets.setdefault(ns, []).append(col)
        rows = []
        for ns, cols in sorted(buckets.items()):
            rows.append(
                {
                    "namespace": ns,
                    "count": len(cols),
                    "example_cols": ", ".join(cols[:6]),
                }
            )
        return pd.DataFrame(rows)

    def vec8_to_frame(series, *, prefix: str) -> pd.DataFrame:
        def _coerce(v):
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 8:
                return list(v[:8])
            return [np.nan] * 8

        logic_cols = [f"{prefix}_logic_{name}" for name in SFXI_STATE_NAMES]
        intensity_cols = [f"{prefix}_intensity_{name}" for name in SFXI_STATE_NAMES]
        cols = logic_cols + intensity_cols
        if len(series) == 0:
            return pd.DataFrame(columns=cols, index=series.index)
        values = np.vstack([_coerce(v) for v in series])
        return pd.DataFrame(values, columns=cols, index=series.index)

    def compute_sfxi_metrics(
        vecs: np.ndarray,
        *,
        setpoint: np.ndarray,
        beta: float,
        gamma: float,
        delta: float,
        percentile: float,
        eps: float,
    ) -> tuple[pd.DataFrame, float, np.ndarray, np.ndarray]:
        arr = np.asarray(vecs, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 8:
            raise ValueError("Expected vecs with shape (n, 8+).")
        valid = np.isfinite(arr[:, :8]).all(axis=1)
        arr = arr[valid]

        if arr.size == 0:
            empty = pd.DataFrame(columns=["logic_fidelity", "effect_raw", "effect_scaled", "score"])
            return empty, float(eps), np.zeros(4), valid

        v_hat = np.clip(arr[:, 0:4], 0.0, 1.0)
        y_star = arr[:, 4:8]

        sp = np.clip(np.asarray(setpoint, dtype=float).ravel(), 0.0, 1.0)
        sp_sum = float(sp.sum())
        weights = sp / sp_sum if sp_sum > eps else np.zeros_like(sp)

        worst = np.maximum(sp * sp, (1.0 - sp) * (1.0 - sp))
        denom_logic = float(np.sqrt(np.sum(worst)))
        if not np.isfinite(denom_logic) or denom_logic <= 0.0:
            logic = np.ones(v_hat.shape[0], dtype=float)
        else:
            dist = np.linalg.norm(v_hat - sp[None, :], axis=1)
            logic = np.clip(1.0 - (dist / denom_logic), 0.0, 1.0)

        y_lin = np.maximum(0.0, np.power(2.0, y_star) - float(delta))
        effect_raw = (y_lin * weights[None, :]).sum(axis=1)
        denom = float(np.nanpercentile(effect_raw, percentile)) if effect_raw.size else float("nan")
        if not np.isfinite(denom) or denom <= 0.0:
            denom = float(eps)
        effect_scaled = np.clip(effect_raw / denom, 0.0, 1.0)
        score = np.power(logic, beta) * np.power(effect_scaled, gamma)

        metrics = pd.DataFrame(
            {
                "logic_fidelity": logic,
                "effect_raw": effect_raw,
                "effect_scaled": effect_scaled,
                "score": score,
            }
        )
        return metrics, denom, weights, valid

    return namespace_summary, vec8_to_frame


@app.cell
def _(cm, mcolors, pd, plt):
    def plot_umap_scatter(
        df: pd.DataFrame,
        *,
        x_col: str,
        y_col: str,
        color_col: str | None,
        title: str,
        alpha: float,
        size: float,
    ):
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.scatter(df[x_col], df[y_col], s=size, alpha=0.2, color="#c7c7c7", linewidths=0)

        if color_col and color_col in df.columns:
            series = df[color_col]
            if pd.api.types.is_numeric_dtype(series):
                sc = ax.scatter(
                    df[x_col],
                    df[y_col],
                    c=series,
                    s=size,
                    alpha=alpha,
                    cmap="viridis",
                    linewidths=0,
                )
                fig.colorbar(sc, ax=ax, label=color_col)
            else:
                counts = series.astype("string").fillna("NA").value_counts()
                top = counts.head(12).index.tolist()
                mapped = series.astype("string").fillna("NA")
                mapped = mapped.where(mapped.isin(top), other="Other")
                categories = sorted(mapped.unique())
                palette = cm.get_cmap("tab20", len(categories))
                color_map = {cat: mcolors.to_hex(palette(i)) for i, cat in enumerate(categories)}
                for cat in categories:
                    mask = mapped == cat
                    ax.scatter(
                        df.loc[mask, x_col],
                        df.loc[mask, y_col],
                        s=size,
                        alpha=alpha,
                        color=color_map[cat],
                        label=cat,
                        linewidths=0,
                    )
                ax.legend(title=color_col, bbox_to_anchor=(1.02, 1), loc="upper left")

        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.15)
        return fig

    def plot_hist(series, *, title: str, bins: int = 40):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        clean = series.dropna()
        ax.hist(clean, bins=bins, color="#4c78a8", alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel(series.name or "value")
        ax.set_ylabel("count")
        ax.grid(alpha=0.2)
        return fig

    def plot_bar(series, *, title: str, max_items: int = 20):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        counts = series.value_counts().head(max_items)
        ax.bar(counts.index.astype(str), counts.values, color="#72b7b2")
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
        return fig

    return plot_bar, plot_hist, plot_umap_scatter


@app.cell
def _(DATASET_ROOT, DEFAULT_DATASET, mo):
    dataset_names = sorted([p.name for p in DATASET_ROOT.iterdir() if p.is_dir() and (p / "records.parquet").exists()])
    if not dataset_names:
        mo.stop(True, mo.md("No USR datasets found under `src/dnadesign/usr/datasets`."))

    default_value = DEFAULT_DATASET if DEFAULT_DATASET in dataset_names else dataset_names[0]
    dataset_picker = mo.ui.dropdown(
        dataset_names,
        value=default_value,
        label="USR dataset",
        full_width=True,
    )
    custom_path = mo.ui.text(
        value="",
        label="Custom records.parquet path (optional)",
        full_width=True,
    )
    use_custom_path = mo.ui.checkbox(label="Use custom path", value=False)
    include_heavy = mo.ui.checkbox(
        label="Include infer logits column (heavy)",
        value=False,
    )

    mo.vstack(
        [
            mo.md("### Dataset selection"),
            dataset_picker,
            custom_path,
            mo.hstack([use_custom_path, include_heavy]),
        ]
    )
    return custom_path, dataset_picker, include_heavy, use_custom_path


@app.cell
def _(mo):
    id_search = mo.ui.text(
        value="",
        label="ID substring filter (case-insensitive)",
        full_width=True,
    )
    only_labeled = mo.ui.checkbox(
        label="Only labeled (sfxi_8_vector_y_label)",
        value=False,
    )
    only_densegen = mo.ui.checkbox(
        label="Only DenseGen records (densegen__plan present)",
        value=False,
    )
    opal_round_filter = mo.ui.number(
        value=-1,
        label="Only OPAL latest_as_of_round (use -1 for all)",
    )
    table_limit = mo.ui.slider(
        start=50,
        stop=1000,
        value=200,
        step=50,
        label="Curated table rows",
        full_width=True,
    )
    show_explorer = mo.ui.checkbox(label="Show data explorer", value=False)

    mo.vstack(
        [
            mo.md("### Global filters"),
            id_search,
            mo.hstack([only_labeled, only_densegen]),
            opal_round_filter,
            mo.hstack([table_limit, show_explorer]),
        ]
    )
    return (
        id_search,
        only_densegen,
        only_labeled,
        opal_round_filter,
        show_explorer,
        table_limit,
    )


@app.cell
def _(
    DATASET_ROOT,
    OPAL_PREFIX,
    Path,
    custom_path,
    dataset_picker,
    include_heavy,
    mo,
    namespace_summary,
    pd,
    pq,
    use_custom_path,
):
    if use_custom_path.value and custom_path.value.strip():
        dataset_path = Path(custom_path.value).expanduser()
    else:
        dataset_path = DATASET_ROOT / dataset_picker.value / "records.parquet"
    dataset_path = dataset_path.resolve()
    mo.stop(not dataset_path.exists(), mo.md(f"Missing dataset: `{dataset_path}`"))

    parquet = pq.ParquetFile(dataset_path)
    schema = parquet.schema_arrow
    column_names = schema.names

    namespace_df = namespace_summary(column_names)
    schema_df = pd.DataFrame({"column": schema.names, "type": [str(t) for t in schema.types]})

    opal_cols = [c for c in column_names if c.startswith(OPAL_PREFIX)]
    infer_cols = [c for c in column_names if c.startswith("infer__")]
    heavy_cols = [c for c in infer_cols if "logits_mean" in c]
    selected_cols = [
        "id",
        "sequence",
        "bio_type",
        "alphabet",
        "length",
        "source",
        "created_at",
        "densegen__plan",
        "densegen__compression_ratio",
        "densegen__gap_fill_gc_actual",
        "densegen__gap_fill_used",
        "densegen__gap_fill_bases",
        "densegen__library_size",
        "densegen__promoter_constraint",
        "densegen__covers_all_tfs_in_solution",
        "densegen__used_tf_counts",
        "densegen__used_tfbs_detail",
        "densegen__visual",
        "cluster__ldn_v1",
        "cluster__ldn_v1__umap_x",
        "cluster__ldn_v1__umap_y",
        "sfxi_8_vector_y_label",
    ]
    selected_cols.extend([c for c in infer_cols if c.endswith("__ll_mean")])
    selected_cols.extend(opal_cols)
    if include_heavy.value:
        selected_cols.extend(heavy_cols)

    present_cols = [c for c in selected_cols if c in column_names]
    missing_cols = sorted(set(selected_cols) - set(present_cols))
    table = pq.read_table(dataset_path, columns=present_cols)
    df = table.to_pandas()
    return column_names, df, missing_cols, namespace_df, schema_df


@app.cell
def _(df, pd):
    missing_df = pd.DataFrame(
        {
            "column": df.columns,
            "null_pct": df.isna().mean() * 100.0,
            "non_null": df.notna().sum(),
        }
    ).sort_values("null_pct", ascending=False)
    return (missing_df,)


@app.cell
def _(column_names, df, missing_cols, missing_df, mo, namespace_df, schema_df):
    summary_md = mo.md(
        f"""
    ### Dataset summary
    - Rows: **{len(df):,}**
    - Columns loaded: **{len(df.columns)}** / {len(column_names)}
    - Missing requested cols: **{len(missing_cols)}**
    """
    )
    missing_md = mo.md(
        "Missing columns (requested but not present): " + (", ".join(missing_cols) if missing_cols else "None")
    )
    overview_panel = mo.vstack(
        [
            summary_md,
            missing_md,
            mo.md("#### Columns by namespace"),
            namespace_df,
            mo.md("#### Missingness (loaded columns)"),
            missing_df.head(40),
            mo.md("#### Schema (loaded columns)"),
            schema_df.head(40),
        ]
    )
    return (overview_panel,)


@app.cell
def _(df, id_search, only_densegen, only_labeled, opal_round_filter):
    df_filtered = df
    if id_search.value.strip():
        df_filtered = df_filtered[df_filtered["id"].str.contains(id_search.value.strip(), case=False, na=False)]
    if only_labeled.value and "sfxi_8_vector_y_label" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["sfxi_8_vector_y_label"].notna()]
    if only_densegen.value and "densegen__plan" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["densegen__plan"].notna()]

    round_val = opal_round_filter.value
    if round_val is not None and float(round_val) >= 0:
        opal_round_cols = [c for c in df_filtered.columns if c.endswith("__latest_as_of_round")]
        if opal_round_cols:
            df_filtered = df_filtered[df_filtered[opal_round_cols[0]] == int(round_val)]

    df_filtered = df_filtered.reset_index(drop=True)
    return (df_filtered,)


@app.cell
def _(df_filtered, pd, table_limit, vec8_to_frame):
    curated_cols = [
        col
        for col in [
            "id",
            "sequence",
            "densegen__plan",
            "densegen__compression_ratio",
            "densegen__gap_fill_gc_actual",
            "densegen__gap_fill_used",
            "densegen__library_size",
            "cluster__ldn_v1",
            "sfxi_8_vector_y_label",
        ]
        if col in df_filtered.columns
    ]
    curated = df_filtered[curated_cols].copy()
    table_df = curated.head(int(table_limit.value)).copy()

    if "densegen__used_tf_counts" in df_filtered.columns:
        tf_counts_curated = pd.json_normalize(df_filtered["densegen__used_tf_counts"])
        tf_counts_curated.columns = [f"densegen__tf_count_{c}" for c in tf_counts_curated.columns]
        curated = pd.concat([curated, tf_counts_curated], axis=1)

    if "sfxi_8_vector_y_label" in table_df.columns:
        sfxi_split = vec8_to_frame(table_df["sfxi_8_vector_y_label"], prefix="sfxi")
        table_df = pd.concat([table_df.drop(columns=["sfxi_8_vector_y_label"]), sfxi_split], axis=1)
    return (table_df,)


@app.cell
def _(mo, table_df):
    curated_table = mo.ui.table(
        table_df,
        sortable=True,
        filterable=True,
    )
    curated_table
    return


@app.cell
def _(df_filtered, mo, show_explorer):
    if show_explorer.value:
        explorer = mo.ui.data_explorer(df_filtered)
        explorer
    else:
        mo.md("Data explorer disabled.")
    return


@app.cell
def _(df_filtered, mo):
    active_id_override = mo.ui.text(
        value="",
        label="Active record ID (override)",
        full_width=True,
    )
    max_choices = min(len(df_filtered), 500)
    id_options = df_filtered["id"].head(max_choices).tolist() if max_choices else []
    active_id_picker = mo.ui.dropdown(
        options=id_options,
        value=id_options[0] if id_options else None,
        label="Active record (first 500 filtered rows)",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("### Active record"),
            mo.md("Use the override field or dropdown to change the active record."),
            active_id_override,
            active_id_picker,
        ]
    )
    return active_id_override, active_id_picker


@app.cell(hide_code=True)
def _(active_id_override, active_id_picker, df_filtered, mo):
    if active_id_override.value.strip():
        active_id = active_id_override.value.strip()
    else:
        active_id = active_id_picker.value

    mo.stop(len(df_filtered) == 0, mo.md("No rows after filtering."))
    active_row = df_filtered[df_filtered["id"] == active_id]
    if active_row.empty:
        active_row = df_filtered.head(1)
        active_id = active_row["id"].iloc[0]
    active_row = active_row.iloc[0]
    return active_id, active_row


@app.cell(hide_code=True)
def _(active_id, active_row, mo):
    sequence = active_row.get("sequence", "")
    densegen_visual = active_row.get("densegen__visual", "")
    record_summary = {
        "id": active_id,
        "length": active_row.get("length"),
        "densegen__plan": active_row.get("densegen__plan"),
        "cluster__ldn_v1": active_row.get("cluster__ldn_v1"),
    }
    mo.vstack(
        [
            mo.md("#### Active record summary"),
            record_summary,
            mo.md("#### Sequence"),
            mo.md(f"```\n{sequence}\n```"),
            mo.md("#### DenseGen visual"),
            mo.md(f"```\n{densegen_visual}\n```"),
        ]
    )
    return


@app.cell
def _(active_row, pd):
    tfbs_detail = active_row.get("densegen__used_tfbs_detail", None)
    if isinstance(tfbs_detail, list):
        detail_df = pd.DataFrame(tfbs_detail)
    else:
        detail_df = pd.DataFrame(columns=["tf", "tfbs", "offset", "orientation"])
    detail_df
    return


@app.cell
def _(mo):
    render_baserender = mo.ui.checkbox(label="Render BaseRender preview", value=False)
    highlight_tf = mo.ui.dropdown(
        options=["(all)", "LexA", "CpxR"],
        value="(all)",
        label="Highlight TF",
    )
    mo.hstack([render_baserender, highlight_tf])
    return


app._unparsable_cell(
    r"""
    if not render_baserender.value:
        mo.md(\"BaseRender preview disabled.\")
        return

    try:
        from dnadesign.baserender.src.api import render_image
        from dnadesign.baserender.src.io.parquet import read_parquet_records_by_ids
        from dnadesign.baserender.src.model import SeqRecord

        records = list(read_parquet_records_by_ids(dataset_path, ids=[active_id]))
        if not records:
            mo.md(\"No BaseRender record found for active id.\")
            return
        record = records[0]
        if highlight_tf.value != \"(all)\":
            tag = f\"tf:{highlight_tf.value}\"
            filtered = [a for a in record.annotations if a.tag == tag]
            record = SeqRecord(
                id=record.id,
                alphabet=record.alphabet,
                sequence=record.sequence,
                annotations=filtered,
                guides=record.guides,
            ).validate()
        baserender_fig = render_image(record, fmt=\"png\")
        baserender_fig
    except Exception as exc:
        mo.md(f\"BaseRender failed: `{exc}`\")
    """,
    name="_",
)


@app.cell
def _(df_filtered, mo, pd):
    umap_x_col = "cluster__ldn_v1__umap_x"
    umap_y_col = "cluster__ldn_v1__umap_y"
    mo.stop(
        umap_x_col not in df_filtered.columns or umap_y_col not in df_filtered.columns,
        mo.md("UMAP columns not found in filtered dataset."),
    )

    numeric_cols = [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])]
    color_options = ["cluster__ldn_v1"] + numeric_cols
    default_color = (
        "cluster__ldn_v1" if "cluster__ldn_v1" in color_options else (numeric_cols[0] if numeric_cols else None)
    )
    mo.stop(default_color is None, mo.md("No columns available for UMAP coloring."))
    color_by = mo.ui.dropdown(
        options=color_options,
        value=default_color,
        label="Color by",
    )
    alpha = mo.ui.slider(0.1, 1.0, value=0.6, step=0.05, label="Point alpha")
    size = mo.ui.slider(5, 60, value=18, step=1, label="Point size")

    x_min, x_max = df_filtered[umap_x_col].min(), df_filtered[umap_x_col].max()
    y_min, y_max = df_filtered[umap_y_col].min(), df_filtered[umap_y_col].max()
    x_range = mo.ui.range_slider(x_min, x_max, value=(x_min, x_max), label="UMAP X range")
    y_range = mo.ui.range_slider(y_min, y_max, value=(y_min, y_max), label="UMAP Y range")

    numeric_filter_col = mo.ui.dropdown(
        options=["(none)"] + numeric_cols,
        value="(none)",
        label="Numeric filter column",
    )

    mo.vstack([color_by, mo.hstack([alpha, size]), x_range, y_range, numeric_filter_col])
    return (
        alpha,
        color_by,
        numeric_filter_col,
        size,
        umap_x_col,
        umap_y_col,
        x_range,
        y_range,
    )


@app.cell
def _(df_filtered, mo, numeric_filter_col):
    if numeric_filter_col.value == "(none)" or numeric_filter_col.value not in df_filtered.columns:
        filter_slider = mo.ui.range_slider(0.0, 1.0, value=(0.0, 1.0), label="Filter range")
        filter_active = False
    else:
        series = df_filtered[numeric_filter_col.value].dropna()
        if series.empty:
            filter_slider = mo.ui.range_slider(0.0, 1.0, value=(0.0, 1.0), label="Filter range")
            filter_active = False
        else:
            lo, hi = float(series.min()), float(series.max())
            filter_slider = mo.ui.range_slider(lo, hi, value=(lo, hi), label=f"{numeric_filter_col.value} range")
            filter_active = True
    filter_slider
    return filter_active, filter_slider


@app.cell
def _(
    alpha,
    color_by,
    df_filtered,
    filter_active,
    filter_slider,
    mo,
    plot_bar,
    plot_umap_scatter,
    size,
    umap_x_col,
    umap_y_col,
    x_range,
    y_range,
):
    df_umap = df_filtered.dropna(subset=[umap_x_col, umap_y_col]).copy()
    df_umap = df_umap[
        (df_umap[umap_x_col] >= x_range.value[0])
        & (df_umap[umap_x_col] <= x_range.value[1])
        & (df_umap[umap_y_col] >= y_range.value[0])
        & (df_umap[umap_y_col] <= y_range.value[1])
    ]
    if filter_active:
        col = filter_slider.label.split(" range")[0]
        df_umap = df_umap[(df_umap[col] >= filter_slider.value[0]) & (df_umap[col] <= filter_slider.value[1])]

    umap_fig = plot_umap_scatter(
        df_umap,
        x_col=umap_x_col,
        y_col=umap_y_col,
        color_col=color_by.value,
        title="UMAP region (filtered)",
        alpha=float(alpha.value),
        size=float(size.value),
    )

    plan_plot = (
        plot_bar(df_umap["densegen__plan"], title="DenseGen plan counts")
        if "densegen__plan" in df_umap.columns
        else None
    )
    cluster_plot = (
        plot_bar(df_umap["cluster__ldn_v1"], title="Cluster counts") if "cluster__ldn_v1" in df_umap.columns else None
    )

    umap_summary = df_umap.select_dtypes(include=["number"]).describe().T
    umap_panel = mo.vstack(
        [
            mo.md(f"Rows in region: **{len(df_umap):,}**"),
            umap_fig,
            mo.hstack([plan_plot, cluster_plot]) if plan_plot and cluster_plot else mo.md(""),
            mo.md("#### Numeric summary (region)"),
            umap_summary,
        ]
    )
    return (umap_panel,)


@app.cell
def _(df_filtered, mo):
    plan_options = (
        ["(all)"] + sorted(df_filtered["densegen__plan"].dropna().unique().tolist())
        if "densegen__plan" in df_filtered.columns
        else ["(all)"]
    )
    plan_filter = mo.ui.dropdown(plan_options, value=plan_options[0], label="Plan")
    gap_filter = mo.ui.dropdown(
        options=["(all)", "gap_fill_used", "gap_fill_not_used"],
        value="(all)",
        label="Gap fill",
    )
    cover_filter = mo.ui.dropdown(
        options=["(all)", "covers_all_tfs", "missing_tfs"],
        value="(all)",
        label="Coverage",
    )
    mo.hstack([plan_filter, gap_filter, cover_filter])
    return cover_filter, gap_filter, plan_filter


@app.cell
def _(cover_filter, df_filtered, gap_filter, mo, plan_filter, plot_hist):
    df_dense = df_filtered.copy()
    if plan_filter.value != "(all)" and "densegen__plan" in df_dense.columns:
        df_dense = df_dense[df_dense["densegen__plan"] == plan_filter.value]
    if gap_filter.value == "gap_fill_used" and "densegen__gap_fill_used" in df_dense.columns:
        df_dense = df_dense[df_dense["densegen__gap_fill_used"].fillna(False)]
    if gap_filter.value == "gap_fill_not_used" and "densegen__gap_fill_used" in df_dense.columns:
        df_dense = df_dense[~df_dense["densegen__gap_fill_used"].fillna(False)]
    if cover_filter.value == "covers_all_tfs" and "densegen__covers_all_tfs_in_solution" in df_dense.columns:
        df_dense = df_dense[df_dense["densegen__covers_all_tfs_in_solution"].fillna(False)]
    if cover_filter.value == "missing_tfs" and "densegen__covers_all_tfs_in_solution" in df_dense.columns:
        df_dense = df_dense[~df_dense["densegen__covers_all_tfs_in_solution"].fillna(False)]

    compression_plot = (
        plot_hist(
            df_dense["densegen__compression_ratio"],
            title="Compression ratio",
        )
        if "densegen__compression_ratio" in df_dense.columns
        else None
    )
    gc_plot = (
        plot_hist(
            df_dense["densegen__gap_fill_gc_actual"],
            title="Gap-fill GC",
        )
        if "densegen__gap_fill_gc_actual" in df_dense.columns
        else None
    )

    densegen_panel = mo.vstack(
        [
            mo.md(f"Rows in DenseGen view: **{len(df_dense):,}**"),
            mo.hstack([compression_plot, gc_plot]) if compression_plot and gc_plot else mo.md(""),
        ]
    )
    return densegen_panel, df_dense


@app.cell
def _(df_dense, pd):
    if "densegen__used_tf_counts" in df_dense.columns:
        tf_counts_dense = pd.json_normalize(df_dense["densegen__used_tf_counts"]).fillna(0)
        tf_totals = tf_counts_dense.sum().sort_values(ascending=False)
        tf_usage = pd.DataFrame({"tf": tf_totals.index, "count": tf_totals.values})
    else:
        tf_usage = pd.DataFrame(columns=["tf", "count"])
    tf_usage
    return


@app.cell
def _(OPAL_PREFIX, df_filtered, mo, plot_hist):
    pred_col = f"{OPAL_PREFIX}__latest_pred_scalar"
    opal_latest_round_col = f"{OPAL_PREFIX}__latest_as_of_round"
    label_hist_col = f"{OPAL_PREFIX}__label_hist"

    pred_plot = (
        plot_hist(df_filtered[pred_col], title="OPAL latest pred scalar")
        if pred_col in df_filtered.columns
        else mo.md("No OPAL pred scalar column.")
    )

    mo.vstack(
        [
            mo.md("### OPAL campaign view"),
            pred_plot,
            mo.md(f"Columns: `{pred_col}`, `{opal_latest_round_col}`, `{label_hist_col}`"),
        ]
    )
    return label_hist_col, pred_col


@app.cell
def _(df_filtered, mo, pred_col):
    if pred_col in df_filtered.columns:
        top_n = mo.ui.slider(10, 200, value=50, step=10, label="Top-N by pred scalar")
        top_table = df_filtered.sort_values(pred_col, ascending=False).head(int(top_n.value)).loc[:, ["id", pred_col]]
        mo.vstack([top_n, top_table])
    else:
        mo.md("OPAL pred scalar column not available.")
    return


@app.cell
def _(active_row, label_hist_col, mo, pd, vec8_to_frame):
    raw = active_row.get(label_hist_col, None)
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    if isinstance(raw, list) and raw:
        hist_df = pd.DataFrame(raw)
        if "y" in hist_df.columns:
            y_split = vec8_to_frame(hist_df["y"], prefix="label")
            hist_df = pd.concat([hist_df.drop(columns=["y"]), y_split], axis=1)
        hist_df
    else:
        mo.md("No OPAL label history for active record.")
    return


@app.cell
def _(mo):
    p00 = mo.ui.slider(0.0, 1.0, value=0.0, step=0.01, label="p00")
    p10 = mo.ui.slider(0.0, 1.0, value=0.0, step=0.01, label="p10")
    p01 = mo.ui.slider(0.0, 1.0, value=0.0, step=0.01, label="p01")
    p11 = mo.ui.slider(0.0, 1.0, value=1.0, step=0.01, label="p11")
    beta = mo.ui.number(value=1.0, label="beta (logic exponent)")
    gamma = mo.ui.number(value=1.0, label="gamma (effect exponent)")
    delta = mo.ui.number(value=0.0, label="delta (log2 offset)")
    percentile = mo.ui.slider(50, 99, value=95, step=1, label="denom percentile")
    eps = mo.ui.number(value=1e-8, label="epsilon")

    controls = mo.vstack(
        [
            mo.md("### SFXI setpoint & scaling"),
            mo.hstack([p00, p10, p01, p11]),
            mo.hstack([beta, gamma]),
            mo.hstack([delta, percentile, eps]),
        ]
    )
    controls
    return


app._unparsable_cell(
    r"""
    if \"sfxi_8_vector_y_label\" not in df_filtered.columns:
        mo.md(\"No SFXI labels in dataset.\")
        return

    df_sfxi = df_filtered[df_filtered[\"sfxi_8_vector_y_label\"].notna()].copy()
    if df_sfxi.empty:
        mo.md(\"No labeled rows for SFXI scoring.\")
        return

    raw_vecs = []
    for v in df_sfxi[\"sfxi_8_vector_y_label\"].tolist():
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 8:
            raw_vecs.append(list(v[:8]))
        else:
            raw_vecs.append([np.nan] * 8)
    vecs = np.vstack(raw_vecs)

    setpoint = np.array([p00.value, p10.value, p01.value, p11.value], dtype=float)
    metrics, denom, weights, valid = compute_sfxi_metrics(
        vecs,
        setpoint=setpoint,
        beta=float(beta.value),
        gamma=float(gamma.value),
        delta=float(delta.value),
        percentile=float(percentile.value),
        eps=float(eps.value),
    )
    df_sfxi = df_sfxi.loc[valid].reset_index(drop=True)
    df_sfxi = pd.concat(
        [df_sfxi.reset_index(drop=True), metrics.reset_index(drop=True)],
        axis=1,
    )

    mo.md(
        f\"Computed SFXI metrics on **{len(df_sfxi):,}** labeled rows. \"
        f\"Denom={denom:.4g}, weights={weights.round(3).tolist()}\"
    )
    """,
    name="_",
)


app._unparsable_cell(
    r"""
    if df_sfxi is None or df_sfxi.empty:
        mo.md(\"No SFXI-scored rows to plot.\")
        return

    sfxi_fig = plot_umap_scatter(
        df_sfxi,
        x_col=\"logic_fidelity\",
        y_col=\"effect_scaled\",
        color_col=\"score\",
        title=\"SFXI: logic fidelity vs scaled effect\",
        alpha=0.7,
        size=30,
    )
    top = (
        df_sfxi.sort_values(\"score\", ascending=False)
        .head(50)
        .loc[:, [\"id\", \"score\", \"logic_fidelity\", \"effect_scaled\"]]
    )
    mo.vstack([sfxi_fig, mo.md(\"Top-50 by score\"), top])
    """,
    name="_",
)


app._unparsable_cell(
    r"""
    if df_sfxi is None or df_sfxi.empty:
        return
    row = df_sfxi[df_sfxi[\"id\"] == active_id]
    if row.empty:
        mo.md(\"Active record not labeled for SFXI.\")
        return
    row = row.iloc[0]
    mo.md(
        f\"\"\"
    **Active SFXI score**
    - logic_fidelity: {row['logic_fidelity']:.3f}
    - effect_scaled: {row['effect_scaled']:.3f}
    - score: {row['score']:.3f}
    \"\"\"
    )
    """,
    name="_",
)


@app.cell
def _(mo):
    export_scope = mo.ui.dropdown(
        options=["filtered", "umap_region", "sfxi_top", "active_record"],
        value="filtered",
        label="Export scope",
    )
    export_format = mo.ui.dropdown(
        options=["csv", "parquet"],
        value="csv",
        label="Export format",
    )
    export_name = mo.ui.text(
        value="promoter_eda_export.csv",
        label="Output filename",
        full_width=True,
    )
    export_button = mo.ui.run_button(label="Export selection")
    mo.vstack([export_scope, export_format, export_name, export_button])
    return


app._unparsable_cell(
    r"""
    if not export_button.value:
        mo.md(\"Export not triggered.\")
        return

    if export_scope.value == \"active_record\":
        export_df = active_row.to_frame().T
    elif export_scope.value == \"sfxi_top\" and df_sfxi is not None:
        export_df = df_sfxi.sort_values(\"score\", ascending=False).head(200)
    elif export_scope.value == \"umap_region\":
        export_df = df_umap if isinstance(df_umap, pd.DataFrame) else df_filtered
    else:
        export_df = df_filtered

    out_dir = Path(\"src/dnadesign/opal/notebooks/_outputs\")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / export_name.value

    if export_format.value == \"csv\":
        export_df.to_csv(out_path, index=False)
    else:
        export_df.to_parquet(out_path, index=False)

    mo.md(f\"Exported `{export_scope.value}` to `{out_path}`\")
    """,
    name="_",
)


@app.cell
def _(densegen_panel, mo, overview_panel, umap_panel):
    app_tabs = mo.ui.tabs(
        {
            "Overview": overview_panel,
            "UMAP": umap_panel,
            "DenseGen": densegen_panel,
        }
    )
    app_tabs
    return


if __name__ == "__main__":
    app.run()
