import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl

    alt.data_transformers.disable_max_rows()

    def _dataset_root() -> Path:
        # notebooks/ -> usr/ -> datasets/
        return Path(__file__).resolve().parents[2] / "datasets"

    return Path, alt, mo, np, pd, pl, _dataset_root


@app.cell
def _(mo):
    mo.md(
        """
# USR dataset explorer

Interactive views for USR datasets: quick summaries, filters, and plots.
"""
    )
    return


@app.cell
def _(_dataset_root, mo):
    datasets_root = _dataset_root()
    dataset_names = sorted([p.name for p in datasets_root.iterdir() if p.is_dir() and (p / "records.parquet").exists()])

    dataset_picker = mo.ui.dropdown(
        options=dataset_names,
        value=(dataset_names[0] if dataset_names else None),
        label="Dataset",
    )
    custom_path = mo.ui.text(value="", label="Custom parquet path (optional)", full_width=True)
    sample_n = mo.ui.slider(0, 5000, value=2000, step=100, label="Sample rows (0 = all)")
    seed = mo.ui.number(value=7, label="Sample seed")

    controls = mo.vstack(
        [
            dataset_picker,
            custom_path,
            mo.hstack([sample_n, seed]),
        ]
    )
    controls
    return controls, custom_path, dataset_names, dataset_picker, datasets_root, sample_n, seed


@app.cell
def _(Path, custom_path, dataset_picker, datasets_root, mo):
    custom = custom_path.value.strip()
    if custom:
        records_path = Path(custom).expanduser()
    else:
        if dataset_picker.value is None:
            mo.stop(True, mo.md("No datasets found under `usr/datasets`."))
        records_path = datasets_root / dataset_picker.value / "records.parquet"

    if not records_path.exists():
        mo.stop(True, mo.md(f"records.parquet not found: `{records_path}`"))

    dataset_label = dataset_picker.value or records_path.name
    return dataset_label, records_path


@app.cell
def _(dataset_label, mo, pl, records_path, sample_n, seed):
    df_full = pl.read_parquet(records_path)
    rows_total = df_full.height
    cols_total = len(df_full.columns)

    if sample_n.value and rows_total > sample_n.value:
        df_sample = df_full.sample(n=sample_n.value, seed=int(seed.value))
    else:
        df_sample = df_full

    has_sequence = "sequence" in df_sample.columns
    has_length = "length" in df_sample.columns
    if not has_sequence and not has_length:
        mo.stop(True, mo.md("This notebook requires a `sequence` or `length` column."))

    if has_sequence:
        seq = pl.col("sequence").cast(pl.Utf8)
        seq_len = seq.str.len_chars()
        if not has_length:
            df_sample = df_sample.with_columns(seq_len.alias("length"))
        gc = seq.str.count_matches(r"[GgCc]")
        gc_frac = pl.when(seq_len > 0).then(gc / seq_len).otherwise(None).alias("gc_content")
        df_sample = df_sample.with_columns(gc_frac)

    df_preview = df_sample.head(200).to_pandas()

    summary = mo.md(
        f"**Dataset:** {dataset_label}  \n"
        f"**Rows:** {rows_total:,} (sampled: {df_sample.height:,})  \n"
        f"**Columns:** {cols_total}"
    )
    summary
    return cols_total, df_preview, df_sample, rows_total, summary


@app.cell
def _(df_preview, mo):
    mo.ui.dataframe(df_preview)
    return


@app.cell
def _(df_sample, mo, pl):
    numeric_cols = [name for name, dtype in zip(df_sample.columns, df_sample.dtypes) if pl.datatypes.is_numeric(dtype)]
    numeric_picker = mo.ui.dropdown(
        options=numeric_cols,
        value=(numeric_cols[0] if numeric_cols else None),
        label="Numeric column for histogram",
    )
    numeric_picker
    return numeric_cols, numeric_picker


@app.cell
def _(alt, df_sample, mo, numeric_picker, pd):
    df_plot = df_sample.to_pandas()
    charts = {}

    if "length" in df_plot.columns:
        length_chart = (
            alt.Chart(df_plot)
            .mark_bar()
            .encode(
                x=alt.X("length:Q", bin=alt.Bin(maxbins=40), title="Length (nt)"),
                y=alt.Y("count()", title="Count"),
            )
            .properties(title="Length distribution", height=240, width=360)
        )
        charts["Length"] = mo.ui.altair_chart(length_chart)

    if "gc_content" in df_plot.columns:
        gc_chart = (
            alt.Chart(df_plot)
            .mark_bar()
            .encode(
                x=alt.X("gc_content:Q", bin=alt.Bin(maxbins=40), title="GC fraction"),
                y=alt.Y("count()", title="Count"),
            )
            .properties(title="GC distribution", height=240, width=360)
        )
        charts["GC"] = mo.ui.altair_chart(gc_chart)

    if {"length", "gc_content"}.issubset(df_plot.columns):
        scatter = (
            alt.Chart(df_plot)
            .mark_circle(size=30, opacity=0.5)
            .encode(
                x=alt.X("length:Q", title="Length (nt)"),
                y=alt.Y("gc_content:Q", title="GC fraction"),
                tooltip=["length", "gc_content"],
            )
            .properties(title="GC vs length", height=240, width=360)
        )
        charts["GC vs length"] = mo.ui.altair_chart(scatter)

    if numeric_picker.value and numeric_picker.value in df_plot.columns:
        col = numeric_picker.value
        num_chart = (
            alt.Chart(df_plot)
            .mark_bar()
            .encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40), title=col),
                y=alt.Y("count()", title="Count"),
            )
            .properties(title=f"{col} distribution", height=240, width=360)
        )
        charts["Numeric"] = mo.ui.altair_chart(num_chart)

    null_pct = (
        pd.DataFrame({"column": df_plot.columns, "null_pct": df_plot.isna().mean() * 100.0})
        .sort_values("null_pct", ascending=False)
        .head(30)
    )
    null_chart = (
        alt.Chart(null_pct)
        .mark_bar()
        .encode(
            x=alt.X("column:N", sort="-y", title="Column"),
            y=alt.Y("null_pct:Q", title="Nulls (%)"),
            tooltip=["column", alt.Tooltip("null_pct:Q", format=".1f")],
        )
        .properties(title="Null percentage (top 30)", height=240, width=520)
    )
    charts["Nulls"] = mo.ui.altair_chart(null_chart)

    if charts:
        mo.ui.tabs(charts)
    else:
        mo.md("No plots available for this dataset.")
    return


if __name__ == "__main__":
    app.run()
