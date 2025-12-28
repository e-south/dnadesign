import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import math
    import os
    from dataclasses import dataclass
    from pathlib import Path
    from textwrap import dedent
    from typing import Sequence

    import altair as alt
    import marimo as mo
    import polars as pl

    alt.data_transformers.disable_max_rows()
    return Path, Sequence, alt, dataclass, dedent, math, mo, os, pl


@app.cell(hide_code=True)
def _(Path, Sequence, dataclass, math, pl):
    @dataclass(frozen=True)
    class NumericRule:
        enabled: bool
        column: str | None
        op: str
        value: float | None = None

    SUPPORTED_NUMERIC_OPS = {">=", "<=", "is null", "is not null"}

    def find_repo_root(start: Path) -> Path | None:
        start = Path(start).resolve()
        if start.is_file():
            start = start.parent
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
        return None

    def resolve_usr_root(repo_root: Path | None, env_override: str | None) -> Path:
        if env_override:
            override_path = Path(env_override).expanduser().resolve()
            if not override_path.exists():
                raise ValueError(f"DNADESIGN_USR_ROOT does not exist: {override_path}")
            return override_path
        if repo_root is None:
            raise ValueError("Could not find repo root (pyproject.toml). Provide an absolute path.")
        return repo_root / "src" / "dnadesign" / "usr" / "datasets"

    def list_usr_datasets(usr_root: Path) -> list[str]:
        if not usr_root.exists():
            return []
        datasets: list[str] = []
        for child in usr_root.iterdir():
            if not child.is_dir():
                continue
            if (child / "records.parquet").is_file():
                datasets.append(child.name)
        return sorted(datasets)

    def resolve_dataset_path(
        *,
        repo_root: Path | None,
        usr_root: Path | None,
        dataset_name: str | None,
        custom_path: str | None,
    ) -> tuple[Path, str]:
        custom = (custom_path or "").strip()
        if custom:
            custom_path_obj = Path(custom).expanduser()
            if custom_path_obj.is_absolute():
                return custom_path_obj, "custom"
            if repo_root is None:
                raise ValueError("Relative custom paths require a repo root.")
            return (repo_root / custom_path_obj).resolve(), "custom"
        if usr_root is None:
            raise ValueError("USR root is unavailable; provide a custom path.")
        if not dataset_name:
            raise ValueError("Select a dataset or provide a custom path.")
        return (usr_root / dataset_name / "records.parquet").resolve(), "usr"

    def namespace_summary(columns: Sequence[str], max_examples: int = 3) -> pl.DataFrame:
        buckets: dict[str, list[str]] = {}
        for name in columns:
            if "__" in name:
                namespace = name.split("__", 1)[0]
            else:
                namespace = "core"
            buckets.setdefault(namespace, []).append(name)
        rows = []
        for namespace, cols in sorted(buckets.items()):
            cols_sorted = sorted(cols)
            examples = ", ".join(cols_sorted[:max_examples])
            rows.append({"namespace": namespace, "count": len(cols), "examples": examples})
        if not rows:
            return pl.DataFrame({"namespace": [], "count": [], "examples": []})
        return pl.DataFrame(rows)

    def missingness_summary(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return pl.DataFrame({"column": [], "null_pct": [], "non_null_count": []})
        total = df.height
        null_counts = df.null_count()
        rows = []
        for col in df.columns:
            nulls = int(null_counts[col][0])
            non_null = total - nulls
            pct = (nulls / total) * 100 if total else 0.0
            rows.append({"column": col, "null_pct": pct, "non_null_count": non_null})
        return pl.DataFrame(rows).sort("null_pct", descending=True)

    def opal_label_hist_columns(columns: Sequence[str]) -> list[str]:
        out = [c for c in columns if c.startswith("opal__") and c.endswith("__label_hist")]
        return sorted(out)

    def opal_labeled_mask(df: pl.DataFrame, label_hist_cols: Sequence[str]) -> pl.Series:
        if not label_hist_cols:
            return pl.Series([False] * df.height)
        exprs = [(pl.col(col).is_not_null()) & (pl.col(col).list.len().fill_null(0) > 0) for col in label_hist_cols]
        return df.select(pl.any_horizontal(exprs).alias("opal_labeled"))["opal_labeled"]

    def build_numeric_rule_exprs(rules: Sequence[NumericRule]) -> list[pl.Expr]:
        exprs: list[pl.Expr] = []
        for rule in rules:
            if not rule.enabled:
                continue
            if not rule.column:
                continue
            if rule.op not in SUPPORTED_NUMERIC_OPS:
                raise ValueError(f"Unsupported operator: {rule.op}")
            if rule.op == ">=":
                if rule.value is None:
                    continue
                exprs.append(pl.col(rule.column) >= float(rule.value))
            elif rule.op == "<=":
                if rule.value is None:
                    continue
                exprs.append(pl.col(rule.column) <= float(rule.value))
            elif rule.op == "is null":
                exprs.append(pl.col(rule.column).is_null())
            elif rule.op == "is not null":
                exprs.append(pl.col(rule.column).is_not_null())
        return exprs

    def apply_numeric_rules(df: pl.DataFrame, rules: Sequence[NumericRule]) -> pl.DataFrame:
        exprs = build_numeric_rule_exprs(rules)
        if not exprs:
            return df
        return df.filter(pl.all_horizontal(exprs))

    @dataclass(frozen=True)
    class SFXIParams:
        setpoint: tuple[float, float, float, float]
        weights: tuple[float, float, float, float]
        d: float
        beta: float
        gamma: float
        delta: float
        p: float
        fallback_p: float
        min_n: int
        eps: float

    @dataclass(frozen=True)
    class SFXIResult:
        df: pl.DataFrame
        denom: float
        weights: tuple[float, float, float, float]
        d: float
        pool_size: int
        denom_source: str

    def compute_sfxi_params(
        *,
        setpoint: Sequence[float],
        beta: float,
        gamma: float,
        delta: float,
        p: float,
        fallback_p: float,
        min_n: int,
        eps: float,
    ) -> SFXIParams:
        if len(setpoint) != 4:
            raise ValueError("setpoint must have length 4")
        p0, p1, p2, p3 = (float(x) for x in setpoint)
        total = p0 + p1 + p2 + p3
        if total <= eps:
            weights = (0.0, 0.0, 0.0, 0.0)
        else:
            weights = (p0 / total, p1 / total, p2 / total, p3 / total)
        d = math.sqrt(sum(max(v * v, (1.0 - v) * (1.0 - v)) for v in (p0, p1, p2, p3)))
        if d <= 0:
            d = eps
        return SFXIParams(
            setpoint=(p0, p1, p2, p3),
            weights=weights,
            d=d,
            beta=float(beta),
            gamma=float(gamma),
            delta=float(delta),
            p=float(p),
            fallback_p=float(fallback_p),
            min_n=int(min_n),
            eps=float(eps),
        )

    def _valid_vec8_mask_expr(vec_col: str) -> pl.Expr:
        vec = pl.col(vec_col)
        len_ok = vec.list.len() == 8
        finite_ok = vec.list.eval(pl.element().is_finite()).list.all()
        return vec.is_not_null() & len_ok & finite_ok

    def _effect_raw_expr(vec_col: str, weights: Sequence[float], delta: float) -> pl.Expr:
        y0 = pl.col(vec_col).list.get(4)
        y1 = pl.col(vec_col).list.get(5)
        y2 = pl.col(vec_col).list.get(6)
        y3 = pl.col(vec_col).list.get(7)
        y0_lin = pl.max_horizontal(pl.lit(0.0), (pl.lit(2.0) ** y0) - delta)
        y1_lin = pl.max_horizontal(pl.lit(0.0), (pl.lit(2.0) ** y1) - delta)
        y2_lin = pl.max_horizontal(pl.lit(0.0), (pl.lit(2.0) ** y2) - delta)
        y3_lin = pl.max_horizontal(pl.lit(0.0), (pl.lit(2.0) ** y3) - delta)
        return weights[0] * y0_lin + weights[1] * y1_lin + weights[2] * y2_lin + weights[3] * y3_lin

    def compute_sfxi_metrics(
        *,
        df: pl.DataFrame,
        vec_col: str,
        params: SFXIParams,
        denom_pool_df: pl.DataFrame,
    ) -> SFXIResult:
        if vec_col not in df.columns:
            return SFXIResult(
                df=df.head(0),
                denom=params.eps,
                weights=params.weights,
                d=params.d,
                pool_size=0,
                denom_source="empty",
            )
        valid_mask = _valid_vec8_mask_expr(vec_col)
        df_valid = df.filter(valid_mask)

        p0, p1, p2, p3 = params.setpoint
        v0 = pl.col(vec_col).list.get(0)
        v1 = pl.col(vec_col).list.get(1)
        v2 = pl.col(vec_col).list.get(2)
        v3 = pl.col(vec_col).list.get(3)
        dist = ((v0 - p0) ** 2 + (v1 - p1) ** 2 + (v2 - p2) ** 2 + (v3 - p3) ** 2) ** 0.5
        logic_fidelity = (1.0 - dist / params.d).clip(0.0, 1.0)

        effect_raw_expr = _effect_raw_expr(vec_col, params.weights, params.delta)

        pool_size = 0
        denom_source = "empty"
        denom = params.eps
        if vec_col in denom_pool_df.columns and not denom_pool_df.is_empty():
            pool_dtype = denom_pool_df.schema.get(vec_col, pl.Null)
            if pool_dtype != pl.Null:
                pool_valid = denom_pool_df.filter(_valid_vec8_mask_expr(vec_col))
                pool_effect = pool_valid.select(effect_raw_expr.alias("effect_raw"))
                pool_size = pool_effect.height
                if pool_size >= params.min_n:
                    denom = float(pool_effect["effect_raw"].quantile(params.p / 100.0, interpolation="nearest"))
                    denom_source = "p"
                elif pool_size > 0:
                    denom = float(
                        pool_effect["effect_raw"].quantile(params.fallback_p / 100.0, interpolation="nearest")
                    )
                    denom_source = "fallback_p"

        if not math.isfinite(denom) or denom <= 0:
            denom = params.eps
        denom = max(denom, params.eps)

        df_sfxi = (
            df_valid.with_columns(
                [
                    logic_fidelity.alias("logic_fidelity"),
                    effect_raw_expr.alias("effect_raw"),
                ]
            )
            .with_columns(((pl.col("effect_raw") / pl.lit(denom)).clip(0.0, 1.0).alias("effect_scaled")))
            .with_columns(
                ((pl.col("logic_fidelity") ** params.beta) * (pl.col("effect_scaled") ** params.gamma)).alias("score")
            )
        )

        return SFXIResult(
            df=df_sfxi,
            denom=denom,
            weights=params.weights,
            d=params.d,
            pool_size=pool_size,
            denom_source=denom_source,
        )

    return (
        compute_sfxi_metrics,
        compute_sfxi_params,
        find_repo_root,
        list_usr_datasets,
        missingness_summary,
        namespace_summary,
        resolve_dataset_path,
        resolve_usr_root,
    )


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
    # Promoter dataset explorer

    Explore DenseGen-generated promoter records with filters, Evo 2 UMAPs, OPAL/SFXI
    fields, and per-sequence inspection.

    What this notebook expects
    - Records are DenseGen-generated promoter sequences.
    - TFBS annotations live in `densegen__used_tfbs_detail` when available.
    - OPAL/SFXI columns may or may not be present.
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

    try:
        dataset_path, dataset_mode = resolve_dataset_path(
            repo_root=repo_root,
            usr_root=usr_root,
            dataset_name=dataset_dropdown.value,
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
    df_prelim = df_raw
    return (df_prelim,)


@app.cell
def _(df_prelim, mo):
    _unused = (df_prelim, mo)
    return


@app.cell
def _(df_prelim, mo, namespace_summary):
    namespace_summary_df = namespace_summary(df_prelim.columns)
    namespace_table = mo.ui.table(namespace_summary_df)
    return (namespace_table,)


@app.cell
def _(df_prelim, missingness_summary, mo):
    missingness_df = missingness_summary(df_prelim)
    missingness_table = mo.ui.table(missingness_df)
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
    schema_table = mo.ui.table(schema_df)
    return (schema_table,)


@app.cell
def _(df_prelim, mo):
    dataset_table = mo.ui.table(df_prelim)
    return (dataset_table,)


@app.cell
def _(df_prelim, mo):
    numeric_cols = [name for name, dtype in df_prelim.schema.items() if dtype.is_numeric()]
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
    rows_slider = mo.ui.slider(100, 20000, value=5000, label="Rows to plot (max 20,000)")
    bins_slider = mo.ui.slider(5, 200, value=30, label="Histogram bins")
    return bins_slider, plot_type_dropdown, rows_slider, x_dropdown, y_dropdown


@app.cell
def _(
    alt,
    bins_slider,
    df_prelim,
    mo,
    pl,
    plot_type_dropdown,
    rows_slider,
    x_dropdown,
    y_dropdown,
):
    plot_type = plot_type_dropdown.value
    x_col = x_dropdown.value
    y_col = y_dropdown.value
    note_md = mo.md("")

    if plot_type == "scatter":
        if x_col not in df_prelim.columns or y_col not in df_prelim.columns:
            note_md = mo.md("Select numeric X and Y columns for scatter.")
            df_plot = pl.DataFrame({"x": [], "y": []})
            chart = (
                alt.Chart(df_plot)
                .mark_circle(opacity=0.7, stroke=None, strokeWidth=0)
                .encode(x=alt.X("x"), y=alt.Y("y"))
            )
        else:
            if x_col == y_col:
                y_col_plot = f"{y_col}__y"
                df_plot = df_prelim.select([pl.col(x_col), pl.col(y_col).alias(y_col_plot)]).head(
                    int(rows_slider.value)
                )
            else:
                y_col_plot = y_col
                df_plot = df_prelim.select([x_col, y_col]).head(int(rows_slider.value))
            _brush = alt.selection_interval(name="explorer_brush", encodings=["x", "y"])
            chart = (
                alt.Chart(df_plot)
                .mark_circle(opacity=0.7, stroke=None, strokeWidth=0)
                .encode(
                    x=alt.X(x_col, title=x_col),
                    y=alt.Y(y_col_plot, title=y_col),
                )
                .add_params(_brush)
            )
    else:
        if x_col not in df_prelim.columns:
            note_md = mo.md("Select a numeric column for the histogram.")
            df_plot = pl.DataFrame({"x": []})
            chart = alt.Chart(df_plot).mark_bar().encode(x=alt.X("x", bin=alt.Bin(maxbins=10)), y=alt.Y("count()"))
        else:
            df_plot = df_prelim.select([x_col]).head(int(rows_slider.value))
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

    chart = chart.properties(width=600, height=360)
    dataset_explorer_chart_ui = mo.ui.altair_chart(chart)
    dataset_explorer = mo.vstack(
        [
            mo.hstack([plot_type_dropdown, rows_slider]),
            mo.hstack([x_dropdown, y_dropdown]),
            bins_slider,
            note_md,
            dataset_explorer_chart_ui,
        ]
    )
    return dataset_explorer, dataset_explorer_chart_ui


@app.cell
def _(dataset_explorer_chart_ui, df_prelim, mo, pl, x_dropdown, y_dropdown):
    _selected_raw = dataset_explorer_chart_ui.value

    if _selected_raw is None:
        df_explorer_selected = df_prelim.head(0)
    elif isinstance(_selected_raw, pl.DataFrame):
        df_explorer_selected = _selected_raw
    else:
        df_explorer_selected = pl.from_pandas(_selected_raw)

    if df_prelim.height > 0 and df_explorer_selected.height == df_prelim.height:
        df_explorer_selected = df_prelim.head(0)
        explorer_selection_note_md = mo.md("No selection.")
    else:
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
def _(
    dataset_form,
    dataset_preview_md,
    mo,
    usr_root,
    usr_root_error,
    usr_root_mode,
):
    header = mo.md("## Import and dataset selection")
    usr_status = mo.md("")
    if usr_root_error:
        usr_status = mo.md(f"USR root error: {usr_root_error}")
    elif usr_root:
        usr_status = mo.md(f"USR root ({usr_root_mode}): `{usr_root}`")
    mo.vstack([header, usr_status, dataset_form, dataset_preview_md])
    return


@app.cell
def _(missingness_table, mo, namespace_table, schema_table):
    mo.vstack(
        [
            mo.md(
                "### Column namespaces\n"
                "Columns are grouped by the prefix before `__` and counted to show which "
                "tools/pipelines contribute fields. Includes a few example columns per namespace."
            ),
            namespace_table,
            mo.md(
                "### Missing data\n"
                "Null percentage and non-null counts per column, sorted with the emptiest columns first "
                "to surface gaps quickly."
            ),
            missingness_table,
            mo.md(
                "### Schema & dtypes\n"
                "Every column with its Polars dtype—useful for debugging types and choosing fields for plots/filters."
            ),
            schema_table,
        ]
    )
    return


@app.cell
def _(df_active, mo, pl):
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
    umap_color_cols = [
        name
        for name, dtype in df_active.schema.items()
        if dtype.is_numeric() or dtype in (pl.String, pl.Categorical, pl.Boolean, pl.Enum, pl.Date, pl.Datetime)
    ]
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
    _unused = (alt, pl)
    return


@app.cell
def _(
    alt,
    df_active,
    mo,
    pl,
    umap_color_dropdown,
    umap_opacity_slider,
    umap_size_slider,
    umap_x_input,
    umap_y_input,
):
    _x_col = umap_x_input.value.strip()
    _y_col = umap_y_input.value.strip()
    umap_valid = False
    _x_name = _x_col or "umap_x"
    _y_name = _y_col or "umap_y"
    umap_note = None

    if not _x_col or not _y_col:
        umap_note = mo.md(
            "UMAP missing: provide x/y columns. "
            "To attach coords: `uv run cluster umap --dataset <dataset> "
            "--name ldn_v1 --attach-coords --write --allow-overwrite`"
        )
    elif _x_col not in df_active.columns or _y_col not in df_active.columns:
        umap_note = mo.md(
            "UMAP missing: x/y columns must exist. "
            "To attach coords: `uv run cluster umap --dataset <dataset> "
            "--name ldn_v1 --attach-coords --write --allow-overwrite`"
        )
    elif not (df_active.schema[_x_col].is_numeric() and df_active.schema[_y_col].is_numeric()):
        umap_note = mo.md(
            "UMAP missing: x/y columns must be numeric. "
            "To attach coords: `uv run cluster umap --dataset <dataset> "
            "--name ldn_v1 --attach-coords --write --allow-overwrite`"
        )
    else:
        umap_valid = True

    if not umap_valid:
        df_umap_plot = pl.DataFrame({_x_name: [], _y_name: []})
        _chart = (
            alt.Chart(df_umap_plot)
            .mark_circle(stroke=None, strokeWidth=0)
            .encode(x=alt.X(_x_name), y=alt.Y(_y_name))
            .properties(width=600, height=360)
        )
        umap_chart_note_md = umap_note
    else:
        df_umap_plot = df_active
        umap_chart_note_md = mo.md(f"Plotting full dataset: `{df_umap_plot.height}` points.")

        _color_value = umap_color_dropdown.value
        _color_encoding = alt.Undefined
        if _color_value and _color_value != "(none)" and _color_value in df_umap_plot.columns:
            _dtype = df_umap_plot.schema[_color_value]
            if _dtype.is_numeric():
                _color_encoding = alt.Color(_color_value, type="quantitative")
            else:
                _color_encoding = alt.Color(_color_value, type="nominal")

        _brush = alt.selection_interval(name="umap_brush", encodings=["x", "y"])
        _chart = (
            alt.Chart(df_umap_plot)
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
                tooltip=[c for c in ["id", _x_col, _y_col] if c in df_umap_plot.columns],
            )
            .add_params(_brush)
            .properties(width=600, height=360)
        )

    umap_chart_ui = mo.ui.altair_chart(_chart)
    return df_umap_plot, umap_chart_note_md, umap_chart_ui, umap_valid


@app.cell
def _(
    df_umap_plot,
    mo,
    pl,
    umap_chart_ui,
    umap_valid,
    umap_x_input,
    umap_y_input,
):
    _selected_raw = umap_chart_ui.value

    if not umap_valid:
        df_umap_selected = df_umap_plot.head(0)
        umap_selection_note_md = mo.md("UMAP missing; selection disabled.")
    else:
        if _selected_raw is None:
            df_umap_selected = df_umap_plot
        elif isinstance(_selected_raw, pl.DataFrame):
            df_umap_selected = _selected_raw
        else:
            df_umap_selected = pl.from_pandas(_selected_raw)

        if df_umap_plot.height > 0 and df_umap_selected.height == df_umap_plot.height:
            umap_selection_note_md = mo.md("No selection; showing full UMAP view.")
        else:
            umap_selection_note_md = mo.md(f"Selected rows: `{df_umap_selected.height}`")

    umap_selected_explorer = mo.ui.table(df_umap_selected)
    return df_umap_selected, umap_selected_explorer, umap_selection_note_md


@app.cell(column=1)
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
            dataset_status_md,
            mo.md("Full dataset view (Polars table) for quick inspection of raw records and columns."),
            dataset_table,
            mo.md("Ad-hoc exploratory data analysis using a Polars-backed chart explorer."),
            dataset_explorer,
            mo.md("### Selected rows (from explorer brush)"),
            explorer_selection_note_md,
            explorer_selected_table,
            df_active_note_md,
        ]
    )
    return


@app.cell(column=2)
def _(
    baserender_error_md,
    id_dropdown,
    id_override_input,
    id_selector_note_md,
    inspect_pool_dropdown,
    inspect_view_mode_dropdown,
    mo,
    render_element,
    save_pdf_button,
    save_pdf_status_md,
    sequence_md,
    summary_md,
    tfbs_note_md,
    tfbs_table_ui,
):
    mo.vstack(
        [
            mo.md("## Inspect a sequence"),
            inspect_pool_dropdown,
            id_override_input,
            id_dropdown,
            id_selector_note_md,
            summary_md,
            sequence_md,
            inspect_view_mode_dropdown,
            baserender_error_md,
            render_element,
            tfbs_note_md,
            tfbs_table_ui,
            save_pdf_button,
            save_pdf_status_md,
        ]
    )
    return


@app.cell
def _(mo):
    export_source_dropdown = mo.ui.dropdown(
        options=[
            "df_active",
            "df_umap_selected",
            "df_sfxi_selected",
            "active_record",
        ],
        value="df_active",
        label="Export source",
        full_width=True,
    )
    export_format_dropdown = mo.ui.dropdown(
        options=["csv", "parquet"],
        value="parquet",
        label="Format",
    )
    export_button = mo.ui.run_button(label="Export")
    return export_button, export_format_dropdown, export_source_dropdown


@app.cell
def _(
    active_record,
    df_active,
    df_sfxi_selected,
    df_umap_selected,
    export_button,
    export_format_dropdown,
    export_source_dropdown,
    mo,
    repo_root,
):
    export_status_md = None

    if export_button.value:
        source = export_source_dropdown.value
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
            export_status_md = mo.md(f"Exported to `{out_path}`")
    return (export_status_md,)


@app.cell
def _(
    export_button,
    export_format_dropdown,
    export_source_dropdown,
    export_status_md,
    mo,
):
    mo.vstack(
        [
            mo.md("## Optional - Export"),
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

    try:
        preview_path, preview_mode = resolve_dataset_path(
            repo_root=repo_root,
            usr_root=usr_root,
            dataset_name=dataset_dropdown.value,
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
    compute_sfxi_metrics,
    compute_sfxi_params,
    df_active,
    df_raw,
    mo,
    sfxi_beta_input,
    sfxi_delta_input,
    sfxi_eps_input,
    sfxi_fallback_p_input,
    sfxi_gamma_input,
    sfxi_min_n_input,
    sfxi_p00_slider,
    sfxi_p01_slider,
    sfxi_p10_slider,
    sfxi_p11_slider,
    sfxi_p_input,
    sfxi_pool_dropdown,
):
    if "sfxi_8_vector_y_label" not in df_active.columns:
        df_sfxi = df_active.head(0)
        sfxi_meta_md = mo.md("No SFXI labels: missing sfxi_8_vector_y_label.")
        sfxi_notice_md = sfxi_meta_md
    else:
        params = compute_sfxi_params(
            setpoint=[
                sfxi_p00_slider.value,
                sfxi_p10_slider.value,
                sfxi_p01_slider.value,
                sfxi_p11_slider.value,
            ],
            beta=sfxi_beta_input.value,
            gamma=sfxi_gamma_input.value,
            delta=sfxi_delta_input.value,
            p=sfxi_p_input.value,
            fallback_p=sfxi_fallback_p_input.value,
            min_n=int(sfxi_min_n_input.value),
            eps=sfxi_eps_input.value,
        )

        denom_pool_df = df_active if sfxi_pool_dropdown.value == "df_active" else df_raw
        sfxi_result = compute_sfxi_metrics(
            df=df_active,
            vec_col="sfxi_8_vector_y_label",
            params=params,
            denom_pool_df=denom_pool_df,
        )

        df_sfxi = sfxi_result.df
        sfxi_meta_md = mo.md(
            "\n".join(
                [
                    f"Denom: {sfxi_result.denom:.4g}",
                    f"Pool size: {sfxi_result.pool_size}",
                    f"Denom source: {sfxi_result.denom_source}",
                ]
            )
        )
        sfxi_notice_md = mo.md("")
        if df_sfxi.is_empty():
            sfxi_notice_md = mo.md("No valid SFXI vectors after filtering.")
    return df_sfxi, sfxi_meta_md, sfxi_notice_md


@app.cell(column=3)
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
            mo.md("## Evo 2 - UMAP explorer"),
            umap_x_input,
            umap_y_input,
            umap_color_dropdown,
            umap_size_slider,
            umap_opacity_slider,
            umap_chart_note_md,
            umap_chart_ui,
            umap_selection_note_md,
            umap_selected_explorer,
        ]
    )
    return


@app.cell
def _(df_active, mo):
    opal_panel_prefixes = sorted(
        {col.split("__", 2)[1] for col in df_active.columns if col.startswith("opal__") and col.count("__") >= 2}
    )
    opal_panel_prefix_dropdown = None
    opal_panel_prefix_default = None
    if opal_panel_prefixes:
        opal_panel_prefix_default = opal_panel_prefixes[0]
        opal_panel_prefix_dropdown = mo.ui.dropdown(
            options=opal_panel_prefixes,
            value=opal_panel_prefix_default,
            label="OPAL prefix",
            full_width=True,
        )
    opal_panel_note_md = mo.md(
        f"OPAL prefixes: {opal_panel_prefixes}" if opal_panel_prefixes else "No OPAL columns found."
    )
    return (
        opal_panel_note_md,
        opal_panel_prefix_default,
        opal_panel_prefix_dropdown,
    )


@app.cell
def _(
    alt,
    df_active,
    mo,
    opal_panel_prefix_default,
    opal_panel_prefix_dropdown,
):
    if opal_panel_prefix_dropdown is None:
        opal_panel_content = mo.md("No OPAL columns available.")
    else:
        prefix = opal_panel_prefix_dropdown.value or opal_panel_prefix_default
        label_hist_col = f"opal__{prefix}__label_hist"
        latest_pred_col = f"opal__{prefix}__latest_pred_scalar"
        latest_round_col = f"opal__{prefix}__latest_as_of_round"

        lines = [f"Using OPAL prefix: `{prefix}`"]
        available_cols = [c for c in [label_hist_col, latest_pred_col, latest_round_col] if c in df_active.columns]
        lines.append(f"Available OPAL cols: {available_cols}")

        charts = []
        if latest_pred_col in df_active.columns:
            _chart = (
                alt.Chart(df_active)
                .mark_bar()
                .encode(
                    x=alt.X(latest_pred_col, bin=alt.Bin(maxbins=30), title=latest_pred_col),
                    y=alt.Y("count()", title="count"),
                )
                .properties(width=300, height=200, title="OPAL latest_pred_scalar")
            )
            charts.append(_chart)
        if latest_round_col in df_active.columns:
            _chart = (
                alt.Chart(df_active)
                .mark_bar()
                .encode(
                    x=alt.X(latest_round_col, title=latest_round_col),
                    y=alt.Y("count()", title="count"),
                )
                .properties(width=300, height=200, title="OPAL latest_as_of_round")
            )
            charts.append(_chart)

        if charts:
            charts_row = mo.hstack([mo.ui.altair_chart(c) for c in charts])
        else:
            charts_row = mo.md("No OPAL summary charts available for this prefix.")

        opal_panel_content = mo.vstack([mo.md("\n".join(lines)), charts_row])
    return (opal_panel_content,)


@app.cell
def _(mo):
    sfxi_p00_slider = mo.ui.slider(0.0, 1.0, value=0.9, label="p00", step=0.01)
    sfxi_p10_slider = mo.ui.slider(0.0, 1.0, value=0.1, label="p10", step=0.01)
    sfxi_p01_slider = mo.ui.slider(0.0, 1.0, value=0.1, label="p01", step=0.01)
    sfxi_p11_slider = mo.ui.slider(0.0, 1.0, value=0.9, label="p11", step=0.01)
    sfxi_beta_input = mo.ui.number(value=1.0, label="beta")
    sfxi_gamma_input = mo.ui.number(value=1.0, label="gamma")
    sfxi_delta_input = mo.ui.number(value=1.0, label="delta")
    sfxi_p_input = mo.ui.number(value=95.0, label="percentile p")
    sfxi_fallback_p_input = mo.ui.number(value=75.0, label="fallback percentile")
    sfxi_min_n_input = mo.ui.number(value=10, label="min_n")
    sfxi_eps_input = mo.ui.number(value=1e-6, label="eps")
    sfxi_pool_dropdown = mo.ui.dropdown(
        options=["df_active", "df_raw"],
        value="df_active",
        label="Denominator pool",
        full_width=True,
    )
    return (
        sfxi_beta_input,
        sfxi_delta_input,
        sfxi_eps_input,
        sfxi_fallback_p_input,
        sfxi_gamma_input,
        sfxi_min_n_input,
        sfxi_p00_slider,
        sfxi_p01_slider,
        sfxi_p10_slider,
        sfxi_p11_slider,
        sfxi_p_input,
        sfxi_pool_dropdown,
    )


@app.cell
def _(alt, df_sfxi, mo, pl):
    if df_sfxi.is_empty():
        empty_df = pl.DataFrame({"logic_fidelity": [], "effect_scaled": [], "score": []})
        _chart = (
            alt.Chart(empty_df)
            .mark_circle(opacity=0.7, stroke=None, strokeWidth=0)
            .encode(
                x=alt.X("logic_fidelity", title="Logic fidelity"),
                y=alt.Y("effect_scaled", title="Effect (scaled)"),
                color=alt.Color("score", type="quantitative"),
            )
            .properties(width=600, height=360)
        )
        sfxi_chart_note_md = mo.md("No SFXI data to plot.")
    else:
        _brush = alt.selection_interval(name="sfxi_brush")
        _chart = (
            alt.Chart(df_sfxi)
            .mark_circle(opacity=0.7, stroke=None, strokeWidth=0)
            .encode(
                x=alt.X("logic_fidelity", title="Logic fidelity"),
                y=alt.Y("effect_scaled", title="Effect (scaled)"),
                color=alt.Color("score", type="quantitative"),
                tooltip=["id", "logic_fidelity", "effect_scaled", "score"],
            )
            .add_params(_brush)
            .properties(width=600, height=360)
        )
        sfxi_chart_note_md = mo.md("Brush to select points.")

    sfxi_chart_ui = mo.ui.altair_chart(_chart)
    return sfxi_chart_note_md, sfxi_chart_ui


@app.cell
def _(df_sfxi, mo, pl, sfxi_chart_ui):
    _selected_raw = sfxi_chart_ui.value

    if _selected_raw is None:
        df_sfxi_selected = df_sfxi
    elif isinstance(_selected_raw, pl.DataFrame):
        df_sfxi_selected = _selected_raw
    else:
        df_sfxi_selected = pl.from_pandas(_selected_raw)

    if df_sfxi.height > 0 and df_sfxi_selected.height == df_sfxi.height:
        sfxi_selected_note_md = mo.md("No selection; showing full SFXI view.")
    else:
        sfxi_selected_note_md = mo.md(f"Selected rows: `{df_sfxi_selected.height}`")

    sfxi_selected_explorer = mo.ui.table(df_sfxi_selected)
    return df_sfxi_selected, sfxi_selected_explorer, sfxi_selected_note_md


@app.cell(column=4)
def _(
    mo,
    opal_panel_content,
    opal_panel_note_md,
    opal_panel_prefix_dropdown,
    sfxi_beta_input,
    sfxi_chart_note_md,
    sfxi_chart_ui,
    sfxi_delta_input,
    sfxi_eps_input,
    sfxi_fallback_p_input,
    sfxi_gamma_input,
    sfxi_meta_md,
    sfxi_min_n_input,
    sfxi_notice_md,
    sfxi_p00_slider,
    sfxi_p01_slider,
    sfxi_p10_slider,
    sfxi_p11_slider,
    sfxi_p_input,
    sfxi_pool_dropdown,
    sfxi_selected_explorer,
    sfxi_selected_note_md,
):
    sfxi_controls = mo.hstack(
        [
            mo.vstack([sfxi_p00_slider, sfxi_p10_slider, sfxi_p01_slider, sfxi_p11_slider]),
            mo.vstack([sfxi_beta_input, sfxi_gamma_input, sfxi_delta_input]),
            mo.vstack(
                [
                    sfxi_p_input,
                    sfxi_fallback_p_input,
                    sfxi_min_n_input,
                    sfxi_eps_input,
                    sfxi_pool_dropdown,
                ]
            ),
        ]
    )

    opal_controls = []
    if opal_panel_prefix_dropdown is not None:
        opal_controls = [opal_panel_prefix_dropdown]

    mo.vstack(
        [
            mo.md("## OPAL and SFXI"),
            mo.md("### OPAL overview"),
            opal_panel_note_md,
            *opal_controls,
            opal_panel_content,
            mo.md("### SFXI scoring"),
            sfxi_controls,
            sfxi_meta_md,
            sfxi_notice_md,
            sfxi_chart_note_md,
            sfxi_chart_ui,
            sfxi_selected_note_md,
            sfxi_selected_explorer,
        ]
    )
    return


@app.cell
def _(mo):
    inspect_pool_dropdown = mo.ui.dropdown(
        options=[
            "df_active",
            "df_umap_selected",
            "df_sfxi_selected",
        ],
        value="df_active",
        label="Record pool",
        full_width=True,
    )
    return (inspect_pool_dropdown,)


@app.cell
def _(df_active, df_sfxi_selected, df_umap_selected, inspect_pool_dropdown):
    pool_choice = inspect_pool_dropdown.value
    if pool_choice == "df_umap_selected":
        df_pool = df_umap_selected
    elif pool_choice == "df_sfxi_selected":
        df_pool = df_sfxi_selected
    else:
        df_pool = df_active
    df_pool_label = pool_choice
    return df_pool, df_pool_label


@app.cell
def _(df_pool, mo):
    id_override_input = mo.ui.text(
        value="",
        label="ID override (exact or substring)",
        full_width=True,
    )

    id_values = []
    if "id" in df_pool.columns:
        id_values = df_pool.select("id").drop_nulls().to_series().to_list()

    max_ids = 1000
    truncated = len(id_values) > max_ids
    id_values = id_values[:max_ids]
    id_dropdown = mo.ui.dropdown(
        options=id_values or ["(none)"],
        value=id_values[0] if id_values else "(none)",
        label="Record id",
        full_width=True,
    )
    id_selector_note_md = mo.md(f"ID list truncated to {max_ids}." if truncated else "")
    return id_dropdown, id_override_input, id_selector_note_md


@app.cell
def _(df_pool, id_dropdown, id_override_input, mo, pl):
    active_record_id = None
    active_record = None

    if not df_pool.is_empty() and "id" in df_pool.columns:
        override = id_override_input.value.strip()
        if override:
            exact = df_pool.filter(pl.col("id") == override)
            if exact.height:
                active_record_id = exact["id"][0]
            else:
                matches = df_pool.filter(pl.col("id").cast(pl.Utf8).str.contains(override))
                if matches.height:
                    active_record_id = matches["id"][0]
        else:
            active_record_id = id_dropdown.value if id_dropdown.value != "(none)" else None

        if active_record_id is None:
            pass
        else:
            active_record = df_pool.filter(pl.col("id") == active_record_id).head(1)
            if active_record.is_empty():
                active_record = None
    return active_record, active_record_id


@app.cell
def _(active_record, active_record_id, df_pool_label, mo):
    if active_record is None or active_record.is_empty():
        summary_md = mo.md("No active record to display.")
        sequence_md = mo.md("")
    else:
        fields = ["id", "densegen__plan", "cluster__ldn_v1", "bio_type", "created_at"]
        summary = {}
        for field in fields:
            if field in active_record.columns:
                summary[field] = active_record[field][0]
        summary_md = mo.md(
            "\n".join(
                [
                    f"Pool: `{df_pool_label}`",
                    f"Active id: `{active_record_id}`",
                    f"Summary: {summary}",
                ]
            )
        )

        seq_text = ""
        if "sequence" in active_record.columns:
            seq_text = active_record["sequence"][0]
        sequence_md = mo.md(f"Sequence:\n\n`{seq_text}`" if seq_text else "Sequence column missing.")
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
    return (inspect_view_mode_dropdown,)


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

                fig = _baserender_api.render_image(_record, fmt="png")
                render_element = fig
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
                render_element = mo.md(active_record["densegen__visual"][0])
            else:
                render_element = mo.md("densegen__visual not available.")
    return baserender_error_md, render_element


@app.cell
def _(mo):
    save_pdf_button = mo.ui.run_button(label="Save still (PDF)")
    return (save_pdf_button,)


@app.cell
def _(
    active_record,
    dataset_name,
    dataset_path,
    mo,
    repo_root,
    save_pdf_button,
):
    save_pdf_status_md = None
    save_pdf_path = None

    if save_pdf_button.value:
        if active_record is None or active_record.is_empty():
            save_pdf_status_md = mo.md("No active record to save.")
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

                _baserender_out_dir = repo_root / "src" / "dnadesign" / "opal" / "notebooks" / "_outputs" / "baserender"
                _baserender_out_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{dataset_name}__{_active_id}__baserender.pdf"
                save_pdf_path = _baserender_out_dir / filename

                _baserender_api.render_image(_record, out_path=save_pdf_path, fmt="pdf")
                save_pdf_status_md = mo.md(f"Saved PDF: `{save_pdf_path}`")
            except Exception as exc:
                save_pdf_status_md = mo.md(f"Failed to save PDF: {exc}")
    return (save_pdf_status_md,)


if __name__ == "__main__":
    app.run()
